import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import threading

from config import config
import notification_service
import utils
import market_data_processor

logger = logging.getLogger(__name__)

class DetectorMonitor:
    def __init__(self):
        self._detector_health_status = defaultdict(lambda: {
            "last_run_time_utc": None,
            "last_successful_run_time_utc": None,
            "last_error_time_utc": None,
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "detected_count_last_run": 0,
            "status": "Not Run",
            "error_message_last_run": None,
            "warning_message_last_run": None,
            "zero_detections_count": 0,
            "consecutive_zero_detections": 0
        })
        self._detector_health_lock = threading.Lock()
        self._detector_health_status = {} # {detector_name: {"status": "OK/Warning/Failed", "last_run_time_utc": datetime, "error_message_last_run": str, "warning_message_last_run": str, "consecutive_zero_detections": int}}
        self._detector_health_lock = threading.Lock() # Untuk keamanan thread
        self._last_health_report_time = {} # {detector_name: datetime}


    # PERUBAHAN DI SINI: Gunakan *args dan **kwargs
    def wrap_detector_function(self, func, *args, **kwargs):
        """
        Membungkus fungsi detektor untuk memantau status kesehatannya.
        """
        detector_name = func.__name__ # Nama fungsi sebagai nama detektor
        
        with self._detector_health_lock:
            # PERBAIKAN PENTING DI SINI: Inisialisasi status detektor jika belum ada
            if detector_name not in self._detector_health_status:
                self._detector_health_status[detector_name] = {
                    "status": "Not Run",
                    "last_run_time_utc": None,
                    "error_message_last_run": None,
                    "warning_message_last_run": None,
                    "consecutive_zero_detections": 0,
                    "total_runs": 0,
                    "total_successes": 0,
                    "total_failures": 0,
                    "last_detection_count": 0, # Tambahkan ini untuk melacak jumlah deteksi terakhir
                    "last_success_time_utc": None, # Tambahkan ini
                    "last_failure_time_utc": None # Tambahkan ini
                }
            
            # Perbarui metrik sebelum menjalankan fungsi
            self._detector_health_status[detector_name]["total_runs"] += 1 # <<< Ini adalah baris 50 yang error
            self._detector_health_status[detector_name]["status"] = "Running"
            self._detector_health_status[detector_name]["last_run_time_utc"] = datetime.now(timezone.utc)

        result = None
        try:
            result = func(*args, **kwargs)
            
            with self._detector_health_lock:
                self._detector_health_status[detector_name]["status"] = "OK"
                self._detector_health_status[detector_name]["total_successes"] += 1
                self._detector_health_status[detector_name]["last_success_time_utc"] = datetime.now(timezone.utc)
                self._detector_health_status[detector_name]["error_message_last_run"] = None # Bersihkan pesan error
                
                # Perbarui last_detection_count jika fungsi mengembalikan jumlah deteksi
                if isinstance(result, int): # Jika fungsi detektor mengembalikan int (jumlah deteksi)
                    self._detector_health_status[detector_name]["last_detection_count"] = result
                    if result == 0:
                        self._detector_health_status[detector_name]["consecutive_zero_detections"] += 1
                        self._detector_health_status[detector_name]["warning_message_last_run"] = "0 deteksi."
                        if self._detector_health_status[detector_name]["consecutive_zero_detections"] >= getattr(config.Monitoring, 'DETECTOR_ZERO_DETECTIONS_THRESHOLD', 3):
                            self._detector_health_status[detector_name]["status"] = "Warning" # Set status ke Warning
                    else:
                        self._detector_health_status[detector_name]["consecutive_zero_detections"] = 0 # Reset jika ada deteksi
                        self._detector_health_status[detector_name]["warning_message_last_run"] = None
                else: # Jika fungsi tidak mengembalikan int, asumsikan sukses > 0
                    self._detector_health_status[detector_name]["consecutive_zero_detections"] = 0
                    self._detector_health_status[detector_name]["warning_message_last_run"] = None

        except Exception as e:
            with self._detector_health_lock:
                self._detector_health_status[detector_name]["status"] = "Failed"
                self._detector_health_status[detector_name]["total_failures"] += 1
                self._detector_health_status[detector_name]["last_failure_time_utc"] = datetime.now(timezone.utc)
                self._detector_health_status[detector_name]["error_message_last_run"] = str(e)
                logger.error(f"Detector '{detector_name}' FAILED: {e}", exc_info=True)
                # Notifikasi error mungkin ditangani oleh loop scheduler, atau di sini jika error sangat kritis.
                # Untuk saat ini, biarkan scheduler loop menangani notifikasi error umum.
        return result
    
    def get_health_status(self):
        with self._detector_health_lock:
            return self._detector_health_status.copy()

    def send_health_report(self): # <<< PASTIKAN METODE INI ADA DAN KONSISTEN
        """
        Mengirim laporan kesehatan detektor saat ini ke Telegram.
        Hanya mengirim notifikasi untuk anomali.
        """
        anomalies_found = []
        current_time = datetime.now(timezone.utc)
        
        with self._detector_health_lock:
            # Periksa setiap detektor
            for detector_name, status_data in self._detector_health_status.items():
                last_notified = self._last_health_report_time.get(detector_name, datetime.min.replace(tzinfo=timezone.utc))
                
                # Check cooldown notification per detector
                telegram_cooldown = getattr(config.Telegram, 'NOTIFICATION_COOLDOWN_SECONDS', 3600) # Default 1 jam
                if (current_time - last_notified).total_seconds() < telegram_cooldown:
                    continue # Skip if still in cooldown

                # Kriteria Deteksi Anomali (sesuai dengan scheduler.py sebelumnya)
                if status_data["status"] == "Failed":
                    anomalies_found.append(f"🔴 Detektor *{detector_name}* GAGAL! Error: `{utils._escape_markdown(status_data.get('error_message_last_run', 'Tidak ada pesan'))}`. Cek log lebih lanjut.")
                    self._last_health_report_time[detector_name] = current_time # Update waktu notifikasi

                elif status_data["status"] == "Warning" and status_data["consecutive_zero_detections"] >= getattr(config.Monitoring, 'DETECTOR_ZERO_DETECTIONS_THRESHOLD', 3):
                    anomalies_found.append(f"🟡 Detektor *{detector_name}* Peringatan: {status_data['consecutive_zero_detections']} run berturut-turut dengan 0 deteksi. Pesan: `{utils._escape_markdown(status_data.get('warning_message_last_run', 'Tidak ada pesan'))}`.")
                    self._last_health_report_time[detector_name] = current_time # Update waktu notifikasi

                elif status_data["status"] == "Not Run" and \
                    (status_data["last_run_time_utc"] is None or \
                     (current_time - status_data["last_run_time_utc"]).total_seconds() > config.Monitoring.DETECTOR_HEALTH_CHECK_INTERVAL_SECONDS * 1.5):
                    anomalies_found.append(f"⚪ Detektor *{detector_name}* belum pernah berjalan atau terlalu lama tidak aktif ({status_data['status']}).")
                    self._last_health_report_time[detector_name] = current_time # Update waktu notifikasi

        # Jika ada anomali yang ditemukan, kirim notifikasi Telegram
        if anomalies_found:
            message = "⚠️ *Anomali Detektor Terdeteksi:*\n\n" + "\n".join(anomalies_found)
            notification_service.send_telegram_message(message)
            logger.warning(f"Mengirim notifikasi anomali detektor: {len(anomalies_found)} anomali.")
        else:
            logger.info("Tidak ada anomali detektor yang terdeteksi.")

# Instance global dari DetectorMonitor
detector_monitor = DetectorMonitor() # <<< PASTIKAN BARIS INI ADA DI AKHIR FILE

            
