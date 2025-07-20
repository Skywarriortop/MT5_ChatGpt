# scheduler.py (Fungsi periodic_mt5_trade_data_update_loop yang Dimodifikasi)

# Pastikan semua import ini ada di bagian atas file scheduler.py
import threading
import time
import sys
from datetime import datetime, timezone, timedelta
from config import config
import logging
from collections import defaultdict
import data_updater # Pastikan ini diimpor
import ai_consensus_manager
import rule_based_signal_generator
import notification_service # Pastikan ini diimpor
import database_manager # Pastikan ini diimpor
import market_data_processor
import detector_monitoring
# from ai_consensus_manager import _format_price # Import _format_price dari ai_consensus_manager
import fundamental_data_service
import utils
import json
from decimal import Decimal
import chart_generator
import pandas as pd
import mt5_connector
import auto_trade_manager # Pastikan ini diimpor

logger = logging.getLogger(__name__)

# Tambahkan variabel global untuk mengontrol apakah fitur bergantung sudah bisa berjalan
_feature_backfill_completed = False

def _json_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# --- Scheduler Task Functions (Self-Looping) ---

def periodic_session_data_update_loop(stop_event):
    """Loop untuk memperbarui data sesi secara periodik."""
    interval = config.Scheduler.UPDATE_INTERVALS.get('periodic_session_data_update_loop', 300)
    while not stop_event.is_set():
        try:
            # Pastikan current_mt5_datetime_for_session_update menggunakan waktu UTC aware
            current_mt5_datetime_for_session_update = utils.to_utc_datetime_or_none(datetime.now(timezone.utc))
            latest_tick_from_db = database_manager.get_latest_price_tick(config.TRADING_SYMBOL)
            if latest_tick_from_db and latest_tick_from_db.get('time'):
                tick_dt = utils.to_utc_datetime_or_none(latest_tick_from_db['time'])
                if tick_dt and (datetime.now(timezone.utc) - tick_dt).total_seconds() < 3600:
                    current_mt5_datetime_for_session_update = tick_dt
                else:
                    logger.warning("Waktu tick terakhir dari DB terlalu lama atau tidak valid. Menggunakan waktu sistem untuk update sesi.")
            else:
                current_mt5_tick_info_raw = mt5_connector.get_current_tick_info(config.TRADING_SYMBOL)
                if current_mt5_tick_info_raw and current_mt5_tick_info_raw.get('time_msc'):
                    current_mt5_datetime_for_session_update = utils.to_utc_datetime_or_none(current_mt5_tick_info_raw['time_msc'] / 1000)
                else:
                    logger.warning("Tidak ada tick info dari MT5. Menggunakan waktu sistem untuk update sesi.")

            logger.info("Memperbarui data sesi...")
            market_data_processor.calculate_and_update_session_data(config.TRADING_SYMBOL, current_mt5_datetime_for_session_update)
        except Exception as e:
            logger.error(f"Gagal memperbarui data sesi: {e}", exc_info=True)
        stop_event.wait(interval)


def periodic_market_status_update_loop(stop_event):
    """Loop untuk memperbarui status pasar secara periodik."""
    interval = config.Scheduler.UPDATE_INTERVALS.get('periodic_market_status_update_loop', 300)
    while not stop_event.is_set():
        try:
            logger.info("Memperbarui status pasar (berdasarkan tick terakhir)...")
            current_time_for_status = utils.to_utc_datetime_or_none(datetime.now(timezone.utc))
            latest_tick_from_db = database_manager.get_latest_price_tick(config.TRADING_SYMBOL)
            if latest_tick_from_db and latest_tick_from_db.get('time'):
                tick_dt = utils.to_utc_datetime_or_none(latest_tick_from_db['time'])
                if tick_dt and (datetime.now(timezone.utc) - tick_dt).total_seconds() < 3600:
                    current_time_for_status = tick_dt
                else:
                    logger.warning("Waktu tick terakhir dari DB terlalu lama atau tidak valid. Menggunakan waktu sistem untuk update status pasar.")
            else:
                current_mt5_tick_info_raw = mt5_connector.get_current_tick_info(config.TRADING_SYMBOL)
                if current_mt5_tick_info_raw and current_mt5_tick_info_raw.get('time_msc'):
                    current_time_for_status = utils.to_utc_datetime_or_none(current_mt5_tick_info_raw['time_msc'] / 1000)
                else:
                    logger.warning("Tidak ada tick info dari MT5. Menggunakan waktu sistem untuk update status pasar.")

            market_data_processor.calculate_and_update_session_data(config.TRADING_SYMBOL, current_time_for_status)
        except Exception as e:
            logger.error(f"Gagal memperbarui status pasar: {e}", exc_info=True)
        stop_event.wait(interval)


def periodic_mt5_trade_data_update_loop(stop_event):
    """Loop untuk memperbarui data perdagangan MT5 (posisi, order, akun)."""
    interval = config.Scheduler.UPDATE_INTERVALS.get('periodic_mt5_trade_data_update_loop', 60)
    while not stop_event.is_set():
        try:
            logger.info("Memperbarui data perdagangan MT5...")
            for attempt in range(config.System.MAX_RETRIES):
                try:
                    database_manager.update_mt5_trade_data_periodically(config.TRADING_SYMBOL)
                    break
                except OperationalError as oe:
                    if "database is locked" in str(oe):
                        logger.warning(f"Database locked saat mengupdate data perdagangan MT5. Coba lagi dalam {config.System.RETRY_DELAY_SECONDS} detik (Percobaan {attempt + 1}/{config.System.MAX_RETRIES}). Error: {oe}", exc_info=True)
                        time.sleep(config.System.RETRY_DELAY_SECONDS)
                    else:
                        logger.error(f"OperationalError (non-lock) saat mengupdate data perdagangan MT5: {oe}", exc_info=True)
                        raise
                except Exception as e:
                    logger.error(f"Error tak terduga saat mengupdate data perdagangan MT5: {e}", exc_info=True)
                    break
            else:
                logger.error(f"Gagal mengupdate data perdagangan MT5 setelah {config.System.MAX_RETRIES} percobaan.")
        except Exception as e:
            logger.error(f"Gagal memperbarui data perdagangan MT5: {e}", exc_info=True)
        stop_event.wait(interval)

def periodic_daily_pnl_check_loop(stop_event):
    """Loop untuk memeriksa profit/loss harian dan mengambil tindakan."""
    interval = config.Scheduler.UPDATE_INTERVALS.get('periodic_daily_pnl_check_loop', 300)
    while not stop_event.is_set():
        try:
            logger.info("Memeriksa status Profit/Loss Harian...")
            # Panggil fungsi perhitungan P/L harian
            data_updater.calculate_and_update_daily_pnl()
            # Panggil fungsi pemeriksaan dan tindakan P/L harian
            auto_trade_manager.check_daily_pnl_and_act(config.TRADING_SYMBOL)
        except Exception as e:
            logger.error(f"Gagal memeriksa status P/L harian: {e}", exc_info=True)
        stop_event.wait(interval)



def daily_cleanup_scheduler_loop(stop_event, hour=3, minute=0):
    """Loop untuk menjadwalkan pembersihan database harian."""
    interval = config.Scheduler.UPDATE_INTERVALS.get('daily_cleanup_scheduler_loop', 3600 * 24)
    while not stop_event.is_set():
        now_utc = datetime.now(timezone.utc)
        target_time = now_utc.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now_utc > target_time:
            target_time += timedelta(days=1)
        wait_seconds = (target_time - now_utc).total_seconds()
        logger.info(f"Pembersihan DB berikutnya dalam {wait_seconds / 3600:.2f} jam.")
        if stop_event.wait(wait_seconds): break
        try:
            logger.info("Menjalankan pembersihan database...")
            database_manager.clean_old_price_ticks()
            database_manager.clean_old_mt5_trade_data()
            database_manager.clean_old_historical_candles()

            chart_output_dir = "charts_output"
            days_to_keep_charts = 7

            if os.path.exists(chart_output_dir):
                cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_to_keep_charts)
                deleted_chart_count = 0
                for f in glob.glob(os.path.join(chart_output_dir, "*.png")):
                    try:
                        file_timestamp = datetime.fromtimestamp(os.path.getmtime(f), tz=timezone.utc)
                        if file_timestamp < cutoff_time:
                            os.remove(f)
                            deleted_chart_count += 1
                    except Exception as e:
                        logger.error(f"Gagal menghapus file chart {f}: {e}")
                logger.info(f"Dibersihkan {deleted_chart_count} file chart lama dari {chart_output_dir}.")

        except Exception as e:
            logger.error(f"Gagal menjalankan pembersihan database: {e}", exc_info=True)
            stop_event.wait(300)


def daily_open_prices_scheduler_loop(stop_event):
    """Loop untuk menjadwalkan pembaruan harga pembukaan harian."""
    interval = config.Scheduler.UPDATE_INTERVALS.get('daily_open_prices_scheduler_loop', 24 * 60 * 60)
    while not stop_event.is_set():
        now_utc = datetime.now(timezone.utc)
        target_run_time = (now_utc + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
        sleep_duration_seconds = (target_run_time - now_utc).total_seconds()
        logger.info(f"Update Daily Open berikutnya pada {target_run_time:%Y-%m-%d %H:%M:%S UTC}.")
        if stop_event.wait(sleep_duration_seconds): break
        try:
            logger.info("Menjalankan update_daily_open_prices_logic().")
            import market_data_processor
            market_data_processor.update_daily_open_prices_logic(config.TRADING_SYMBOL)
        except Exception as e:
            logger.error(f"Gagal menjalankan update Daily Open: {e}", exc_info=True)
            stop_event.wait(300)


def automatic_signal_generation_loop(stop_event):
    """Loop untuk generasi sinyal otomatis dari AI."""
    interval = config.Scheduler.UPDATE_INTERVALS.get('automatic_signal_generation_loop', 300)
    while not stop_event.is_set():
        try:
            logger.info("Memanggil analisis pasar AI (multiple analis & konsensus)...")
            
            # Dapatkan harga real-time terakhir dari data_updater
            current_price_from_tick, current_price_timestamp = data_updater.get_latest_realtime_price()
            
            # Pastikan ada harga yang valid sebelum meneruskan ke AI
            if current_price_from_tick is None or current_price_from_tick <= Decimal('0.0'):
                logger.warning("Harga real-time tidak valid atau belum tersedia. Melewatkan generasi sinyal AI.")
                stop_event.wait(interval)
                continue

            # Teruskan harga real-time ke fungsi analisis konsensus
            ai_consensus_manager.analyze_market_and_get_consensus(
                current_price=current_price_from_tick,
                current_price_timestamp=current_price_timestamp
            )

        except Exception as e:
            logger.error(f"ERROR (Scheduler - AI Signal): {e}", exc_info=True)
        stop_event.wait(interval)

def periodic_historical_data_update_loop(stop_event: threading.Event):
    """
    Loop untuk memperbarui data candle historis TERBARU dan indikator terkait secara periodik.
    Fokus pada data terbaru untuk pembaruan yang sering, menghindari duplikasi backfill penuh.
    """
    interval = config.Scheduler.UPDATE_INTERVALS.get('periodic_historical_data_update_loop', 1800)
    while not stop_event.is_set():
        if _feature_backfill_completed:
            try:
                data_updater.periodic_historical_data_update_loop(stop_event)
            except Exception as e:
                logger.error(f"Gagal update historical candles & indikator: {e}", exc_info=True)
        else:
            # Modifikasi: Tambahkan logging yang lebih spesifik saat menunggu backfill
            logger.info("Periodic Historical Data Update loop menunggu backfill fitur selesai...")
        stop_event.wait(interval)
 #

def rule_based_signal_loop(stop_event: threading.Event):
    """Loop periodik untuk generasi sinyal berbasis aturan."""
    interval = config.Scheduler.UPDATE_INTERVALS.get('rule_based_signal_loop', 60)
    
    while not stop_event.is_set():
        if _feature_backfill_completed: # Memastikan backfill fitur sudah selesai
            try:
                logger.info("Memicu generasi sinyal berbasis aturan...")
                latest_tick = database_manager.get_latest_price_tick(config.TRADING_SYMBOL)

                if not latest_tick or latest_tick.get('last_price') is None or latest_tick.get('last_price') <= 0:
                    logger.warning("Tidak ada tick terakhir yang valid untuk memicu sinyal berbasis aturan.")
                    stop_event.wait(interval)
                    continue

                current_time = latest_tick['time']
                current_price = Decimal(str(latest_tick['last_price']))

                rule_signal = rule_based_signal_generator.generate_signal(
                    config.TRADING_SYMBOL, current_time, current_price
                )

                if rule_signal and rule_signal['action'] in ["BUY", "SELL", "HOLD"]:
                    # --- BAGIAN INI SELALU DIJALANKAN (LOGGING DAN SIMPAN KE DB) ---
                    logger.info(f"Sinyal Berbasis Aturan: {rule_signal['action']} (Conf: {rule_signal['confidence']})")

                    database_manager.save_ai_analysis_result(
                        symbol=config.TRADING_SYMBOL,
                        timestamp=current_time,
                        summary=rule_signal.get('reasoning', 'Sinyal berbasis aturan.'),
                        potential_direction=rule_signal.get('potential_direction', 'Undefined'),
                        recommendation_action=rule_signal['action'],
                        entry_price=rule_signal.get('entry_price_suggestion'),
                        stop_loss=rule_signal.get('stop_loss_suggestion'),
                        take_profit=rule_signal.get('take_profit_suggestion'),
                        reasoning=rule_signal.get('reasoning', 'N/A'),
                        ai_confidence=rule_signal.get('confidence', 'High'), # Gunakan confidence dari rule_signal
                        raw_response_json=json.dumps(rule_signal, default=_json_default),
                        analyst_id="Rule_Based_Strategy"
                    )
                    logger.info("Sinyal berbasis aturan berhasil disimpan ke DB.")
                    # --- AKHIR BAGIAN SELALU DIJALANKAN ---

                    # --- BAGIAN INI HANYA DIJALANKAN JIKA KONDISI JELAS TERPENUHI (NOTIFIKASI TELEGRAM) ---
                    # MODIFIKASI: Kirim BUY/SELL tanpa peduli Confidence, HOLD jangan dikirim
                    if rule_signal['action'] != "HOLD":
                        
                        message_lines = [f"⚙️ *Sinyal Berbasis Aturan: {rule_signal['action']}* untuk {config.TRADING_SYMBOL}"]
                        
                        # Buat daftar TP
                        tp_prices = []
                        if rule_signal.get('take_profit_suggestion') is not None:
                            tp_prices.append(rule_signal['take_profit_suggestion'])
                        if rule_signal.get('tp2_suggestion') is not None:
                            tp_prices.append(rule_signal['tp2_suggestion'])
                        if rule_signal.get('tp3_suggestion') is not None:
                            tp_prices.append(rule_signal['tp3_suggestion'])

                        # Panggil _get_entry_strategy_section untuk mendapatkan format SL/TP dengan pips
                        # Pastikan config.Telegram.PIP_UNIT_IN_DOLLAR sudah didefinisikan di config.py
                        entry_strategy_section = ai_consensus_manager._get_entry_strategy_section(
                            rule_signal['action'],
                            rule_signal.get('entry_price_suggestion'),
                            rule_signal.get('stop_loss_suggestion'),
                            tp_prices,
                            [], # Kosongkan confirmations_list untuk rule-based signal jika tidak ada
                            config.PIP_UNIT_IN_DOLLAR # Modifikasi: Gunakan config.PIP_UNIT_IN_DOLLAR
                        )
                        message_lines.extend(entry_strategy_section)

                        if rule_signal.get('reasoning'):
                            # Hapus backslash yang tidak diinginkan yang mungkin sudah ada di string
                            raw_reasoning = rule_signal['reasoning']
                            cleaned_reasoning = raw_reasoning.replace('\\(', '(').replace('\\)', ')') \
                                                           .replace('\\.', '.').replace('\\-', '-') \
                                                           .replace('\\*', '*')
                            message_lines.append(f"Alasan: {cleaned_reasoning}")
                        
                        notification_service.send_telegram_message("\n".join(message_lines))
                        logger.info(f"Notifikasi sinyal berbasis aturan {rule_signal['action']} (Conf: {rule_signal['confidence']}) dikirim.")
                    else:
                        logger.info(f"Notifikasi sinyal berbasis aturan {rule_signal['action']} (Conf: {rule_signal['confidence']}) dilewati (action HOLD).")
                    # --- AKHIR MODIFIKASI NOTIFIKASI ---

                    # --- BAGIAN INI UNTUK EKSEKUSI TRADE (JUGA KONDISIONAL) ---
                    # Logika ini masih menggunakan confidence untuk eksekusi trade
                    MIN_CONFIDENCE_FOR_TRADE_EXECUTION = "Medium" 
                    confidence_levels_map = {"Low": 1, "Medium": 2, "High": 3} 
                    current_signal_confidence_value = confidence_levels_map.get(rule_signal['confidence'], 1)
                    min_trade_confidence_value = confidence_levels_map.get(MIN_CONFIDENCE_FOR_TRADE_EXECUTION, 1)

                    if config.Trading.auto_trade_enabled and rule_signal['action'] in ["BUY", "SELL"]:
                        if current_signal_confidence_value >= min_trade_confidence_value:
                            import auto_trade_manager
                            trade_result = auto_trade_manager.execute_ai_trade(
                                symbol=config.TRADING_SYMBOL,
                                action=rule_signal['action'],
                                volume=config.Trading.AUTO_TRADE_VOLUME,
                                entry_price=rule_signal.get('entry_price_suggestion'),
                                stop_loss=rule_signal.get('stop_loss_suggestion'),
                                take_profit=rule_signal.get('take_profit_suggestion'),
                                magic_number=config.Trading.AUTO_TRADE_MAGIC_NUMBER,
                                slippage=config.Trading.AUTO_TRADE_SLIPPAGE
                            )
                            notification_service.notify_trade_execution(trade_result)
                        else:
                            logger.info(f"Eksekusi trade untuk sinyal berbasis aturan {rule_signal['action']} dilewatkan (kepercayaan rendah untuk trade).")
            except Exception as e:
                    logger.error(f"ERROR (Scheduler - Rule Based Signal): {e}", exc_info=True)
        else:
                # Modifikasi: Tambahkan logging yang lebih spesifik saat menunggu backfill
                logger.info("Rule-based Signal loop menunggu backfill fitur selesai...")
        stop_event.wait(interval)


def daily_summary_report_loop(stop_event: threading.Event):
    """Loop untuk menjadwalkan pembersihan database harian."""
    interval = config.Scheduler.UPDATE_INTERVALS.get('daily_summary_report_loop', 3600 * 24)
    while not stop_event.is_set():
        now_utc = datetime.now(timezone.utc)
        target_time = now_utc.replace(hour=23, minute=59, second=0, microsecond=0)
        
        if now_utc > target_time:
            target_time += timedelta(days=1)

        wait_seconds = (target_time - now_utc).total_seconds()
        logger.info(f"Laporan ringkasan harian berikutnya dalam {wait_seconds / 3600:.2f} jam.")
        
        if stop_event.wait(wait_seconds):
            break

        try:
            logger.info("Menjalankan laporan ringkasan harian...")
            notification_service.notify_daily_summary(config.TRADING_SYMBOL)
        except Exception as e:
            logger.error(f"Gagal menjalankan laporan ringkasan harian: {e}", exc_info=True)
            stop_event.wait(60)

def periodic_combined_advanced_detection_loop(stop_event):
    """Loop untuk S&R, S&D, Fibonacci, ICT/SMC, dan Key Levels."""
    interval = config.Scheduler.UPDATE_INTERVALS.get('periodic_combined_advanced_detection_loop', 900)
    while not stop_event.is_set():
        if _feature_backfill_completed:
            try:
                logger.info("Menjalankan deteksi lanjutan...")
                market_data_processor.update_all_sr_sd_data(config.TRADING_SYMBOL)
                market_data_processor.update_all_fibonacci_levels(config.TRADING_SYMBOL)
                market_data_processor.identify_key_levels_across_timeframes(config.TRADING_SYMBOL)
            except Exception as e:
                logger.error(f"Gagal menjalankan deteksi lanjutan: {e}", exc_info=True)
        else:
            # Modifikasi: Tambahkan logging yang lebih spesifik saat menunggu backfill
            logger.info("Advanced Detection loop menunggu backfill fitur selesai...")
        stop_event.wait(interval)

def periodic_volume_profile_update_loop(stop_event):
    """Loop untuk memperbarui Volume Profile secara periodik."""
    interval = config.Scheduler.UPDATE_INTERVALS.get('periodic_volume_profile_update_loop', 1800)
    while not stop_event.is_set():
        if _feature_backfill_completed:
            try:
                logger.info("Menjalankan update Volume Profiles...")
                # market_data_processor.update_volume_profiles(config.TRADING_SYMBOL) # Ini sekarang harus dipanggil via data_updater
            except Exception as e:
                logger.error(f"Gagal menjalankan update Volume Profiles: {e}", exc_info=True)
        else:
            # Modifikasi: Tambahkan logging yang lebih spesifik saat menunggu backfill
            logger.info("Volume Profile loop menunggu backfill fitur selesai...")
        stop_event.wait(interval)

def periodic_fundamental_data_update_loop(stop_event):
    """
    Loop untuk memperbarui data fundamental (kalender dan berita) secara periodik.
    """
    interval = config.Scheduler.UPDATE_INTERVALS.get('periodic_fundamental_data_update_loop', 14400)
    fund_service = fundamental_data_service.FundamentalDataService()

    while not stop_event.is_set():
        now_utc = datetime.now(timezone.utc)

        logger.info(f"Pembaruan fundamental berikutnya dijadwalkan dalam {interval / 3600:.1f} jam.")

        stopped = stop_event.wait(interval)
        if stopped:
            break

        try:
            logger.info("Menjalankan update data fundamental (scraping dan penyimpanan)...")

            comprehensive_data = fund_service.get_comprehensive_fundamental_data(
                days_past=7, # Bisa disesuaikan di config.py jika ingin lebih luas
                days_future=7, # Bisa disesuaikan
                min_impact_level="Low", # Bisa disesuaikan
                target_currency="USD", # Asumsi fokus USD untuk XAUUSD, bisa jadi parameter
                include_news_topics=["gold", "usd", "suku bunga", "perang", "fed", "inflation", "market", "economy", "geopolitics", "commodities", "finance", "conflict", "trade"] # Bisa disesuaikan
            )
            
            logger.info("Selesai menyimpan data fundamental.")
        except Exception as e:
            logger.error(f"Gagal dalam siklus update fundamental: {e}", exc_info=True)
            stop_event.wait(3600)


def _scenario_analysis_loop_wrapper(stop_event: threading.Event):
    interval = config.Scheduler.UPDATE_INTERVALS.get('scenario_analysis_loop', 3600)
    while not stop_event.is_set():
        if _feature_backfill_completed:
            try:
                logger.info("Memicu analisis skenario...")
                ai_consensus_manager.analyze_and_notify_scenario(
                    symbol=config.TRADING_SYMBOL,
                    timeframe="H1" # Default timeframe untuk analisis skenario
                )
            except Exception as e:
                logger.error(f"ERROR (Scheduler - Scenario Analysis): {e}", exc_info=True)
        else:
            # Modifikasi: Tambahkan logging yang lebih spesifik saat menunggu backfill
            logger.info("Scenario Analysis loop menunggu backfill fitur selesai...")
        stop_event.wait(interval)

def detector_health_monitoring_loop(stop_event: threading.Event):
    """
    Loop untuk secara periodik memeriksa status kesehatan detektor
    dan mengirim notifikasi Telegram jika ada anomali.
    """
    # Mengambil interval dari konfigurasi (config.Monitoring.DETECTOR_HEALTH_CHECK_INTERVAL_SECONDS)
    # Asumsi config.Monitoring sudah ada dan memiliki DETECTOR_HEALTH_CHECK_INTERVAL_SECONDS
    # Jika belum, Anda perlu menambahkannya di config.py
    interval = config.Scheduler.UPDATE_INTERVALS.get('detector_health_monitoring_loop', 3600)
    
    # defaultdict untuk melacak kapan terakhir kali anomali detektor ini dinotifikasi,
    # agar tidak spam notifikasi.
    last_notified_anomalies = defaultdict(datetime) 

    while not stop_event.is_set(): # Menggunakan stop_event yang diteruskan sebagai argumen
        logger.info("Menjalankan pemantauan kesehatan detektor...")
        
        # Mengambil salinan status kesehatan detektor dari modul detector_monitoring
        # Menggunakan detector_monitoring.detector_monitor untuk mengakses instance global
        with detector_monitoring.detector_monitor._detector_health_lock:
            current_health_status = detector_monitoring.detector_monitor.get_health_status()

        anomalies_found = []
        for detector_name, status_data in current_health_status.items():
            current_time = datetime.now(timezone.utc)
            
            # Cek cooldown notifikasi untuk detektor ini
            # Asumsi config.Telegram.NOTIFICATION_COOLDOWN_SECONDS sudah ada di config.py
            # Jika belum, Anda perlu menambahkannya di config.py
            telegram_cooldown = getattr(config.Telegram, 'NOTIFICATION_COOLDOWN_SECONDS', 3600)
            if (current_time - last_notified_anomalies[detector_name]).total_seconds() < telegram_cooldown:
                continue

            # Kriteria Deteksi Anomali
            if status_data["status"] == "Failed":
                anomalies_found.append(f"🔴 Detektor *{detector_name}* GAGAL! Error: `{utils._escape_markdown(status_data.get('error_message_last_run', 'Tidak ada pesan'))}`. Cek log lebih lanjut.")
                last_notified_anomalies[detector_name] = current_time
            elif status_data["status"] == "Warning" and status_data["consecutive_zero_detections"] >= getattr(config.Monitoring, 'DETECTOR_ZERO_DETECTIONS_THRESHOLD', 5):
                anomalies_found.append(f"🟡 Detektor *{detector_name}* Peringatan: {status_data['consecutive_zero_detections']} run berturut-turut dengan 0 deteksi. Pesan: `{utils._escape_markdown(status_data.get('warning_message_last_run', 'Tidak ada pesan'))}`.")
                last_notified_anomalies[detector_name] = current_time
            # Opsi: Notifikasi jika detektor belum pernah berjalan (status "Not Run") setelah waktu tertentu
            elif status_data["status"] == "Not Run" and \
                (status_data["last_run_time_utc"] is None or \
                 (current_time - status_data["last_run_time_utc"]).total_seconds() > interval * 1.5):
                anomalies_found.append(f"⚪ Detektor *{detector_name}* belum pernah berjalan atau terlalu lama tidak aktif ({status_data['status']}).")
                last_notified_anomalies[detector_name] = current_time


        # Jika ada anomali yang ditemukan, kirim notifikasi Telegram
        if anomalies_found:
            message = "⚠️ *Anomali Detektor Terdeteksi:*\n\n" + "\n".join(anomalies_found)
            notification_service.send_telegram_message(message)
            logger.warning(f"Mengirim notifikasi anomali detektor: {len(anomalies_found)} anomali.")
        else:
            logger.info("Tidak ada anomali detektor yang terdeteksi.")
            
        stop_event.wait(interval)

# Modifikasi _start_data_update_threads untuk mengatur flag _feature_backfill_completed
def _start_data_update_threads(initial_run=False):
    global _feature_backfill_completed

    # Perubahan di sini: pastikan setiap fungsi dari data_updater memiliki (symbol, stop_event)
    TASKS = {
        'periodic_realtime_tick_loop': data_updater.periodic_realtime_tick_loop,
        'periodic_session_data_update_loop': data_updater.periodic_session_data_update_loop,
        'periodic_market_status_update_loop': data_updater.periodic_market_status_update_loop,
        'periodic_mt5_trade_data_update_loop': data_updater.periodic_mt5_trade_data_update_loop,
        'daily_cleanup_scheduler_loop': data_updater.daily_cleanup_scheduler_loop,
        'daily_open_prices_scheduler_loop': data_updater.daily_open_prices_scheduler_loop,
        'periodic_historical_data_update_loop': data_updater.periodic_historical_data_update_loop,
        'periodic_fundamental_data_update_loop': data_updater.periodic_fundamental_data_update_loop,
        'daily_summary_report_loop': data_updater.daily_summary_report_loop,
        'rule_based_signal_loop': data_updater.rule_based_signal_loop,
        'automatic_signal_generation_loop': automatic_signal_generation_loop, # Ini dari scheduler itu sendiri
        'scenario_analysis_loop': data_updater.scenario_analysis_loop,
        'detector_health_monitoring_loop': data_updater.detector_health_monitoring_loop,
        "periodic_daily_pnl_check_loop": data_updater.periodic_daily_pnl_check_loop,
        "periodic_volume_profile_update_loop": data_updater.periodic_volume_profile_update_loop,
        "periodic_combined_advanced_detection_loop": data_updater.periodic_combined_advanced_detection_loop,
        "monthly_historical_feature_backfill_loop": data_updater.monthly_historical_feature_backfill_loop,
    }

    with config.Scheduler._data_update_thread_restart_lock:
        if not initial_run:
            logger.info("Menghentikan thread data update yang sedang berjalan...")
            config.Scheduler._data_update_stop_event.set()
            
            for thread in config.Scheduler._data_update_threads:
                thread.join(timeout=2.0)

            config.Scheduler._data_update_threads = []
            config.Scheduler._data_update_stop_event.clear()
            logger.info("Semua thread telah dihentikan.")

        if config.Scheduler.AUTO_DATA_UPDATE_ENABLED:
            logger.info("Memulai thread data update...")
             
            threads_to_start = []
            stop_event = config.Scheduler._data_update_stop_event

            for task_name, is_enabled in config.Scheduler.ENABLED_LOOPS.items():
                if is_enabled and task_name in TASKS: # Hapus pengecualian 'monthly_historical_feature_backfill_loop' jika Anda ingin scheduler menjadwalkannya.
                    task_function = TASKS[task_name]
                    
                    # Logika penting: Pastikan argumen symbol dan stop_event selalu diteruskan
                    # Ini akan bekerja jika tanda tangan di data_updater.py sudah benar
                    thread = threading.Thread(target=task_function, args=(config.TRADING_SYMBOL, stop_event,), daemon=True)
                    threads_to_start.append(thread)
                    logger.info(f"Menjadwalkan tugas '{task_name}'.")

            for t in threads_to_start:
                t.start()
            
            config.Scheduler._data_update_threads = threads_to_start
            logger.info(f"{len(threads_to_start)} thread berhasil dimulai.")
        else:
            logger.info("Auto-Data Update dinonaktifkan, tidak ada thread yang dimulai.")


def _stop_data_update_threads():
    with config.Scheduler._data_update_thread_restart_lock:
        config.Scheduler._data_update_stop_event.set()
        for thread in config.Scheduler._data_update_threads:
            thread.join(timeout=5)
        config.Scheduler._data_update_threads = []
        config.Scheduler._data_update_stop_event.clear()
        logger.info("Semua thread scheduler telah dihentikan dan dibersihkan.")
