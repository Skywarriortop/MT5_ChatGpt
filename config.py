# config.py

import os
import threading
import MetaTrader5 as mt5 # Diimpor, namun tidak digunakan secara langsung di file config ini.
from dotenv import load_dotenv
from datetime import datetime, timezone
import sys # Diimpor, namun tidak digunakan secara langsung di file config ini.
import logging
from decimal import Decimal, getcontext # Import getcontext sudah benar

# Set presisi Decimal untuk perhitungan keuangan secara global.
# Ini penting untuk akurasi perhitungan harga dan volume trading.
getcontext().prec = 10

# Dapatkan path absolut dari direktori saat ini.
# Digunakan untuk menemukan file .env dan file status auto-trade.
basedir = os.path.abspath(os.path.dirname(__file__))

# --- Muat Environment Variables dari file .env ---
# Memastikan semua variabel konfigurasi dapat diatur melalui file .env.
load_dotenv(os.path.join(basedir, '.env'))

# Inisialisasi logger untuk modul ini.
# Digunakan untuk mencatat pesan error atau informasi penting.
logger = logging.getLogger(__name__)

# Pindahkan AUTO_TRADE_STATUS_FILENAME ke level modul (global di dalam modul config.py).
# File ini digunakan untuk menyimpan status auto-trade (aktif/nonaktif).
AUTO_TRADE_STATUS_FILENAME = "auto_trade_status.tmp"

class Config:
    """
    Kelas dasar untuk konfigurasi aplikasi.
    Menggunakan nested classes untuk organisasi yang lebih baik dan modularitas.
    """
    APP_ENV = os.getenv('APP_ENV', 'development') # Lingkungan aplikasi (development, production, dll.)

    TRADING_SYMBOL = os.getenv("TRADING_SYMBOL", "XAUUSD") # Simbol trading utama, diambil dari .env

    # --- Variabel global untuk nilai point simbol dan nilai pip dolar ---
    # Nilai default ini akan di-override dari .env di validate_config() jika ada.
    # Ini penting untuk perhitungan pips dan profit/loss.
    TRADING_SYMBOL_POINT_VALUE = Decimal("0.001") # Default nilai point untuk XAUUSD
    PIP_UNIT_IN_DOLLAR = Decimal("0.01") # Default nilai pip dalam dolar untuk XAUUSD

    # Semua kelas bersarang di bawah ini HARUS diindentasi 4 spasi ke dalam dari `class Config:`
    class Database:
        """Konfigurasi terkait database."""
        URL = os.getenv("DATABASE_URL") # URL koneksi database

    class Paths:
        """Konfigurasi terkait jalur file dan direktori."""
        MQL5_DATA_FILE = os.getenv("MQL5_DATA_FILE_PATH") # Jalur file data MetaTrader 5

    class APIKeys:
        """Konfigurasi untuk kunci API layanan eksternal."""
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
        MYFXBOOK_COOKIES_STRING = os.getenv("MYFXBOOK_COOKIES_STRING")

    class Trading:
        """Konfigurasi terkait parameter trading dan manajemen posisi."""

        # Level Take Profit Parsial.
        # Ini adalah daftar dictionary yang mendefinisikan strategi take profit bertahap.
        # Nilai-nilai Decimal di sini adalah default jika tidak di-override oleh logika lain.
        DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "M5")
        PARTIAL_TP_LEVELS = [
            {
                # Level 1: TP1 (Contoh: RR 1:1)
                # Harga akan bergerak sejauh 1 kali jarak SL untuk TP ini.
                'price_multiplier': Decimal('1.0'),  # RR 1:1
                'volume_percentage_to_close': Decimal('0.50'), # Tutup 50% volume
                'move_sl_to_breakeven_after_partial': False, # Belum ke BE murni
                'move_sl_to_price_after_partial': None      # Tidak ke harga spesifik
            },
            {
                # Level 2: Breakeven (BE) dengan sedikit profit (Contoh: RR 1:0.5)
                # Ketika harga bergerak sejauh 0.5 kali jarak SL menguntungkan, SL akan dipindahkan.
                'price_multiplier': Decimal('2.0'), # RR 1:0.5
                'volume_percentage_to_close': Decimal('1.00'), # TIDAK menutup volume, hanya memindahkan SL
                'move_sl_to_breakeven_after_partial': True, # Pindahkan SL ke harga entry aktual (breakeven)
                'move_sl_to_price_after_partial': None # Tidak ke harga spesifik (opsi lain adalah mengunci profit)
            },
            {
                # Level 3: TP2 (Contoh: RR 1:2)
                # Harga akan bergerak sejauh 2 kali jarak SL untuk TP ini.
                'price_multiplier': Decimal('2.0'), # RR 1:2
                'volume_percentage_to_close': Decimal('1.00'), # Tutup 100% sisa volume
                'move_sl_to_breakeven_after_partial': False,
                'move_sl_to_price_after_partial': None
            }
        ]

        # --- Daily Profit/Loss Limit Configuration ---
        # Nilai default, akan di-override dari .env di validate_config.
        DAILY_PROFIT_TARGET_PERCENT = Decimal('0.01') # Target profit harian (1%)
        DAILY_LOSS_LIMIT_PERCENT = Decimal('0.02')   # Batas kerugian harian (2%)
        SPREAD_POINTS = Decimal(os.getenv("SPREAD_POINTS", "16")) # Exness XAUUSD spread ~11 pips * 100 points/pip
        COMMISSION_PER_LOT = Decimal(os.getenv("COMMISSION_PER_LOT", "0.00")) # Exness Pro Account for XAUUSD has 0 commission

        BACKTEST_HIT_TOLERANCE_POINTS = Decimal(os.getenv("BACKTEST_HIT_TOLERANCE_POINTS", "0.5")) # Toleransi hit backtest dalam poin
        HAS_REAL_VOLUME_DATA_FROM_BROKER = False
        HAS_TICK_VOLUME_DATA_FROM_BROKER = True # Exness menyediakan tick volume
        HAS_REALTIME_TICK_VOLUME_FROM_BROKER = False
        MIN_SPREAD_POINTS = Decimal(os.getenv("MIN_SPREAD_POINTS", "0.5"))
        MAX_SPREAD_POINTS = Decimal(os.getenv("MAX_SPREAD_POINTS", "2.0"))
        MIN_SLIPPAGE_POINTS = Decimal(os.getenv("MIN_SLIPPAGE_POINTS", "0.1"))
        MAX_SLIPPAGE_POINTS = Decimal(os.getenv("MAX_SLIPPAGE_POINTS", "0.5"))

        RISK_PERCENT_PER_TRADE = Decimal(os.getenv("RISK_PERCENT_PER_TRADE", "0.01"))
        MAX_POSITION_HOLD_MINUTES = int(os.getenv("MAX_POSITION_HOLD_MINUTES", "240"))
        @staticmethod
        def _read_auto_trade_status():
            """Membaca status auto-trade dari file sementara."""
            try:
                file_path = os.path.join(basedir, AUTO_TRADE_STATUS_FILENAME)
                with open(file_path, 'r') as f:
                    content = f.read().strip().lower()
                    return content == 'true'
            except FileNotFoundError:
                return False # Default jika file tidak ditemukan
            except Exception as e:
                logger.error(f"Error reading auto-trade status file: {e}", exc_info=True)
                return False

        @staticmethod
        def _write_auto_trade_status(status: bool):
            """Menulis status auto-trade ke file sementara."""
            try:
                file_path = os.path.join(basedir, AUTO_TRADE_STATUS_FILENAME)
                with open(file_path, 'w') as f:
                    f.write(str(status).lower())
            except Exception as e:
                logger.error(f"Error writing auto-trade status to file: {e}", exc_info=True)

        # Inisialisasi status auto-trade saat kelas dimuat.
        _auto_trade_enabled_value = _read_auto_trade_status()

        @property
        def auto_trade_enabled(self):
            """Properti untuk mendapatkan status auto-trade."""
            return self._auto_trade_enabled_value

        @auto_trade_enabled.setter
        def auto_trade_enabled(self, value: bool):
            """Properti untuk mengatur status auto-trade dan menyimpannya ke file."""
            self._auto_trade_enabled_value = value
            self._write_auto_trade_status(value)

        # Nilai default untuk parameter trading, akan di-override dari .env.
        AUTO_TRADE_VOLUME = Decimal(os.getenv("AUTO_TRADE_VOLUME", "0.01"))
        AUTO_TRADE_SLIPPAGE = int(os.getenv("AUTO_TRADE_SLIPPAGE", "0")) # Toleransi slippage dalam poin
        AUTO_TRADE_MAGIC_NUMBER = int(os.getenv("AUTO_TRADE_MAGIC_NUMBER", "12345")) 
        RISK_PER_TRADE_PERCENT = Decimal(os.getenv("RISK_PER_TRADE_PERCENT", "1.0")) 
        MAX_DAILY_DRAWDOWN_PERCENT = Decimal(os.getenv("MAX_DAILY_DRAWDOWN_PERCENT", "5.0")) 
        TRAILING_STOP_PIPS = Decimal(os.getenv("TRAILING_STOP_PIPS", "15")) 
        TRAILING_STOP_STEP_PIPS = Decimal(os.getenv("TRAILING_STOP_STEP_PIPS", "5")) 
        TRADING_START_HOUR_UTC = int(os.getenv("TRADING_START_HOUR_UTC", "0")) 
        TRADING_END_HOUR_UTC = int(os.getenv("TRADING_END_HOUR_UTC", "23")) 
        MARKET_CLOSE_BUFFER_MINUTES = int(os.getenv("MARKET_CLOSE_BUFFER_MINUTES", "30")) 
        MARKET_OPEN_BUFFER_MINUTES = int(os.getenv("MARKET_OPEN_BUFFER_MINUTES", "30")) 
        MIN_SL_PIPS = Decimal(os.getenv("MIN_SL_PIPS", "0")) 
        MIN_TP_PIPS = Decimal(os.getenv("MIN_TP_PIPS", "0")) 
        SWAP_RATE_PER_LOT_BUY = Decimal(os.getenv("SWAP_RATE_PER_LOT_BUY", "-44.0")) # Swap rate buy per lot
        SWAP_RATE_PER_LOT_SELL = Decimal(os.getenv("SWAP_RATE_PER_LOT_SELL", "-44.0")) # Swap rate sell per lot


    class Scheduler:
        """Konfigurasi untuk interval dan status loop scheduler."""
        # Interval update untuk berbagai loop dalam aplikasi (dalam detik).
        # Nilai default ini akan di-override dari .env di validate_config.
        UPDATE_INTERVALS = {
            "periodic_realtime_tick_loop": 300.0,
            "periodic_session_data_update_loop": 300.0,
            "periodic_market_status_update_loop": 60.0,
            "periodic_mt5_trade_data_update_loop": 10.0,
            "daily_cleanup_scheduler_loop": float(3600*24),
            "daily_open_prices_scheduler_loop": float(3600*24),
            "automatic_signal_generation_loop": 1800.0,
            "periodic_historical_data_update_loop": 3600.0,
            "periodic_volume_profile_update_loop": 3600.0,
            "periodic_combined_advanced_detection_loop": 300.0,
            "periodic_fundamental_data_update_loop": 3600.0,
            "rule_based_signal_loop": 60.0,
            "daily_summary_report_loop": float(3600*24),
            "monthly_historical_feature_backfill_loop": float(3600*24*30),
            "scenario_analysis_loop": 300.0,
            "detector_health_monitoring_loop": 60.0,
            "periodic_daily_pnl_check_loop": 300.0,
        }

        # Status aktif/nonaktif untuk berbagai loop scheduler.
        # Nilai default ini akan di-override dari .env di validate_config.
        ENABLED_LOOPS = {
            "periodic_realtime_tick_loop": True,
            "periodic_session_data_update_loop": True,
            "periodic_market_status_update_loop": True,
            "periodic_mt5_trade_data_update_loop": True,
            "daily_cleanup_scheduler_loop": False,
            "daily_open_prices_scheduler_loop": True,
            "automatic_signal_generation_loop": False,
            "periodic_historical_data_update_loop": True,
            "periodic_volume_profile_update_loop": True,
            "periodic_combined_advanced_detection_loop": True,
            "periodic_fundamental_data_update_loop": False,
            "rule_based_signal_loop": True,
            "daily_summary_report_loop": True,
            "monthly_historical_feature_backfill_loop": True,
            "scenario_analysis_loop": True,
            "detector_health_monitoring_loop": True,
            "periodic_daily_pnl_check_loop": True,
        }

        _data_update_thread_restart_lock = threading.Lock()
        _data_update_stop_event = threading.Event()
        _data_update_threads = []

        AUTO_DATA_UPDATE_ENABLED = os.getenv("AUTO_DATA_UPDATE_ENABLED", "true").lower() == 'true'

    class Sessions:
        """Konfigurasi untuk jam buka/tutup sesi pasar utama (UTC)."""
        ASIA_SESSION_START_HOUR_UTC = int(os.getenv("ASIA_SESSION_START_HOUR_UTC", "0"))
        ASIA_SESSION_END_HOUR_UTC = int(os.getenv("ASIA_SESSION_END_HOUR_UTC", "9")) # Umumnya sampai 09:00 UTC

        EUROPE_SESSION_START_HOUR_UTC = int(os.getenv("EUROPE_SESSION_START_HOUR_UTC", "7"))
        EUROPE_SESSION_END_HOUR_UTC = int(os.getenv("EUROPE_SESSION_END_HOUR_UTC", "16"))

        NEWYORK_SESSION_START_HOUR_UTC = int(os.getenv("NEWYORK_SESSION_START_HOUR_UTC", "12")) # Sesuai DST
        NEWYORK_SESSION_END_HOUR_UTC = int(os.getenv("NEWYORK_SESSION_END_HOUR_UTC", "21")) # Sesuai DST

    class MarketData:
        """
        Konfigurasi terkait pengumpulan data pasar, timeframe, dan parameter detektor.
        INI ADALAH SATU-SATUNYA DEFINISI UNTUK KELAS MarketData.
        Pastikan semua properti yang terkait dengan data pasar ada di sini.
        """
        # Jumlah candle yang akan dikumpulkan untuk setiap timeframe.
        COLLECT_TIMEFRAMES = {
            "M1": 1000,
            "M5": 5000,
            "M15": 5000,
            "M30": 5000,
            "H1": 5000,
            "H4": 5000,
            "D1": 5000
        }

        # Status aktif/nonaktif untuk setiap timeframe.
        ENABLED_TIMEFRAMES = {
            "M1": os.getenv("ENABLE_M1_TF", "false").lower() == 'true', # Default 'false' untuk M1
            "M5": os.getenv("ENABLE_M5_TF", "true").lower() == 'true',
            "M15": os.getenv("ENABLE_M15_TF", "true").lower() == 'true',
            "M30": os.getenv("ENABLE_M30_TF", "true").lower() == 'true',
            "H1": os.getenv("ENABLE_H1_TF", "true").lower() == 'true',
            "H4": os.getenv("ENABLE_H4_TF", "true").lower() == 'true',
            "D1": os.getenv("ENABLE_D1_TF", "true").lower() == 'true'
        }

        MA_PERIODS_TO_CALCULATE = [Decimal('20'), Decimal('50'), Decimal('100')] # Periode MA default
        MA_TYPES_TO_CALCULATE = ["SMA", "EMA"] # Tipe MA default

        HISTORICAL_DATA_RETENTION_DAYS = {
            "M1": int(os.getenv("RETENTION_M1_DAYS", "7")),
            "M5": int(os.getenv("RETENTION_M5_DAYS", "14")),
            "M15": int(os.getenv("RETENTION_M15_DAYS", "30")),
            "M30": int(os.getenv("RETENTION_M30_DAYS", "60")),
            "H1": int(os.getenv("RETENTION_H1_DAYS", "90")),
            "H4": int(os.getenv("RETENTION_H4_DAYS", "180")),
            "D1": int(os.getenv("RETENTION_D1_DAYS", "0")) # 0 berarti tidak ada retensi (semua data)
        }

        HISTORICAL_DATA_START_DATE_FULL = os.getenv("HISTORICAL_DATA_START_DATE_FULL", "2024-01-01")

        _market_status_data_lock = threading.Lock()
        market_status_data = {
            "session_status": [],
            "overlap_status": "Not Determined",
            "current_utc_time": datetime.now(timezone.utc).isoformat(),
            "detailed_sessions": []
        }

        TIMEZONE_FOR_DST_CHECK = os.getenv("TIMEZONE_FOR_DST_CHECK", "America/New_York")

        # --- PENGATURAN UMUM UNTUK DETEKTOR YANG MENGGUNAKAN ATR UNTUK TOLERANSI ---
        ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14")) # Periode ATR
        ATR_MULTIPLIER_FOR_TOLERANCE = Decimal(os.getenv("ATR_MULTIPLIER_FOR_TOLERANCE", "0.5")) # Multiplier ATR untuk toleransi

        # --- PENGATURAN DETEKTOR ---
        ENABLE_SR_DETECTION = os.getenv("ENABLE_SR_DETECTION", "true").lower() == 'true'
        SR_LOOKBACK_CANDLES = int(os.getenv("SR_LOOKBACK_CANDLES", "200"))
        SR_ZONE_ATR_MULTIPLIER = Decimal(os.getenv("SR_ZONE_ATR_MULTIPLIER", "0.5"))
        MIN_SR_STRENGTH = int(os.getenv("MIN_SR_STRENGTH", "1"))

        ENABLE_OB_FVG_DETECTION = os.getenv("ENABLE_OB_FVG_DETECTION", "true").lower() == 'true'
        FVG_MIN_ATR_MULTIPLIER = Decimal(os.getenv("FVG_MIN_ATR_MULTIPLIER", "0.2"))
        OB_MIN_VOLUME_MULTIPLIER = Decimal(os.getenv("OB_MIN_VOLUME_MULTIPLIER", "1.5"))
        OB_FVG_MITIGATION_LOOKBACK_CANDLES = int(os.getenv("OB_FVG_MITIGATION_LOOKBACK_CANDLES", "50"))

        ENABLE_LIQUIDITY_DETECTION = os.getenv("ENABLE_LIQUIDITY_DETECTION", "true").lower() == 'true'
        LIQUIDITY_CANDLE_RANGE_PERCENT = Decimal(os.getenv("LIQUIDITY_CANDLE_RANGE_PERCENT", "0.5"))

        ENABLE_FIBONACCI_DETECTION = os.getenv("ENABLE_FIBONACCI_DETECTION", "true").lower() == 'true'
        # FIBO_RETRACTION_LEVELS akan ditarik dari .env di validate_config
        FIBO_RETRACTION_LEVELS = [Decimal('0.236'), Decimal('0.382'), Decimal('0.5'), Decimal('0.618'), Decimal('0.786')]

        ENABLE_MARKET_STRUCTURE_DETECTION = os.getenv("ENABLE_MARKET_STRUCTURE_DETECTION", "true").lower() == 'true'
        BOS_CHOCH_MIN_PIPS_CONFIRMATION = Decimal(os.getenv("BOS_CHOCH_MIN_PIPS_CONFIRMATION", "10.0"))

        ENABLE_SWING_DETECTION = os.getenv("ENABLE_SWING_DETECTION", "true").lower() == 'true'
        SWING_LOOKBACK_CANDLES = int(os.getenv("SWING_LOOKBACK_CANDLES", "50"))

        ENABLE_DIVERGENCE_DETECTION = os.getenv("ENABLE_DIVERGENCE_DETECTION", "true").lower() == 'true'
        RSI_DIVERGENCE_PERIODS = int(os.getenv("RSI_DIVERGENCE_PERIODS", "14"))
        MACD_DIVERGENCE_FAST_PERIOD = int(os.getenv("MACD_DIVERGENCE_FAST_PERIOD", "12"))
        MACD_DIVERGENCE_SLOW_PERIOD = int(os.getenv("MACD_DIVERGENCE_SLOW_PERIOD", "26"))
        MACD_DIVERGENCE_SIGNAL_PERIOD = int(os.getenv("MACD_DIVERGENCE_SIGNAL_PERIOD", "9"))

        ENABLE_RSI_CALCULATION = os.getenv("ENABLE_RSI_CALCULATION", "true").lower() == 'true'
        # --- DUPLIKASI ATRIBUT UNTUK KOMPATIBILITAS ---
        # Atribut ini juga ada di AIAnalysts untuk memenuhi dependensi di sana.
        RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
        RSI_OVERBOUGHT_LEVEL = Decimal(os.getenv("RSI_OVERBOUGHT_LEVEL", "70.0"))
        RSI_OVERSOLD_LEVEL = Decimal(os.getenv("RSI_OVERSOLD_LEVEL", "30.0"))

        ENABLE_MACD_CALCULATION = os.getenv("ENABLE_MACD_CALCULATION", "true").lower() == 'true'
        # --- DUPLIKASI ATRIBUT UNTUK KOMPATIBILITAS ---
        # Atribut ini juga ada di AIAnalysts untuk memenuhi dependensi di sana.
        MACD_FAST_PERIOD = int(os.getenv("MACD_FAST_PERIOD", "12"))
        MACD_SLOW_PERIOD = int(os.getenv("MACD_SLOW_PERIOD", "26"))
        MACD_SIGNAL_PERIOD = int(os.getenv("MACD_SIGNAL_PERIOD", "9"))
        # --- AKHIR DUPLIKASI ---

        ENABLE_EMA_CROSS_DETECTION = os.getenv("ENABLE_EMA_CROSS_DETECTION", "true").lower() == 'true'
        EMA_FAST_PERIOD = int(os.getenv("EMA_FAST_PERIOD", "50"))
        EMA_SLOW_PERIOD = int(os.getenv("EMA_SLOW_PERIOD", "200"))

        ENABLE_MA_TREND_DETECTION = os.getenv("ENABLE_MA_TREND_DETECTION", "true").lower() == 'true'
        # MA_TREND_PERIODS akan ditarik dari .env di validate_config
        MA_TREND_PERIODS = [Decimal('20'), Decimal('50'), Decimal('200')]
        MA_TREND_TIMEFRAMES = ["M15", "M30", "H1", "H4", "D1"]
        MA_TREND_TIMEFRAME_WEIGHTS = {
            "M15": 1,
            "M30": 1,
            "H1": 1,
            "H4": 2,
            "D1": 3
        }

        ENABLE_VOLUME_PROFILE_DETECTION = os.getenv("ENABLE_VOLUME_PROFILE_DETECTION", "true").lower() == 'true'
        ENABLE_PREVIOUS_HIGH_LOW_DETECTION = os.getenv("ENABLE_PREVIOUS_HIGH_LOW_DETECTION", "true").lower() == 'true'

        # --- Pengaturan terkait Konfluensi ---
        CONFLUENCE_PROXIMITY_TOLERANCE_PIPS = Decimal(os.getenv("CONFLUENCE_PROXIMITY_TOLERANCE_PIPS", "10.0"))
        CONFLUENCE_SCORE_PER_LEVEL = int(os.getenv("CONFLUENCE_SCORE_PER_LEVEL", "1"))

        # --- Pengaturan Khusus untuk Order Blocks (OB) Clustering ---
        OB_CONSOLIDATION_TOLERANCE_POINTS = Decimal(os.getenv("OB_CONSOLIDATION_TOLERANCE_POINTS", "25.0"))
        OB_SHOULDER_LENGTH = int(os.getenv("OB_SHOULDER_LENGTH", "3")) # Duplikat di RuleBasedStrategy jika sama

        # --- Pengaturan Khusus untuk Fair Value Gaps (FVG) ---
        FVG_MIN_CANDLE_BODY_PERCENT_FOR_STRENGTH = Decimal(os.getenv("FVG_MIN_CANDLE_BODY_PERCENT_FOR_STRENGTH", "0.7"))
        FVG_VOLUME_FACTOR_FOR_STRENGTH = Decimal(os.getenv("FVG_VOLUME_FACTOR_FOR_STRENGTH", "0.5"))


    class AIAnalysts:
        """Konfigurasi untuk modul analisis AI."""
        OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        AI_MIN_ANALYSTS_FOR_CONSENSUS = int(os.getenv("AI_MIN_ANALYSTS_FOR_CONSENSUS", "2"))

        FVG_MIN_DOLLARS = Decimal(os.getenv("FVG_MIN_DOLLARS", "0.0005"))
        SWING_EXT_BARS = int(os.getenv("SWING_EXT_BARS", "1"))

        # --- ATRIBUT INI DIKEMBALIKAN KE SINI SESUAI PERMINTAAN TRACEBACK SEBELUMNYA ---
        # Ini adalah atribut yang dilaporkan hilang dalam traceback sebelumnya (dari market_data_processor.py).
        # Sekarang diduplikasi di MarketData juga untuk memenuhi semua dependensi.
        RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
        RSI_OVERBOUGHT = Decimal(os.getenv("RSI_OVERBOUGHT", "70.0"))
        RSI_OVERSOLD = Decimal(os.getenv("RSI_OVERSOLD", "30.0"))
        MACD_FAST_PERIOD = int(os.getenv("MACD_FAST_PERIOD", "12"))
        MACD_SLOW_PERIOD = int(os.getenv("MACD_SLOW_PERIOD", "26"))
        MACD_SIGNAL_PERIOD = int(os.getenv("MACD_SIGNAL_PERIOD", "9"))
        # --- AKHIR PENGEMBALIAN / DUPLIKASI ---
        TREND_MA_SHORT_PERIOD = int(os.getenv("TREND_MA_SHORT_PERIOD", "20")) # Contoh nilai, bisa disesuaikan
        TREND_MA_MEDIUM_PERIOD = int(os.getenv("TREND_MA_MEDIUM_PERIOD", "50")) # Contoh nilai, bisa disesuaikan
        TREND_MA_LONG_PERIOD = int(os.getenv("TREND_MA_LONG_PERIOD", "200")) # Contoh nilai, bisa disesuaikan

        # jumlah angka yg muncul di telegram (ini lebih ke visualisasi AI)
        MAX_FVG_DISPLAY = int(os.getenv("AI_MAX_FVG_DISPLAY", "3"))
        MAX_OB_DISPLAY = int(os.getenv("AI_MAX_OB_DISPLAY", "3"))
        MAX_KEY_LEVEL_DISPLAY = int(os.getenv("AI_MAX_KEY_LEVEL_DISPLAY", "5"))
        MAX_SR_DISPLAY = int(os.getenv("AI_MAX_SR_DISPLAY", "3"))
        MAX_LIQUIDITY_DISPLAY = int(os.getenv("AI_MAX_LIQUIDITY_DISPLAY", "0"))
        MAX_DIVERGENCE_DISPLAY = int(os.getenv("AI_MAX_DIVERGENCE_DISPLAY", "0"))

        CHART_BUFFER_PERCENTAGE = float(os.getenv("CHART_BUFFER_PERCENTAGE", "0.05"))
        CHART_MIN_STRENGTH_SCORE = int(os.getenv("CHART_MIN_STRENGTH_SCORE", "1"))
        CHART_INCLUDE_KEY_LEVELS_ONLY = os.getenv("CHART_INCLUDE_KEY_LEVELS_ONLY", "false").lower() == 'true'
        CHART_MAX_FVGS_TO_PLOT = int(os.getenv("CHART_MAX_FVGS_TO_PLOT", "5"))
        CHART_MAX_SR_TO_PLOT = int(os.getenv("CHART_MAX_SR_TO_PLOT", "15"))
        CHART_MAX_OB_TO_PLOT = int(os.getenv("CHART_MAX_OB_TO_PLOT", "10"))
        CHART_MAX_MS_TO_PLOT = int(os.getenv("CHART_MAX_MS_TO_PLOT", "10"))
        CHART_MAX_LIQ_TO_PLOT = int(os.getenv("CHART_MAX_LIQ_TO_PLOT", "8"))
        CHART_MAX_FIB_TO_PLOT = int(os.getenv("CHART_MAX_FIB_TO_PLOT", "10"))
        CHART_MAX_FIB_SETS_TO_PLOT = int(os.getenv("CHART_MAX_FIB_SETS_TO_PLOT", "5"))

        ANALYSIS_CONFIGS = {
            "Technical_Trend": {
                "enabled": os.getenv("ANALYST_TECHNICAL_TREND_ENABLED", "true").lower() == 'true',
                "persona_prompt": "Anda adalah seorang analis tren teknikal pasar keuangan yang ahli dalam mengidentifikasi arah pergerakan harga jangka menengah hingga panjang menggunakan moving averages dan struktur pasar yang besar. Fokus pada bias tren (bullish/bearish/sideways) dan konfirmasi tren. Berikan alasan yang jelas berdasarkan konfluensi MA (misalnya 50 SMA di atas 200 SMA) dan Break of Structure (BoS) atau Change of Character (ChoCh) di timeframe tinggi (H4, D1).",
                "confidence_threshold": "Moderate",
                "relevant_timeframes": ["H4", "D1"]
            },
            "Technical_Levels": {
                "enabled": os.getenv("ANALYST_TECHNICAL_LEVELS_ENABLED", "true").lower() == 'true',
                "persona_prompt": "Anda adalah seorang analis level kunci teknikal pasar keuangan yang ahli dalam mengidentifikasi area Supply/Demand, Order Blocks (OB), Fair Value Gaps (FVG), Support/Resistance (S&R), dan Liquidity Zones di timeframe H1 dan H4. Fokus pada di mana harga bereaksi terhadap level-level penting. Berikan rekomendasi trading berdasarkan reaksi harga terhadap level tersebut (misalnya, harga kembali ke OB Bullish, mengisi FVG Bearish).",
                "confidence_threshold": "Moderate",
                "relevant_timeframes": ["H1", "H4"]
            },
            "Technical_Momentum": {
                "enabled": os.getenv("ANALYST_TECHNICAL_MOMENTUM_ENABLED", "true").lower() == 'true',
                "persona_prompt": "Anda adalah seorang analis momentum teknikal pasar keuangan yang ahli dalam mengidentifikasi divergensi (RSI, MACD) dan pola candlestick yang kuat di timeframe M15 dan H1. Fokus pada perubahan momentum dan potensi pembalikan atau kelanjutan tren. Berikan rekomendasi trading berdasarkan sinyal momentum (misalnya, divergensi bullish RSI, pola engulfing candle).",
                "confidence_threshold": "Moderate",
                "relevant_timeframes": ["M15", "H1"]
            },
            "Fundamental_Analyst": {
                "enabled": os.getenv("ANALYST_FUNDAMENTAL_ENABLED", "true").lower() == 'true',
                "persona_prompt": "Anda adalah seorang analis fundamental makroekonomi yang ahli dalam memahami dampak berita ekonomi (NFP, CPI, suku bunga, FOMC) dan peristiwa geopolitik (perang, konflik, kebijakan perdagangan) pada pasar XAUUSD dan USD. Fokus pada interpretasi data fundamental dan implikasinya terhadap arah harga aset. Berikan analisis dan rekomendasi berdasarkan pandangan fundamental Anda.",
                "confidence_threshold": "High",
                "relevant_timeframes": ["Daily"]
            }
        }

    class RuleBasedStrategy:
        """Konfigurasi untuk strategi trading berbasis aturan."""
        DEFAULT_SL_PIPS = Decimal(os.getenv("RULE_SL_PIPS", "50"))
        TP1_PIPS = Decimal(os.getenv("RULE_TP1_PIPS", "20"))
        TP2_PIPS = Decimal(os.getenv("RULE_TP2_PIPS", "30")) # Default 30 pips
        TP3_PIPS = Decimal(os.getenv("RULE_TP3_PIPS", "100")) # Default 100 pips

        # Parameter toleransi yang sekarang akan menggunakan ATR_MULTIPLIER_FOR_TOLERANCE sebagai dasar
        # Ini akan menjadi nilai fallback jika ATR tidak tersedia
        RULE_SR_TOLERANCE_POINTS = Decimal(os.getenv("RULE_SR_TOLERANCE_POINTS", "15"))
        RULE_EQUAL_LEVEL_TOLERANCE_POINTS = Decimal(os.getenv("RULE_EQUAL_LEVEL_TOLERANCE_POINTS", "15"))
        RULE_OB_CONSOLIDATION_TOLERANCE_POINTS = Decimal(os.getenv("RULE_OB_CONSOLIDATION_TOLERANCE_POINTS", "25"))

        EMA_SHORT_PERIOD = int(os.getenv("RULE_EMA_SHORT_PERIOD", "9"))
        EMA_LONG_PERIOD = int(os.getenv("RULE_EMA_LONG_PERIOD", "21"))
        LOOKBACK_CANDLES_LTF = int(os.getenv("RULE_LOOKBACK_LTF", "200"))
        LOOKBACK_CANDLES_HTF = int(os.getenv("RULE_LOOKBACK_HTF", "100"))

        CANDLE_BODY_MIN_RATIO = Decimal(os.getenv("CANDLE_BODY_MIN_RATIO", "0.4"))
        CANDLE_MIN_SIZE_PIPS = Decimal(os.getenv("CANDLE_MIN_SIZE_PIPS", "10.0"))
        STRUCTURE_OFFSET_PIPS = Decimal(os.getenv("STRUCTURE_OFFSET_PIPS", "10.0"))
        OB_SHOULDER_LENGTH = int(os.getenv("OB_SHOULDER_LENGTH", "3")) # Duplikat di MarketData jika sama

        # --- S&R Strength Calculation ---
        SR_STRENGTH_RETEST_WINDOW_CANDLES = int(os.getenv("SR_STRENGTH_RETEST_WINDOW_CANDLES", "50"))
        SR_STRENGTH_BREAK_TOLERANCE_MULTIPLIER = Decimal(os.getenv("SR_STRENGTH_BREAK_TOLERANCE_MULTIPLIER", "1.5"))

        # --- OB Strength Calculation ---
        OB_MIN_IMPULSIVE_CANDLE_BODY_PERCENT = Decimal(os.getenv("OB_MIN_IMPULSIVE_CANDLE_BODY_PERCENT", "0.5"))
        OB_MIN_IMPULSIVE_MOVE_MULTIPLIER = Decimal(os.getenv("OB_MIN_IMPULSIVE_MOVE_MULTIPLIER", "3.0"))
        OB_VOLUME_FACTOR_MULTIPLIER = Decimal(os.getenv("OB_VOLUME_FACTOR_MULTIPLIER", "0.5"))

        # --- FVG Strength Calculation ---
        FVG_MIN_CANDLE_BODY_PERCENT_FOR_STRENGTH = Decimal(os.getenv("FVG_MIN_CANDLE_BODY_PERCENT_FOR_STRENGTH", "0.7"))
        FVG_VOLUME_FACTOR_FOR_STRENGTH = Decimal(os.getenv("FVG_VOLUME_FACTOR_FOR_STRENGTH", "0.5"))

        # --- S&D Impulsive Move Calculation ---
        SD_MIN_IMPULSIVE_MOVE_ATR_MULTIPLIER = Decimal(os.getenv("SD_MIN_IMPULSIVE_MOVE_ATR_MULTIPLIER", "1.5"))

        # --- Divergence Price Tolerance ---
        DIVERGENCE_PRICE_TOLERANCE_ATR_MULTIPLIER = Decimal(os.getenv("DIVERGENCE_PRICE_TOLERANCE_ATR_MULTIPLIER", "0.2"))

        # --- Market Structure Break Tolerance ---
        MS_BREAK_ATR_MULTIPLIER = Decimal(os.getenv("MS_BREAK_ATR_MULTIPLIER", "0.2"))


        # aturan charting ini juga harus di ubah agar bisa di akses chart loader
        CONFLUENCE_PROXIMITY_TOLERANCE_PIPS = Decimal(os.getenv("CONFLUENCE_PROXIMITY_TOLERANCE_PIPS", "10.0"))
        CONFLUENCE_SCORE_PER_LEVEL = int(os.getenv("CONFLUENCE_SCORE_PER_LEVEL", "1"))

        # --- SIGNAL RULES ---
        # Ini adalah struktur data kompleks yang paling baik dikelola secara hardcoded di dalam kode
        # atau dimuat dari file konfigurasi terpisah (misalnya JSON/YAML) jika sangat dinamis.
        # Mencoba menarik ini dari variabel lingkungan string sangat tidak praktis dan rawan error.
        SIGNAL_RULES = [
            {
                "name": "Bullish Rejection from Demand Zone (High Conf.)",
                "action": "BUY",
                "aggressiveness_level": "High-Probability",
                "priority": 1,
                "enabled": True,
                "timeframes": ["H1", "H4"],
                "conditions": [
                    "TREND_H4_BULLISH",
                    "PRICE_IN_ACTIVE_DEMAND_ZONE",
                    "DEMAND_ZONE_STRENGTH_GE_2",
                    "CANDLE_REJECTION_BULLISH_H1",
                    "RSI_H1_OVERSOLD_OR_NOT_OVERBOUGHT",
                    "MACD_H1_BULLISH_CROSS_OR_POSITIVE"
                ],
                "entry_price_logic": "DEMAND_ZONE_TOP",
                "stop_loss_logic": "BELOW_DEMAND_ZONE_LOW",
                "take_profit_logic": "AT_NEXT_SUPPLY_ZONE",
                "risk_reward_ratio_min": Decimal('1.5')
            },
            {
                "name": "Bearish Rejection from Supply Zone (Moderate)",
                "action": "SELL",
                "aggressiveness_level": "Moderate",
                "priority": 2,
                "enabled": True,
                "timeframes": ["H1"],
                "conditions": [
                    "TREND_H1_BEARISH",
                    "PRICE_IN_ACTIVE_SUPPLY_ZONE",
                    "SUPPLY_ZONE_STRENGTH_GE_1",
                    "CANDLE_REJECTION_BEARISH_H1",
                    "RSI_H1_OVERBOUGHT_OR_NOT_OVERSOLD"
                ],
                "entry_price_logic": "SUPPLY_ZONE_BOTTOM",
                "stop_loss_logic": "ABOVE_SUPPLY_ZONE_HIGH",
                "take_profit_logic": "AT_NEXT_DEMAND_ZONE",
                "risk_reward_ratio_min": Decimal('1.0')
            },
            {
                "name": "Trend Continuation - Pullback to OB (Aggressive)",
                "action": "BUY",
                "aggressiveness_level": "Aggressive",
                "priority": 3,
                "enabled": False,
                "timeframes": ["M15"],
                "conditions": [
                    "TREND_M15_BULLISH",
                    "PRICE_IN_ACTIVE_BULLISH_OB",
                    "OB_MITIGATED_ONCE"
                ],
                "entry_price_logic": "OB_MIDPOINT",
                "stop_loss_logic": "BELOW_OB_LOW",
                "take_profit_logic": "FIXED_RR_1_5",
                "risk_reward_ratio_min": Decimal('1.5')
            },
        ]


    class Telegram:
        """Konfigurasi untuk notifikasi Telegram."""
        NOTIFICATION_COOLDOWN_SECONDS = int(os.getenv("TELEGRAM_NOTIFICATION_COOLDOWN_SECONDS", "14400"))
        MAX_EVENTS_TO_NOTIFY = int(os.getenv("TELEGRAM_MAX_EVENTS_TO_NOTIFY", "10"))
        MAX_ARTICLES_TO_NOTIFY = int(os.getenv("TELEGRAM_MAX_ARTICLES_TO_NOTIFY", "10"))
        NOTIF_MAX_LEVELS_PER_TYPE_PER_TF = int(os.getenv("TELEGRAM_NOTIF_MAX_LEVELS_PER_TYPE_PER_TF", "3"))
        NOTIF_MAX_FVG_PER_TF = int(os.getenv("TELEGRAM_NOTIF_MAX_FVG_PER_TF", "3"))
        NOTIF_MAX_OB_PER_TF = int(os.getenv("TELEGRAM_NOTIF_MAX_OB_PER_TF", "3"))
        NOTIF_MAX_KEY_LEVELS_PER_TF = int(os.getenv("TELEGRAM_NOTIF_MAX_KEY_LEVELS_PER_TF", "5"))
        NOTIF_MAX_RESISTANCE_PER_TF = int(os.getenv("TELEGRAM_NOTIF_MAX_RESISTANCE_PER_TF", "3"))
        NOTIF_MAX_SUPPORT_PER_TF = int(os.getenv("TELEGRAM_NOTIF_MAX_SUPPORT_PER_TF", "3"))
        NOTIF_MAX_FIBO_PER_TF = int(os.getenv("TELEGRAM_NOTIF_MAX_FIBO_PER_TF", "0"))
        NOTIF_MAX_SWING_PER_TF = int(os.getenv("TELEGRAM_NOTIF_MAX_SWING_PER_TF", "1"))

        SEND_SIGNAL_NOTIFICATIONS = os.getenv("TELEGRAM_SEND_SIGNAL_NOTIFICATIONS", "true").lower() == 'true'
        SEND_TRADE_NOTIFICATIONS = os.getenv("TELEGRAM_SEND_TRADE_NOTIFICATIONS", "true").lower() == 'true'
        SEND_ACCOUNT_NOTIFICATIONS = os.getenv("TELEGRAM_SEND_ACCOUNT_NOTIFICATIONS", "true").lower() == 'true'
        SEND_DAILY_SUMMARY = os.getenv("TELEGRAM_SEND_DAILY_SUMMARY", "true").lower() == 'true'
        SEND_ERROR_NOTIFICATIONS = os.getenv("TELEGRAM_SEND_ERROR_NOTIFICATIONS", "true").lower() == 'true'
        SEND_APP_STATUS_NOTIFICATIONS = os.getenv("TELEGRAM_SEND_APP_STATUS_NOTIFICATIONS", "true").lower() == 'true'
        SEND_FUNDAMENTAL_NOTIFICATIONS = os.getenv("TELEGRAM_SEND_FUNDAMENTAL_NOTIFICATIONS", "true").lower() == 'true'
        SEND_INDIVIDUAL_ANALYST_SIGNALS = os.getenv("TELEGRAM_SEND_INDIVIDUAL_ANALYST_SIGNALS", "false").lower() == 'true'


    class System:
        """Konfigurasi terkait pengaturan sistem umum dan penanganan error."""
        MAX_RETRIES = int(os.getenv("SYSTEM_MAX_RETRIES", "5")) # Maksimal percobaan ulang
        RETRY_DELAY_SECONDS = Decimal(os.getenv("SYSTEM_RETRY_DELAY_SECONDS", "0.5")) # Delay retry
        DATABASE_BATCH_SIZE = int(os.getenv("SYSTEM_DATABASE_BATCH_SIZE", "1000")) # Ukuran batch database

    class Monitoring:
        """Konfigurasi untuk pemantauan kesehatan detektor."""
        DETECTOR_ZERO_DETECTIONS_THRESHOLD = int(os.getenv("DETECTOR_ZERO_DETECTIONS_THRESHOLD", "3"))
        DETECTOR_HEALTH_CHECK_INTERVAL_SECONDS = int(os.getenv("DETECTOR_HEALTH_CHECK_INTERVAL_SECONDS", "3600"))


    @classmethod
    def validate_config(cls):
        """
        Validasi semua konfigurasi setelah dimuat dari environment variables.
        Ini memastikan tipe data yang benar dan nilai yang masuk akal untuk setiap parameter.
        Metode ini dipanggil saat modul config dimuat.
        """
        # Validasi PARTIAL_TP_LEVELS (saat ini hardcoded, validasi memastikan strukturnya benar)
        if not isinstance(cls.Trading.PARTIAL_TP_LEVELS, list):
            raise ValueError("PARTIAL_TP_LEVELS must be a list.")
        for i, tp_level in enumerate(cls.Trading.PARTIAL_TP_LEVELS):
            if not isinstance(tp_level, dict):
                raise ValueError(f"PARTIAL_TP_LEVELS[{i}] must be a dictionary.")
            if 'price_multiplier' not in tp_level or not isinstance(tp_level['price_multiplier'], Decimal):
                raise ValueError(f"PARTIAL_TP_LEVELS[{i}] missing or invalid 'price_multiplier' (must be Decimal).")
            if tp_level['price_multiplier'] <= 0:
                raise ValueError(f"PARTIAL_TP_LEVELS[{i}] 'price_multiplier' must be greater than 0.")
            if 'volume_percentage_to_close' not in tp_level or not isinstance(tp_level['volume_percentage_to_close'], Decimal):
                raise ValueError(f"PARTIAL_TP_LEVELS[{i}] missing or invalid 'volume_percentage_to_close' (must be Decimal).")
            if not (Decimal('0.0') <= tp_level['volume_percentage_to_close'] <= Decimal('1.0')):
                raise ValueError(f"PARTIAL_TP_LEVELS[{i}] 'volume_percentage_to_close' must be between 0.0 and 1.0.")
            if 'move_sl_to_breakeven_after_partial' not in tp_level or not isinstance(tp_level['move_sl_to_breakeven_after_partial'], bool):
                raise ValueError(f"PARTIAL_TP_LEVELS[{i}] missing or invalid 'move_sl_to_breakeven_after_partial' (must be boolean).")
            if 'move_sl_to_price_after_partial' in tp_level and tp_level['move_sl_to_price_after_partial'] is not None and not isinstance(tp_level['move_sl_to_price_after_partial'], Decimal):
                raise ValueError(f"PARTIAL_TP_LEVELS[{i}] 'move_sl_to_price_after_partial' must be Decimal or None.")


        # Validasi dan set Daily Profit/Loss Limit
        daily_profit_target_env = os.getenv("DAILY_PROFIT_TARGET_PERCENT")
        if daily_profit_target_env is not None:
            try:
                cls.Trading.DAILY_PROFIT_TARGET_PERCENT = Decimal(daily_profit_target_env)
            except Exception:
                raise ValueError("DAILY_PROFIT_TARGET_PERCENT must be a valid decimal number.")
        if cls.Trading.DAILY_PROFIT_TARGET_PERCENT < 0:
            raise ValueError("DAILY_PROFIT_TARGET_PERCENT must be non-negative.")

        daily_loss_limit_env = os.getenv("DAILY_LOSS_LIMIT_PERCENT")
        if daily_loss_limit_env is not None:
            try:
                cls.Trading.DAILY_LOSS_LIMIT_PERCENT = Decimal(daily_loss_limit_env)
            except Exception:
                raise ValueError("DAILY_LOSS_LIMIT_PERCENT must be a valid decimal number.")
        if cls.Trading.DAILY_LOSS_LIMIT_PERCENT < 0:
            raise ValueError("DAILY_LOSS_LIMIT_PERCENT must be non-negative.")

        # Validasi dan set TRADING_SYMBOL_POINT_VALUE
        trading_symbol_point_value_env = os.getenv("TRADING_SYMBOL_POINT_VALUE")
        if trading_symbol_point_value_env is not None:
            try:
                cls.TRADING_SYMBOL_POINT_VALUE = Decimal(trading_symbol_point_value_env)
            except Exception:
                raise ValueError("TRADING_SYMBOL_POINT_VALUE must be a valid decimal number.")

        # Validasi dan set PIP_UNIT_IN_DOLLAR
        pip_unit_in_dollar_env = os.getenv("PIP_UNIT_IN_DOLLAR")
        if pip_unit_in_dollar_env is not None:
            try:
                cls.PIP_UNIT_IN_DOLLAR = Decimal(pip_unit_in_dollar_env)
            except Exception:
                raise ValueError("PIP_UNIT_IN_DOLLAR must be a valid decimal number.")

        # Validasi dan set MA_TREND_PERIODS (dari string dipisahkan koma di .env)
        ma_trend_periods_env = os.getenv("MA_TREND_PERIODS")
        if ma_trend_periods_env:
            try:
                cls.MarketData.MA_TREND_PERIODS = [Decimal(p.strip()) for p in ma_trend_periods_env.split(',')]
            except Exception:
                raise ValueError("MA_TREND_PERIODS must be a comma-separated list of valid decimal numbers (e.g., '20,50,200').")

        # Validasi dan set FIBO_RETRACTION_LEVELS (dari string dipisahkan koma di .env)
        fibo_retraction_levels_env = os.getenv("FIBO_RETRACTION_LEVELS")
        if fibo_retraction_levels_env:
            try:
                cls.MarketData.FIBO_RETRACTION_LEVELS = [Decimal(r.strip()) for r in fibo_retraction_levels_env.split(',')]
            except Exception:
                raise ValueError("FIBO_RETRACTION_LEVELS must be a comma-separated list of valid decimal numbers (e.g., '0.236,0.382,0.5').")


        # Loop validasi dan penimpaan nilai dari .env ke properti kelas yang sesuai.
        # Ini mencakup sebagian besar parameter numerik dan string.
        attrs_to_validate_from_env = {
            # Class Config (root)
            'TRADING_SYMBOL': str,
            'DEFAULT_TIMEFRAME': str,
            # Class Trading
            'AUTO_TRADE_VOLUME': Decimal,
            'BACKTEST_HIT_TOLERANCE_POINTS': Decimal,
            'AUTO_TRADE_SLIPPAGE': int,
            'AUTO_TRADE_MAGIC_NUMBER': int,
            'RISK_PER_TRADE_PERCENT': Decimal,
            'MAX_DAILY_DRAWDOWN_PERCENT': Decimal,
            'TRAILING_STOP_PIPS': Decimal,
            'TRAILING_STOP_STEP_PIPS': Decimal,
            'TRADING_START_HOUR_UTC': int,
            'TRADING_END_HOUR_UTC': int,
            'MARKET_CLOSE_BUFFER_MINUTES': int,
            'MARKET_OPEN_BUFFER_MINUTES': int,
            'MIN_SL_PIPS': Decimal,
            'MIN_TP_PIPS': Decimal,
            'SWAP_RATE_PER_LOT_BUY': Decimal,
            'SWAP_RATE_PER_LOT_SELL': Decimal,

            # Class Scheduler
            'AUTO_DATA_UPDATE_ENABLED': bool,

            # Class Sessions
            'ASIA_SESSION_START_HOUR_UTC': int,
            'ASIA_SESSION_END_HOUR_UTC': int,
            'EUROPE_SESSION_START_HOUR_UTC': int,
            'EUROPE_SESSION_END_HOUR_UTC': int,
            'NEWYORK_SESSION_START_HOUR_UTC': int,
            'NEWYORK_SESSION_END_HOUR_UTC': int,
            'TIMEZONE_FOR_DST_CHECK': str,

            # Class MarketData (Detektor & Umum)
            'ATR_PERIOD': int,
            'ATR_MULTIPLIER_FOR_TOLERANCE': Decimal,
            'SR_LOOKBACK_CANDLES': int,
            'SR_ZONE_ATR_MULTIPLIER': Decimal,
            'MIN_SR_STRENGTH': int,
            'FVG_MIN_ATR_MULTIPLIER': Decimal,
            'OB_MIN_VOLUME_MULTIPLIER': Decimal,
            'OB_FVG_MITIGATION_LOOKBACK_CANDLES': int,
            'LIQUIDITY_CANDLE_RANGE_PERCENT': Decimal,
            'BOS_CHOCH_MIN_PIPS_CONFIRMATION': Decimal,
            'SWING_LOOKBACK_CANDLES': int,
            'RSI_DIVERGENCE_PERIODS': int,
            'MACD_DIVERGENCE_FAST_PERIOD': int,
            'MACD_DIVERGENCE_SLOW_PERIOD': int,
            'MACD_DIVERGENCE_SIGNAL_PERIOD': int,
            'ENABLE_SR_DETECTION': bool,
            'ENABLE_OB_FVG_DETECTION': bool,
            'ENABLE_LIQUIDITY_DETECTION': bool,
            'ENABLE_FIBONACCI_DETECTION': bool,
            'ENABLE_MARKET_STRUCTURE_DETECTION': bool,
            'ENABLE_SWING_DETECTION': bool,
            'ENABLE_DIVERGENCE_DETECTION': bool,
            'ENABLE_RSI_CALCULATION': bool,
            'ENABLE_MACD_CALCULATION': bool,
            'ENABLE_EMA_CROSS_DETECTION': bool,
            'ENABLE_MA_TREND_DETECTION': bool,
            'ENABLE_VOLUME_PROFILE_DETECTION': bool,
            'ENABLE_PREVIOUS_HIGH_LOW_DETECTION': bool,
            'CONFLUENCE_PROXIMITY_TOLERANCE_PIPS': Decimal,
            'CONFLUENCE_SCORE_PER_LEVEL': int,
            'OB_CONSOLIDATION_TOLERANCE_POINTS': Decimal,
            'OB_SHOULDER_LENGTH': int,
            'FVG_MIN_CANDLE_BODY_PERCENT_FOR_STRENGTH': Decimal,
            'FVG_VOLUME_FACTOR_FOR_STRENGTH': Decimal,
            'HISTORICAL_DATA_START_DATE_FULL': str,
            # Tambahkan atribut yang diduplikasi di MarketData
            'RSI_PERIOD': int,
            'RSI_OVERBOUGHT_LEVEL': Decimal,
            'RSI_OVERSOLD_LEVEL': Decimal,
            'MACD_FAST_PERIOD': int,
            'MACD_SLOW_PERIOD': int,
            'MACD_SIGNAL_PERIOD': int,


            # Class AIAnalysts
            'OPENAI_MODEL_NAME': str,
            'OPENAI_TEMPERATURE': float,
            'AI_MIN_ANALYSTS_FOR_CONSENSUS': int,
            'FVG_MIN_DOLLARS': Decimal,
            'SWING_EXT_BARS': int,
            # --- ATRIBUT INI DIKEMBALIKAN KE AIAnalysts (dan diduplikasi) ---
            'RSI_PERIOD': int,
            'RSI_OVERBOUGHT': Decimal,
            'RSI_OVERSOLD': Decimal,
            'MACD_FAST_PERIOD': int,
            'MACD_SLOW_PERIOD': int,
            'MACD_SIGNAL_PERIOD': int,
            # --- AKHIR PENGEMBALIAN / DUPLIKASI ---
            'MAX_FVG_DISPLAY': int,
            'MAX_OB_DISPLAY': int,
            'MAX_KEY_LEVEL_DISPLAY': int,
            'MAX_SR_DISPLAY': int,
            'MAX_LIQUIDITY_DISPLAY': int,
            'MAX_DIVERGENCE_DISPLAY': int,
            'CHART_BUFFER_PERCENTAGE': float,
            'CHART_MIN_STRENGTH_SCORE': int,
            'CHART_INCLUDE_KEY_LEVELS_ONLY': bool,
            'CHART_MAX_FVGS_TO_PLOT': int,
            'CHART_MAX_SR_TO_PLOT': int,
            'CHART_MAX_OB_TO_PLOT': int,
            'CHART_MAX_MS_TO_PLOT': int,
            'CHART_MAX_LIQ_TO_PLOT': int,
            'CHART_MAX_FIB_TO_PLOT': int,
            'CHART_MAX_FIB_SETS_TO_PLOT': int,

            # Class RuleBasedStrategy
            'DEFAULT_SL_PIPS': Decimal,
            'TP1_PIPS': Decimal,
            'TP2_PIPS': Decimal,
            'TP3_PIPS': Decimal,
            'RULE_SR_TOLERANCE_POINTS': Decimal,
            'RULE_EQUAL_LEVEL_TOLERANCE_POINTS': Decimal,
            'RULE_OB_CONSOLIDATION_TOLERANCE_POINTS': Decimal,
            'EMA_SHORT_PERIOD': int,
            'EMA_LONG_PERIOD': int,
            'LOOKBACK_CANDLES_LTF': int,
            'LOOKBACK_CANDLES_HTF': int,
            'CANDLE_BODY_MIN_RATIO': Decimal,
            'CANDLE_MIN_SIZE_PIPS': Decimal,
            'STRUCTURE_OFFSET_PIPS': Decimal,
            'SR_STRENGTH_RETEST_WINDOW_CANDLES': int,
            'SR_STRENGTH_BREAK_TOLERANCE_MULTIPLIER': Decimal,
            'OB_MIN_IMPULSIVE_CANDLE_BODY_PERCENT': Decimal,
            'OB_MIN_IMPULSIVE_MOVE_MULTIPLIER': Decimal,
            'OB_VOLUME_FACTOR_MULTIPLIER': Decimal,
            'FVG_MIN_CANDLE_BODY_PERCENT_FOR_STRENGTH': Decimal,
            'FVG_VOLUME_FACTOR_FOR_STRENGTH': Decimal,
            'SD_MIN_IMPULSIVE_MOVE_ATR_MULTIPLIER': Decimal,
            'DIVERGENCE_PRICE_TOLERANCE_ATR_MULTIPLIER': Decimal,
            'MS_BREAK_ATR_MULTIPLIER': Decimal,

            # Class Telegram
            'TELEGRAM_NOTIFICATION_COOLDOWN_SECONDS': int,
            'MAX_EVENTS_TO_NOTIFY': int,
            'MAX_ARTICLES_TO_NOTIFY': int,
            'NOTIF_MAX_LEVELS_PER_TYPE_PER_TF': int,
            'NOTIF_MAX_FVG_PER_TF': int,
            'NOTIF_MAX_OB_PER_TF': int,
            'NOTIF_MAX_KEY_LEVELS_PER_TF': int,
            'NOTIF_MAX_RESISTANCE_PER_TF': int,
            'NOTIF_MAX_SUPPORT_PER_TF': int,
            'NOTIF_MAX_FIBO_PER_TF': int,
            'NOTIF_MAX_SWING_PER_TF': int,
            'SEND_SIGNAL_NOTIFICATIONS': bool,
            'SEND_TRADE_NOTIFICATIONS': bool,
            'SEND_ACCOUNT_NOTIFICATIONS': bool,
            'SEND_DAILY_SUMMARY': bool,
            'SEND_ERROR_NOTIFICATIONS': bool,
            'SEND_APP_STATUS_NOTIFICATIONS': bool,
            'SEND_FUNDAMENTAL_NOTIFICATIONS': bool,
            'SEND_INDIVIDUAL_ANALYST_SIGNALS': bool,

            # Class System
            'MAX_RETRIES': int,
            'RETRY_DELAY_SECONDS': Decimal,
            'DATABASE_BATCH_SIZE': int,

            # Class Monitoring
            'DETECTOR_ZERO_DETECTIONS_THRESHOLD': int,
            'DETECTOR_HEALTH_CHECK_INTERVAL_SECONDS': int,
        }

        # Loop untuk validasi dan penimpaan nilai dari .env ke properti kelas yang sesuai.
        for attr_name, attr_type in attrs_to_validate_from_env.items():
            env_var_name = attr_name.upper()
            env_val = os.getenv(env_var_name)

            if env_val is not None:
                # Perbarui di MarketData jika atribut ada di sana
                if hasattr(cls.MarketData, attr_name):
                    try:
                        if attr_type == bool:
                            setattr(cls.MarketData, attr_name, env_val.lower() == 'true')
                        else:
                            setattr(cls.MarketData, attr_name, attr_type(env_val))
                    except ValueError:
                        raise ValueError(
                            f"MarketData.{attr_name} from environment ('{env_var_name}') "
                            f"is not a valid {attr_type.__name__}: '{env_val}'. Check your .env file."
                        )
                # Perbarui di AIAnalysts jika atribut ada di sana
                if hasattr(cls.AIAnalysts, attr_name):
                    try:
                        if attr_type == bool:
                            setattr(cls.AIAnalysts, attr_name, env_val.lower() == 'true')
                        else:
                            setattr(cls.AIAnalysts, attr_name, attr_type(env_val))
                    except ValueError:
                        raise ValueError(
                            f"AIAnalysts.{attr_name} from environment ('{env_var_name}') "
                            f"is not a valid {attr_type.__name__}: '{env_val}'. Check your .env file."
                        )
                # Lanjutkan dengan kelas lain yang tidak diduplikasi
                if hasattr(cls, attr_name): # Untuk atribut di root Config
                    try:
                        if attr_type == bool:
                            setattr(cls, attr_name, env_val.lower() == 'true')
                        else:
                            setattr(cls, attr_name, attr_type(env_val))
                    except ValueError:
                        raise ValueError(f"Config.{attr_name} from environment ('{env_var_name}') is not a valid {attr_type.__name__}: '{env_val}'. Check your .env file.")
                elif hasattr(cls.Trading, attr_name):
                    try:
                        if attr_type == bool:
                            setattr(cls.Trading, attr_name, env_val.lower() == 'true')
                        else:
                            setattr(cls.Trading, attr_name, attr_type(env_val))
                    except ValueError:
                        raise ValueError(f"Trading.{attr_name} from environment ('{env_var_name}') is not a valid {attr_type.__name__}: '{env_val}'. Check your .env file.")
                elif hasattr(cls.Scheduler, attr_name):
                    try:
                        if attr_type == bool:
                            setattr(cls.Scheduler, attr_name, env_val.lower() == 'true')
                        else:
                            setattr(cls.Scheduler, attr_name, attr_type(env_val))
                    except ValueError:
                        raise ValueError(f"Scheduler.{attr_name} from environment ('{env_var_name}') is not a valid {attr_type.__name__}: '{env_val}'. Check your .env file.")
                elif hasattr(cls.Sessions, attr_name):
                    try:
                        if attr_type == bool:
                            setattr(cls.Sessions, attr_name, env_val.lower() == 'true')
                        else:
                            setattr(cls.Sessions, attr_name, attr_type(env_val))
                    except ValueError:
                        raise ValueError(f"Sessions.{attr_name} from environment ('{env_var_name}') is not a valid {attr_type.__name__}: '{env_val}'. Check your .env file.")
                elif hasattr(cls.RuleBasedStrategy, attr_name):
                    try:
                        if attr_type == bool:
                            setattr(cls.RuleBasedStrategy, attr_name, env_val.lower() == 'true')
                        else:
                            setattr(cls.RuleBasedStrategy, attr_name, attr_type(env_val))
                    except ValueError:
                        raise ValueError(f"RuleBasedStrategy.{attr_name} from environment ('{env_var_name}') is not a valid {attr_type.__name__}: '{env_val}'. Check your .env file.")
                elif hasattr(cls.Telegram, attr_name):
                    try:
                        if attr_type == bool:
                            setattr(cls.Telegram, attr_name, env_val.lower() == 'true')
                        else:
                            setattr(cls.Telegram, attr_name, attr_type(env_val))
                    except ValueError:
                        raise ValueError(f"Telegram.{attr_name} from environment ('{env_var_name}') is not a valid {attr_type.__name__}: '{env_val}'. Check your .env file.")
                elif hasattr(cls.System, attr_name):
                    try:
                        if attr_type == bool:
                            setattr(cls.System, attr_name, env_val.lower() == 'true')
                        else:
                            setattr(cls.System, attr_name, attr_type(env_val))
                    except ValueError:
                        raise ValueError(f"System.{attr_name} from environment ('{env_var_name}') is not a valid {attr_type.__name__}: '{env_val}'. Check your .env file.")
                elif hasattr(cls.Monitoring, attr_name):
                    try:
                        if attr_type == bool:
                            setattr(cls.Monitoring, attr_name, env_val.lower() == 'true')
                        else:
                            setattr(cls.Monitoring, attr_name, attr_type(env_val))
                    except ValueError:
                        raise ValueError(f"Monitoring.{attr_name} from environment ('{env_var_name}') is not a valid {attr_type.__name__}: '{env_val}'. Check your .env file.")


        # Validasi khusus untuk Scheduler.UPDATE_INTERVALS (nilai-nilai float)
        for key in cls.Scheduler.UPDATE_INTERVALS.keys():
            env_var_name = key.upper().replace('.', '_')
            env_val = os.getenv(env_var_name)
            if env_val is not None:
                try:
                    cls.Scheduler.UPDATE_INTERVALS[key] = float(env_val)
                except ValueError:
                    raise ValueError(
                        f"Scheduler.UPDATE_INTERVALS['{key}'] from environment ('{env_var_name}') "
                        f"is not a valid float: '{env_val}'. Check .env file."
                    )

        # Validasi untuk Scheduler.ENABLED_LOOPS (nilai-nilai boolean)
        for key in cls.Scheduler.ENABLED_LOOPS.keys():
            env_var_name = key.upper().replace('.', '_')
            env_val = os.getenv(env_var_name)
            if env_val is not None:
                if env_val.lower() not in ['true', 'false']:
                    raise ValueError(
                        f"Scheduler.ENABLED_LOOPS['{key}'] from environment ('{env_var_name}') "
                        f"must be 'true' or 'false': '{env_val}'. Check .env file."
                    )
                cls.Scheduler.ENABLED_LOOPS[key] = env_val.lower() == 'true'

        # Validasi untuk MarketData.ENABLED_TIMEFRAMES (nilai-nilai boolean)
        for key in cls.MarketData.ENABLED_TIMEFRAMES.keys():
            env_var_name = f"ENABLE_{key.upper()}_TF"
            env_val = os.getenv(env_var_name)
            if env_val is not None:
                if env_val.lower() not in ['true', 'false']:
                    raise ValueError(
                        f"MarketData.ENABLED_TIMEFRAMES['{key}'] from environment ('{env_var_name}') "
                        f"must be 'true' or 'false': '{env_val}'. Check .env file."
                    )
                cls.MarketData.ENABLED_TIMEFRAMES[key] = env_val.lower() == 'true'

        # Validasi untuk AIAnalysts.ANALYSIS_CONFIGS (field 'enabled')
        for analyst_name, analyst_config in cls.AIAnalysts.ANALYSIS_CONFIGS.items():
            env_var_name = f"ANALYST_{analyst_name.upper()}_ENABLED"
            env_val = os.getenv(env_var_name)
            if env_val is not None:
                if env_val.lower() not in ['true', 'false']:
                    raise ValueError(
                        f"AIAnalysts.ANALYSIS_CONFIGS['{analyst_name}']['enabled'] from environment ('{env_var_name}') "
                        f"must be 'true' or 'false': '{env_val}'. Check .env file."
                    )
                analyst_config["enabled"] = env_val.lower() == 'true'

        # Validasi format tanggal untuk Historical Data Start Date
        try:
            datetime.strptime(cls.MarketData.HISTORICAL_DATA_START_DATE_FULL, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            raise ValueError(
                f"HISTORICAL_DATA_START_DATE_FULL ('{cls.MarketData.HISTORICAL_DATA_START_DATE_FULL}') "
                f"must be in YYYY-MM-DD format. Check .env file."
            )

        # Validasi `RuleBasedStrategy.SIGNAL_RULES` (struktur dasarnya saja)
        # Ini adalah struktur hardcoded, validasi memastikan integritasnya.
        if not isinstance(cls.RuleBasedStrategy.SIGNAL_RULES, list):
            raise ValueError("RuleBasedStrategy.SIGNAL_RULES must be a list.")
        for i, rule in enumerate(cls.RuleBasedStrategy.SIGNAL_RULES):
            if not isinstance(rule, dict):
                raise ValueError(f"RuleBasedStrategy.SIGNAL_RULES[{i}] must be a dictionary.")
            required_keys = ["name", "action", "aggressiveness_level", "priority", "enabled", "timeframes", "conditions", "entry_price_logic", "stop_loss_logic", "take_profit_logic", "risk_reward_ratio_min"]
            if not all(k in rule for k in required_keys):
                missing_keys = [k for k in required_keys if k not in rule]
                raise ValueError(f"RuleBasedStrategy.SIGNAL_RULES[{i}] is missing required keys: {', '.join(missing_keys)}.")
            if rule['action'] not in ["BUY", "SELL", "HOLD"]:
                raise ValueError(f"RuleBasedStrategy.SIGNAL_RULES[{i}] 'action' must be 'BUY', 'SELL', or 'HOLD'.")
            if not isinstance(rule['enabled'], bool):
                raise ValueError(f"RuleBasedStrategy.SIGNAL_RULES[{i}] 'enabled' must be boolean.")
            if not isinstance(rule['timeframes'], list) or not all(isinstance(tf, str) for tf in rule['timeframes']):
                raise ValueError(f"RuleBasedStrategy.SIGNAL_RULES[{i}] 'timeframes' must be a list of strings.")
            if not isinstance(rule['conditions'], list) or not all(isinstance(c, str) for c in rule['conditions']):
                raise ValueError(f"RuleBasedStrategy.SIGNAL_RULES[{i}] 'conditions' must be a list of strings.")
            if not isinstance(rule['risk_reward_ratio_min'], Decimal) or rule['risk_reward_ratio_min'] <= 0:
                raise ValueError(f"RuleBasedStrategy.SIGNAL_RULES[{i}] 'risk_reward_ratio_min' must be a Decimal > 0.")


# Instansiasi objek Config dan panggil validasi saat modul dimuat.
# Objek 'config' ini yang akan diimpor oleh modul lain.
config = Config()
config.validate_config()
