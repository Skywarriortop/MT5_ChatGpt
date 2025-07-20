# data_updater.py

import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal 
import pandas as pd
from sqlalchemy.exc import OperationalError 
import glob
import os
import threading # <<< PENTING: Import ini >>>

# Impor modul lain yang dibutuhkan
import database_manager
from config import config
import mt5_connector
import market_data_processor
import notification_service
import auto_trade_manager 
import fundamental_data_service
import chart_generator
import ai_analyzer 
import ai_consensus_manager 
import detector_monitoring 
import utils 
import rule_based_signal_generator 
import detector_monitoring

logger = logging.getLogger(__name__)

# --- INI ADALAH DEKLARASI VARIABEL GLOBAL (DIKELOLA SECARA INTERNAL OLEH MODUL INI) ---
_latest_realtime_price = Decimal('0.0') 
_latest_real_time_price_timestamp_utc = None # <-- Ini variabel yang benar
_daily_realized_pnl = Decimal('0.0')
_last_daily_pnl_calculation_time = None

last_tick_time_msc = 0
_tick_buffer = []
_last_tick_save_time = time.time()
_TICK_SAVE_INTERVAL_SECONDS = 5
_MAX_TICK_BUFFER_SIZE = 50
_last_raw_tick_mt5 = None 
# --- AKHIR DEKLARASI VARIABEL GLOBAL ---
_fundamental_service_instance = fundamental_data_service.FundamentalDataService()

# --- Fungsi untuk mendapatkan harga real-time terakhir ---
def get_latest_realtime_price() -> tuple[Decimal, datetime]:
    """
    Mengembalikan harga real-time terakhir yang diketahui dan timestamp-nya.
    """
    global _latest_realtime_price, _latest_real_time_price_timestamp_utc # <<< Pastikan nama variabel yang di-global di sini sudah benar
    return _latest_realtime_price, _latest_real_time_price_timestamp_utc # <<< Pastikan nama variabel yang di-return di sini sudah benar


# --- Fungsi untuk mendapatkan P/L harian yang terealisasi ---
def get_daily_realized_pnl() -> Decimal:
    """
    Mengembalikan nilai P/L harian terealisasi terakhir.
    """
    global _daily_realized_pnl
    return _daily_realized_pnl

# --- FUNCTIONS FOR PERIODIC LOOPS ---

def periodic_realtime_tick_loop(symbol: str, stop_event: threading.Event):
    """
    Mengambil tick real-time, menghitung perubahannya, dan menyimpannya ke database.
    Loop ini berjalan secara periodik.
    """
    if symbol is None or not isinstance(symbol, str):
        logger.error("periodic_realtime_tick_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return

    global _latest_realtime_price, _latest_real_time_price_timestamp_utc
    last_tick = None

    while not stop_event.is_set():
        try:
            new_tick_data = mt5_connector.get_current_tick_info(symbol)

            if new_tick_data and new_tick_data.get('last') is not None and new_tick_data.get('time') is not None:
                last_tick = new_tick_data
                _latest_realtime_price = utils.to_decimal_or_none(last_tick['last'])
                _latest_real_time_price_timestamp_utc = utils.to_utc_datetime_or_none(last_tick['time'])
                # Logger ini seharusnya muncul jika berhasil mendapatkan dan mengupdate tick
                logger.debug(f"Tick realtime updated: Price={float(_latest_realtime_price):.5f}, Time={_latest_real_time_price_timestamp_utc.isoformat()}") # <<< Log ini
            else:
                logger.warning(f"Gagal mengambil tick untuk {symbol}. mt5.symbol_info_tick() mengembalikan None atau tick tidak valid. Melewatkan pemrosesan tick ini.")
                time.sleep(float(config.Scheduler.UPDATE_INTERVALS["periodic_realtime_tick_loop"]))
                continue



            tick_timestamp_utc_dt = utils.to_utc_datetime_or_none(last_tick.get('time'))
            tick_last_price = utils.to_decimal_or_none(last_tick.get('last'))

            if tick_timestamp_utc_dt is None or tick_last_price is None:
                logger.warning(f"Tick data tidak lengkap atau tidak valid untuk {symbol}. Melewatkan pemrosesan tick ini.")
                time.sleep(float(config.Scheduler.UPDATE_INTERVALS["periodic_realtime_tick_loop"])) 
                continue

            daily_open_price = database_manager.get_daily_open_price(
                date_ref=datetime.now(timezone.utc).date().isoformat(),
                symbol_param=symbol
            )

            if daily_open_price is None or daily_open_price == Decimal('0.0'):
                logger.debug("Daily Open Price belum tersedia, mencoba mengupdatenya.")
                market_data_processor.update_daily_open_prices_logic(symbol)
                daily_open_price = database_manager.get_daily_open_price(
                    date_ref=datetime.now(timezone.utc).date().isoformat(),
                    symbol_param=symbol
                )
                if daily_open_price is None or daily_open_price == Decimal('0.0'):
                    logger.warning("Gagal mendapatkan Daily Open Price setelah update. Menggunakan harga tick terakhir sebagai fallback untuk perhitungan delta/change.")
                    daily_open_price = tick_last_price 

            delta_point = Decimal('0.0')
            change = Decimal('0.0')
            change_percent = Decimal('0.0')

            if daily_open_price is not None and daily_open_price != Decimal('0.0'):
                delta_point = tick_last_price - daily_open_price
                change = delta_point
                if daily_open_price != Decimal('0.0'):
                    change_percent = (change / daily_open_price) * Decimal('100.0')

            tick_timestamp_utc_ms_int = int(tick_timestamp_utc_dt.timestamp() * 1000)

            database_manager.save_price_tick(
                symbol=symbol,
                time=tick_timestamp_utc_ms_int,
                time_utc_datetime=tick_timestamp_utc_dt,
                last_price=tick_last_price,
                bid_price=utils.to_decimal_or_none(last_tick.get('bid')),
                daily_open_price=daily_open_price,
                delta_point=delta_point,
                change=change,
                change_percent=change_percent
            )

            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["periodic_realtime_tick_loop"])) 

        except Exception as e:
            logger.error(f"Error tak terduga di periodic_realtime_tick_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2) 
    logger.info(f"periodic_realtime_tick_loop untuk {symbol} dihentikan.") 

def periodic_session_data_update_loop(symbol: str, stop_event: threading.Event): # <<< KOREKSI: Tanda tangan eksplisit >>>
    if symbol is None or not isinstance(symbol, str):
        logger.error("periodic_session_data_update_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set(): # <<< PENTING >>>
        try:
            current_mt5_datetime = datetime.now(timezone.utc)
            market_data_processor.calculate_and_update_session_data(symbol, current_mt5_datetime)
            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["periodic_session_data_update_loop"]))
        except Exception as e:
            logger.error(f"Error tak terduga di periodic_session_data_update_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)
    logger.info(f"periodic_session_data_update_loop untuk {symbol} dihentikan.")

def periodic_market_status_update_loop(symbol: str, stop_event: threading.Event): # <<< KOREKSI: Tanda tangan eksplisit >>>
    if symbol is None or not isinstance(symbol, str):
        logger.error("periodic_market_status_update_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set(): # <<< PENTING >>>
        try:
            latest_tick_info = mt5_connector.get_current_tick_info(symbol)
            if latest_tick_info:
                logger.info(f"Status Pasar: Harga terakhir {float(latest_tick_info['last']):.5f} @ {latest_tick_info['time']}")
            else:
                logger.warning(f"data_updater: Tidak ada tick real-time yang tersedia. Menggunakan harga dan waktu default untuk {symbol}.")
            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["periodic_market_status_update_loop"]))
        except Exception as e:
            logger.error(f"Error tak terduga di periodic_market_status_update_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)
    logger.info(f"periodic_market_status_update_loop untuk {symbol} dihentikan.")

def periodic_mt5_trade_data_update_loop(symbol: str, stop_event: threading.Event): # <<< KOREKSI: Tanda tangan eksplisit >>>
    if symbol is None or not isinstance(symbol, str):
        logger.error("periodic_mt5_trade_data_update_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set(): # <<< PENTING >>>
        try:
            database_manager.update_mt5_trade_data_periodically(symbol)
            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["periodic_mt5_trade_data_update_loop"]))
        except Exception as e:
            logger.error(f"Error tak terduga di periodic_mt5_trade_data_update_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)
    logger.info(f"periodic_mt5_trade_data_update_loop untuk {symbol} dihentikan.")

def daily_open_prices_scheduler_loop(symbol: str, stop_event: threading.Event): # <<< KOREKSI: Tanda tangan eksplisit >>>
    if symbol is None or not isinstance(symbol, str):
        logger.error("daily_open_prices_scheduler_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set(): # <<< PENTING >>>
        try:
            market_data_processor.update_daily_open_prices_logic(symbol)
            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["daily_open_prices_scheduler_loop"]))
        except Exception as e:
            logger.error(f"Error tak terduga di daily_open_prices_scheduler_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)
    logger.info(f"daily_open_prices_scheduler_loop untuk {symbol} dihentikan.")

def periodic_historical_data_update_loop(symbol: str, stop_event: threading.Event): # <<< KOREKSI: Tanda tangan eksplisit >>>
    if symbol is None or not isinstance(symbol, str):
        logger.error("periodic_historical_data_update_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set(): # <<< PENTING >>>
        try:
            logger.info(f"Memperbarui candle historis TERBARU dan indikator terkait untuk {symbol}...")
            
            market_data_processor._backfill_single_timeframe_features(symbol, "H1", datetime.now(timezone.utc) - timedelta(days=2), datetime.now(timezone.utc))
            market_data_processor._backfill_single_timeframe_features(symbol, "H4", datetime.now(timezone.utc) - timedelta(days=7), datetime.now(timezone.utc))
            
            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["periodic_historical_data_update_loop"]))
        except Exception as e:
            logger.error(f"Error tak terduga di periodic_historical_data_update_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)
    logger.info(f"periodic_historical_data_update_loop untuk {symbol} dihentikan.")

def periodic_volume_profile_update_loop(symbol: str, stop_event: threading.Event): # <<< KOREKSI: Tanda tangan eksplisit >>>
    if symbol is None or not isinstance(symbol, str):
        logger.error("periodic_volume_profile_update_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set(): # <<< PENTING >>>
        try:
            logger.info(f"Memperbarui Volume Profiles untuk {symbol}...")
            
            candles_h4 = database_manager.get_historical_candles_from_db(symbol, "H4", limit=config.MarketData.COLLECT_TIMEFRAMES.get("H4", 5000))
            if candles_h4: 
                df_candles_h4 = pd.DataFrame(candles_h4)
                df_candles_h4['open_time_utc'] = pd.to_datetime(df_candles_h4['open_time_utc'])
                df_candles_h4.set_index('open_time_utc', inplace=True)
                market_data_processor.update_volume_profiles(symbol, "H4", df_candles_h4)

            candles_d1 = database_manager.get_historical_candles_from_db(symbol, "D1", limit=config.MarketData.COLLECT_TIMEFRAMES.get("D1", 5000))
            if candles_d1:
                df_candles_d1 = pd.DataFrame(candles_d1)
                df_candles_d1['open_time_utc'] = pd.to_datetime(df_candles_d1['open_time_utc'])
                df_candles_d1.set_index('open_time_utc', inplace=True)
                market_data_processor.update_volume_profiles(symbol, "D1", df_candles_d1)
            
            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["periodic_volume_profile_update_loop"]))
        except Exception as e:
            logger.error(f"Error tak terduga di periodic_volume_profile_update_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)
    logger.info(f"periodic_volume_profile_update_loop untuk {symbol} dihentikan.")

def periodic_combined_advanced_detection_loop(symbol: str, stop_event: threading.Event): # <<< KOREKSI: Tanda tangan eksplisit >>>
    if symbol is None or not isinstance(symbol, str):
        logger.error("periodic_combined_advanced_detection_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set(): # <<< PENTING >>>
        try:
            logger.info(f"Memulai loop deteksi lanjutan gabungan untuk {symbol}...")
            
            timeframes_to_process = config.MarketData.ENABLED_TIMEFRAMES 
            
            for tf_name, enabled in timeframes_to_process.items():
                if not enabled: continue
                
                # --- PERBAIKAN PENTING DI SINI: Inisialisasi current_atr_value di awal setiap iterasi loop tf_name ---
                current_atr_value = Decimal('0.0') # Ini memastikan variabel selalu didefinisikan
                # --- AKHIR PERBAIKAN ---

                candles = database_manager.get_historical_candles_from_db(symbol, tf_name, limit=config.MarketData.COLLECT_TIMEFRAMES.get(tf_name, 500))
                if not candles: 
                    logger.debug(f"Tidak ada candle untuk {symbol} {tf_name}. Melewatkan detektor lanjutan untuk TF ini.")
                    continue
                
                df_candles = pd.DataFrame(candles)
                df_candles['open_time_utc'] = pd.to_datetime(df_candles['open_time_utc'])
                df_candles.set_index('open_time_utc', inplace=True)

                df_candles_decimal = df_candles.copy()
                df_candles_decimal = df_candles_decimal.rename(columns={
                    'open_price': 'open', 'high_price': 'high', 'low_price': 'low',
                    'close_price': 'close', 'tick_volume': 'volume',
                    'real_volume': 'real_volume', 'spread': 'spread'
                })
                for col in ['open', 'high', 'low', 'close', 'volume', 'real_volume', 'spread']:
                    if col in df_candles_decimal.columns:
                        df_candles_decimal[col] = df_candles_decimal[col].apply(utils.to_decimal_or_none)
                initial_decimal_len = len(df_candles_decimal)
                df_candles_decimal.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
                if len(df_candles_decimal) < initial_decimal_len:
                    logger.warning(f"Periodic Combined Detection: Dihapus {initial_decimal_len - len(df_candles_decimal)} baris dengan NaN di kolom OHLC untuk {symbol} {tf_name} (setelah konversi Decimal).")
                if df_candles_decimal.empty:
                    logger.warning(f"Periodic Combined Detection: DataFrame Decimal kosong setelah pembersihan NaN untuk {symbol} {tf_name}. Melewatkan detektor lanjutan untuk TF ini.")
                    continue

                df_candles_float = df_candles_decimal.copy()
                for col in ['open', 'high', 'low', 'close', 'volume', 'real_volume', 'spread']:
                    if col in df_candles_float.columns:
                        df_candles_float[col] = df_candles_float[col].apply(utils.to_float_or_none)
                initial_float_len = len(df_candles_float)
                df_candles_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
                if len(df_candles_float) < initial_float_len:
                    logger.warning(f"Periodic Combined Detection: Dihapus {initial_float_len - len(df_candles_float)} baris dengan NaN di kolom OHLC untuk {symbol} {tf_name} (setelah konversi float).")
                if df_candles_float.empty:
                    logger.warning(f"Periodic Combined Detection: DataFrame float kosong setelah pembersihan NaN untuk {symbol} {tf_name}. Melewatkan detektor lanjutan untuk TF ini.")
                    continue

                # Perhitungan ATR (sudah ada di kode Anda)
                if not df_candles_float.empty and config.MarketData.ATR_PERIOD > 0:
                    atr_series = market_data_processor._calculate_atr(df_candles_float, config.MarketData.ATR_PERIOD)
                    if not atr_series.empty and pd.notna(atr_series.iloc[-1]):
                        current_atr_value = utils.to_decimal_or_none(atr_series.iloc[-1])
                logger.debug(f"ATR untuk {symbol} {tf_name}: {float(current_atr_value):.5f}")



                # --- Bagian df_candles_float (tetap sama seperti patch sebelumnya) ---
                df_candles_float = df_candles_decimal.copy() # Mulai dari Decimal DataFrame yang sudah di-rename
                # Kemudian konversi kolom-kolom yang sudah di-rename ke float
                for col in ['open', 'high', 'low', 'close', 'volume', 'real_volume', 'spread']:
                    if col in df_candles_float.columns:
                        df_candles_float[col] = df_candles_float[col].apply(utils.to_float_or_none)
                
                # Pastikan tidak ada NaN di kolom harga kunci setelah konversi ke float
                initial_float_len = len(df_candles_float)
                df_candles_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
                if len(df_candles_float) < initial_float_len:
                    logger.warning(f"Periodic Combined Detection: Dihapus {initial_float_len - len(df_candles_float)} baris dengan NaN di kolom OHLC untuk {symbol} {tf_name} (setelah konversi float).")
                
                if df_candles_float.empty:
                    logger.warning(f"Periodic Combined Detection: DataFrame float kosong setelah pembersihan NaN untuk {symbol} {tf_name}. Melewatkan detektor lanjutan untuk TF ini.")
                    continue

                if config.MarketData.ENABLE_SR_DETECTION:
                    market_data_processor._detect_new_sr_levels_historically(symbol, tf_name, df_candles_decimal, current_atr_value)
                    market_data_processor._update_existing_sr_levels_status(symbol, tf_name, df_candles_decimal, current_atr_value)
                    market_data_processor._detect_new_supply_demand_zones_historically(symbol, tf_name, df_candles_decimal, current_atr_value)
                    market_data_processor._update_existing_supply_demand_zones_status(symbol, tf_name, df_candles_decimal, current_atr_value)

                if config.MarketData.ENABLE_OB_FVG_DETECTION:
                    market_data_processor._detect_new_order_blocks_historically(symbol, tf_name, df_candles_decimal, current_atr_value)
                    market_data_processor._update_existing_order_blocks_status(symbol, tf_name, df_candles_decimal, current_atr_value)
                    market_data_processor._detect_new_fair_value_gaps_historically(symbol, tf_name, df_candles_decimal, current_atr_value)
                    market_data_processor._update_existing_fair_value_gaps_status(symbol, tf_name, df_candles_decimal, current_atr_value)
                
                if config.MarketData.ENABLE_LIQUIDITY_DETECTION:
                    swing_results_df = market_data_processor._calculate_swing_highs_lows_internal(df_candles_float, swing_length=config.AIAnalysts.SWING_EXT_BARS)
                    
                    if not swing_results_df.empty and not swing_results_df['HighLow'].isnull().all():
                        market_data_processor._detect_new_liquidity_zones_historically(symbol, tf_name, df_candles_decimal, current_atr_value)
                        market_data_processor._update_existing_liquidity_zones_status(symbol, tf_name, df_candles_decimal, current_atr_value)
                    else:
                        logger.debug(f"Tidak ada swing points untuk {symbol} {tf_name}. Melewatkan deteksi Likuiditas.")

                if config.MarketData.ENABLE_FIBONACCI_DETECTION:
                    swing_results_df_fib = market_data_processor._calculate_swing_highs_lows_internal(df_candles_float, swing_length=config.AIAnalysts.SWING_EXT_BARS)
                    
                    if not swing_results_df_fib.empty and not swing_results_df_fib['HighLow'].isnull().all():
                        retracement_results_df = market_data_processor._calculate_retracements_internal(df_candles_decimal, swing_results_df_fib)
                        if not retracement_results_df.empty and not retracement_results_df['Direction'].isnull().all():
                            market_data_processor._detect_new_fibonacci_levels_historically(symbol, tf_name, df_candles_decimal, current_atr_value)
                            market_data_processor._update_existing_fibonacci_levels_status(symbol, tf_name, df_candles_decimal, current_atr_value)
                        else:
                            logger.debug(f"Tidak ada retracement yang terdeteksi untuk {symbol} {tf_name}. Melewatkan deteksi Fibonacci Levels.")
                    else:
                        logger.debug(f"Tidak ada swing points untuk {symbol} {tf_name}. Melewatkan deteksi Fibonacci Levels.")


                if config.MarketData.ENABLE_MARKET_STRUCTURE_DETECTION:
                    market_data_processor._detect_new_market_structure_events_historically(symbol, tf_name, df_candles_decimal, current_atr_value)
                    phl_htf = config.MarketData.MA_TREND_TIMEFRAMES[-1] if config.MarketData.MA_TREND_TIMEFRAMES else tf_name
                    market_data_processor._detect_new_previous_high_low_historically(symbol, tf_name, phl_htf, df_candles_decimal, current_atr_value) 
                
                if config.MarketData.ENABLE_RSI_CALCULATION:
                    market_data_processor._detect_and_save_rsi_historically(symbol, tf_name, df_candles_decimal)
                    market_data_processor._detect_overbought_oversold_conditions_historically(symbol, tf_name, df_candles_decimal)

                if config.MarketData.ENABLE_MACD_CALCULATION:
                    market_data_processor._detect_and_save_macd_historically(symbol, tf_name, df_candles_decimal)
                
            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["periodic_combined_advanced_detection_loop"]))
        except Exception as e:
            logger.error(f"Error tak terduga di periodic_combined_advanced_detection_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)

def periodic_fundamental_data_update_loop(arg1, arg2):
    """
    Loop periodik untuk memperbarui data fundamental (kalender ekonomi dan artikel berita).
    Mengambil data berdasarkan konfigurasi scheduler dan mengirim notifikasi ke Telegram.
    Data diambil melalui satu panggilan fungsi ke fundamental_data_service.
    """
    logger.info("Memulai periodic_fundamental_data_update_loop...")
    try:
        # Mengambil konfigurasi dari config.py
        # Anda dapat menyesuaikan rentang hari atau menggunakan start_date/end_date eksplisit
        # Konversi interval scheduler (detik) ke hari untuk rentang data historis
        days_past = int(config.Scheduler.UPDATE_INTERVALS.get("periodic_fundamental_data_update_loop", 3600) / (3600 * 24)) + 1 # +1 hari untuk memastikan cakupan penuh
        days_future = 1 # Biasanya ingin melihat event yang akan datang juga
        min_impact_level = "Low" # Atau sesuai konfigurasi Anda di config.py
        
        # Batas jumlah berita yang akan ditampilkan/dinotifikasi
        MAX_NEWS_ARTICLES_TO_DISPLAY = config.Telegram.MAX_ARTICLES_TO_NOTIFY # Menggunakan konfigurasi Telegram yang sudah ada


        # --- Mengambil semua data fundamental (kalender dan berita) dalam satu panggilan ---
        # Ini akan memicu scraper yang relevan di fundamental_data_service
        comprehensive_data = _fundamental_service_instance.get_comprehensive_fundamental_data(
            days_past=days_past,
            days_future=days_future,
            min_impact_level=min_impact_level,
            include_news_topics=None, # Anda bisa menambahkan topik spesifik jika diperlukan
            read_from_cache_only=False # Biasanya ingin live scrape di loop ini
        )

        economic_events = comprehensive_data.get("economic_calendar", [])
        news_articles = comprehensive_data.get("news_article", [])


        # --- Simpan data yang diambil ke database ---
        # Penyimpanan tetap terpisah karena ada model database yang berbeda untuk event dan berita
        if economic_events:
            database_manager.save_economic_events(economic_events)
            logger.info(f"Berhasil menyimpan {len(economic_events)} event kalender ekonomi ke DB.")
        else:
            logger.info("Tidak ada event kalender ekonomi baru yang ditemukan.")

        if news_articles:
            database_manager.save_news_articles(news_articles)
            logger.info(f"Berhasil menyimpan {len(news_articles)} artikel berita ke DB.")
        else:
            logger.info("Tidak ada artikel berita baru yang ditemukan.")

        # --- Kirim notifikasi ke Telegram (opsional dan berdasarkan konfigurasi) ---
        if config.Telegram.SEND_FUNDAMENTAL_NOTIFICATIONS:
            # Notifikasi untuk event ekonomi
            if economic_events:
                notification_service.notify_only_economic_calendar(
                    economic_events_list=economic_events,
                    min_impact=min_impact_level
                )
                logger.info("Notifikasi kalender ekonomi dikirim.")
            else:
                logger.info("Tidak ada event kalender ekonomi untuk dinotifikasi.")

            # Notifikasi untuk artikel berita (dengan batasan jumlah)
            if news_articles:
                # Batasi jumlah berita yang akan dinotifikasi
                news_to_notify = news_articles[:MAX_NEWS_ARTICLES_TO_DISPLAY]
                notification_service.notify_only_news_articles(
                    news_articles_list=news_to_notify,
                    include_topics=None # Sesuaikan jika Anda memfilter topik di sini
                )
                logger.info(f"Notifikasi {len(news_to_notify)} artikel berita dikirim (dari total {len(news_articles)}).")
            else:
                logger.info("Tidak ada artikel berita untuk dinotifikasi.")
        else:
            logger.info("Notifikasi fundamental dinonaktifkan di konfigurasi.")

        logger.info("periodic_fundamental_data_update_loop selesai.")

    except Exception as e:
        logger.error(f"Error tak terduga di periodic_fundamental_data_update_loop: {e}", exc_info=True)
        # Notifikasi error ke Telegram jika ada masalah
        notification_service.notify_error(f"Gagal memperbarui data fundamental: {e}", "Fundamental Data Update Loop")


def rule_based_signal_loop(symbol: str, stop_event: threading.Event): 
    if symbol is None or not isinstance(symbol, str):
        logger.error("rule_based_signal_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set(): 
        try:
            current_time = datetime.now(timezone.utc)
            
            current_price_val, current_price_time = get_latest_realtime_price()
            
            if current_price_val is None or current_price_val == Decimal('0.0'):
                logger.warning(f"RULE SIGNAL: Tidak ada harga tick terbaru yang valid untuk {symbol}. Melewatkan generasi sinyal.")
                time.sleep(float(config.Scheduler.UPDATE_INTERVALS["rule_based_signal_loop"]))
                continue

            signal = rule_based_signal_generator.generate_signal(
                symbol=symbol,
                current_time=current_time, 
                current_price=current_price_val 
            )
            
            if signal['action'] != "HOLD": 
                logger.info(f"RULE SIGNAL: Sinyal FINAL untuk {symbol}: {signal['action']} (Conf: {signal['confidence']}) Entry:{signal['entry_price_suggestion']:.5f}, SL:{signal['stop_loss_suggestion']:.5f}, TP1:{signal['take_profit_suggestion']:.5f}. Reason: {signal['reasoning']}")
            else:
                logger.info(f"RULE SIGNAL: Sinyal terbaru untuk {symbol}: {signal['action']} (Conf: {signal['confidence']})")
            
            logger.info(f"Sinyal berbasis aturan berhasil disimpan ke DB.")

            if config.Telegram.SEND_SIGNAL_NOTIFICATIONS and signal['action'] != "HOLD":
                notification_service.send_telegram_message(
                    f"🚨 *Sinyal Baru untuk {symbol}*\n"
                    f"Aksi: {signal['action']} ({signal['confidence']})\n"
                    f"Entry: `{signal['entry_price_suggestion']:.5f}`\n"
                    f"SL: `{signal['stop_loss_suggestion']:.5f}`\n"
                    f"TP1: `{signal['take_profit_suggestion']:.5f}`\n"
                    f"Alasan: {signal['reasoning']}",
                    "Signal Notification"
                )
            elif config.Telegram.SEND_SIGNAL_NOTIFICATIONS and signal['action'] == "HOLD":
                 logger.info(f"Notifikasi sinyal berbasis aturan HOLD (Conf: {signal['confidence']}) dilewati (action HOLD).")

            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["rule_based_signal_loop"])) 

        except Exception as e:
            logger.error(f"Error tak terduga di rule_based_signal_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)

def daily_summary_report_loop(symbol: str, stop_event: threading.Event): 
    symbol = symbol # Dapatkan symbol dari args[1]
    if symbol is None or not isinstance(symbol, str):
        logger.error("daily_summary_report_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set(): # <<< PENTING >>>
        try:
            logger.info(f"Menghasilkan laporan ringkasan harian untuk {symbol}...")
            # notification_service.send_daily_summary_report(symbol) # Asumsi fungsi ini ada
            pass 
            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["daily_summary_report_loop"]))
        except Exception as e:
            logger.error(f"Error tak terduga di daily_summary_report_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)

def monthly_historical_feature_backfill_loop(symbol: str, stop_event: threading.Event): 
    symbol = symbol # Dapatkan symbol dari args[1]
    if symbol is None or not isinstance(symbol, str):
        logger.error("monthly_historical_feature_backfill_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set(): # <<< PENTING >>>
        try:
            logger.info(f"Memulai backfill fitur historis bulanan untuk {symbol}...")
            market_data_processor.backfill_historical_features(symbol) 
            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["monthly_historical_feature_backfill_loop"]))

        except Exception as e:
            logger.error(f"Error tak terduga di monthly_historical_feature_backfill_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)

def scenario_analysis_loop(symbol: str, stop_event: threading.Event): 
    symbol = symbol # Dapatkan symbol dari args[1]
    if symbol is None or not isinstance(symbol, str):
        logger.error("scenario_analysis_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set(): # <<< PENTING >>>
        try:
            logger.info(f"Memicu analisis skenario untuk {symbol}...")
            
            current_price_val, current_price_time = get_latest_realtime_price()
            if current_price_val is None or current_price_val == Decimal('0.0'):
                logger.warning("Scenario Analyzer: Tidak ada harga real-time atau timestamp yang valid dari data_updater. Tidak dapat melakukan analisis skenario.")
                time.sleep(float(config.Scheduler.UPDATE_INTERVALS["scenario_analysis_loop"])) 
                continue

            import ai_consensus_manager 

            ai_consensus_manager.analyze_and_notify_scenario(
                symbol=symbol,
                timeframe="H1", 
                current_price=current_price_val,
                current_time=current_price_time 
            )
            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["scenario_analysis_loop"]))
        except Exception as e:
            logger.error(f"Error tak terduga di scenario_analysis_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)

def detector_health_monitoring_loop(symbol: str, stop_event: threading.Event): 
    symbol = symbol # Dapatkan symbol dari args[1]
    if symbol is None or not isinstance(symbol, str):
        logger.error("detector_health_monitoring_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set():
        try:
            logger.info(f"Menjalankan pemantauan kesehatan detektor untuk {symbol}...")
            # KOREKSI PADA BARIS INI: Panggil metode dari instance global
            detector_monitoring.detector_monitor.send_health_report() # <<< Ini adalah perbaikan yang benar

            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["detector_health_monitoring_loop"]))
        except Exception as e:
            logger.error(f"Error tak terduga di detector_health_monitoring_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)
    logger.info(f"detector_health_monitoring_loop untuk {symbol} dihentikan.")


# --- calculate_and_update_daily_pnl (Fungsi Pembantu untuk PnL) ---
def calculate_and_update_daily_pnl(symbol: str): # <<< PENTING >>> Harus ada 'symbol: str'
    """
    Menghitung profit/loss harian terealisasi dari deal history MT5
    dan memperbarui variabel global.
    """
    global _daily_realized_pnl, _last_daily_pnl_calculation_time
    try:
        current_utc_time = datetime.now(timezone.utc)
        start_of_today_utc = current_utc_time.replace(hour=0, minute=0, second=0, microsecond=0)

        deals_today = mt5_connector.get_mt5_deals_history_raw(start_of_today_utc, current_utc_time, group=symbol) # <<< PENTING >>> Pastikan group=symbol

        if deals_today:
            logger.debug(f"Ditemukan {len(deals_today)} deal yang ditutup hari ini untuk perhitungan P/L harian untuk {symbol}.")
        else:
            logger.debug(f"Tidak ada deal yang ditutup hari ini untuk perhitungan P/L harian untuk {symbol}.")

        total_profit_today = Decimal('0.0')
        if deals_today:
            for deal in deals_today:
                if deal.get('entry') == mt5_connector.mt5.DEAL_ENTRY_OUT and deal.get('profit') is not None:
                    total_profit_today += utils.to_decimal_or_none(deal['profit']) 

        _daily_realized_pnl = total_profit_today
        _last_daily_pnl_calculation_time = current_utc_time
        logger.info(f"Profit/Loss Harian Terealisasi: {float(_daily_realized_pnl):.2f} (diperbarui pada {_last_daily_pnl_calculation_time.isoformat()}).")

    except Exception as e:
        logger.error(f"Gagal menghitung profit/loss harian: {e}", exc_info=True)
        _daily_realized_pnl = Decimal('0.0') 
        _last_daily_pnl_calculation_time = None

# --- periodic_daily_pnl_check_loop ---
def periodic_daily_pnl_check_loop(symbol: str, stop_event: threading.Event): # <<< PENTING >>> Harus ada symbol dan stop_event
    if symbol is None or not isinstance(symbol, str):
        logger.error("periodic_daily_pnl_check_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set(): # <<< PENTING >>>
        try:
            calculate_and_update_daily_pnl(symbol) # <<< PENTING >>> Panggil dengan 'symbol'
            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["periodic_daily_pnl_check_loop"]))
        except Exception as e:
            logger.error(f"Error tak terduga di periodic_daily_pnl_check_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)

# --- daily_cleanup_scheduler_loop ---
def daily_cleanup_scheduler_loop(symbol: str, stop_event: threading.Event): # <<< PENTING >>> Harus ada symbol dan stop_event
    if symbol is None or not isinstance(symbol, str):
        logger.error("daily_cleanup_scheduler_loop dipanggil tanpa argumen simbol string yang valid. Menghentikan loop ini.")
        return 
    while not stop_event.is_set(): # <<< PENTING >>>
        try:
            logger.info(f"Memulai tugas pembersihan harian untuk {symbol}...")
            database_manager.clean_old_price_ticks(days_to_keep=config.MarketData.HISTORICAL_DATA_RETENTION_DAYS.get("M1", 7))
            database_manager.clean_old_mt5_trade_data(days_to_keep=30) 
            database_manager.clean_old_historical_candles() 

            logger.info(f"Pembersihan harian untuk {symbol} selesai.")
            time.sleep(float(config.Scheduler.UPDATE_INTERVALS["daily_cleanup_scheduler_loop"]))
        except Exception as e:
            logger.error(f"Error tak terduga di daily_cleanup_scheduler_loop: {e}", exc_info=True)
            time.sleep(float(config.System.RETRY_DELAY_SECONDS) * 2)