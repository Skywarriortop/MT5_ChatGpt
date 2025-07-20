# aggressive_signal_generator.py

import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal, getcontext 
import pandas as pd
import numpy as np 

import database_manager
import config
import utils
import market_data_processor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
getcontext().prec = 10 

def generate_signal(symbol: str, current_time: datetime, current_price: Decimal) -> dict:
    logger.debug(f"AGGR_SIGNAL: Menganalisis {symbol} pada {current_time.strftime('%Y-%m-%d %H:%M:%S')} (Harga: {float(current_price):.5f}).")

    # Pastikan semua nilai konfigurasi yang akan digunakan adalah Decimal
    mt5_point_value = config.TRADING_SYMBOL_POINT_VALUE
    xauusd_dollar_per_pip = config.PIP_UNIT_IN_DOLLAR
    
    default_sl_pips = config.RuleBasedStrategy.DEFAULT_SL_PIPS
    tp1_pips = config.RuleBasedStrategy.TP1_PIPS
    tp2_pips = config.RuleBasedStrategy.TP2_PIPS
    tp3_pips = config.RuleBasedStrategy.TP3_PIPS
    
    momentum_lookback = 3 

    ema_short_period = config.RuleBasedStrategy.EMA_SHORT_PERIOD
    ema_long_period = config.RuleBasedStrategy.EMA_LONG_PERIOD

    rsi_period = config.AIAnalysts.RSI_PERIOD
    rsi_overbought = config.AIAnalysts.RSI_OVERBOUGHT
    rsi_oversold = config.AIAnalysts.RSI_OVERSOLD

    recent_divergence_window_hours = 4 

    # --- DEBUGGING: Log nilai min_candles_for_indicators ---
    min_candles_for_indicators = max(momentum_lookback, ema_long_period, rsi_period) + 5 
    logger.debug(f"AGGR_SIGNAL DEBUG: min_candles_for_indicators dihitung sebagai: {min_candles_for_indicators}")
    # --- AKHIR DEBUGGING ---

    candles_raw_data = database_manager.get_historical_candles_from_db(
        symbol, "M5", limit=min_candles_for_indicators, end_time_utc=current_time, order_asc=True
    )

    # --- DEBUGGING: Log jumlah candle yang diterima ---
    actual_candles_count = len(candles_raw_data) if candles_raw_data else 0
    logger.debug(f"AGGR_SIGNAL DEBUG: Jumlah candle M5 yang diterima dari DB mock: {actual_candles_count}")
    # --- AKHIR DEBUGGING ---

    if not candles_raw_data or actual_candles_count < min_candles_for_indicators:
        logger.debug(f"AGGR_SIGNAL: TIDAK CUKUP CANDLE M5 ({actual_candles_count}) untuk analisis indikator (butuh {min_candles_for_indicators}). Kembali ke HOLD.")
        return {
            "action": "HOLD", "potential_direction": "Sideways", "entry_price_suggestion": float(current_price),
            "stop_loss_suggestion": None, "take_profit_suggestion": None,
            "reasoning": "Tidak cukup data candle untuk sinyal agresif.", "confidence": "Low",
            "tp2_suggestion": None, "tp3_suggestion": None
        }

    df_m5 = pd.DataFrame(candles_raw_data) 
    
    df_m5['open_time_utc'] = pd.to_datetime(df_m5['open_time_utc'])
    df_m5.set_index('open_time_utc', inplace=True)
    df_m5.sort_index(inplace=True)

    numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'tick_volume', 'spread', 'real_volume']
    for col in numeric_cols:
        if col in df_m5.columns:
            df_m5[col] = pd.to_numeric(df_m5[col], errors='coerce') 
    
    initial_df_len = len(df_m5)
    df_m5.dropna(subset=['open_price', 'high_price', 'low_price', 'close_price'], inplace=True) 
    if len(df_m5) < initial_df_len:
        logger.warning(f"AGGR_SIGNAL: Dihapus {initial_df_len - len(df_m5)} baris dengan NaN di kolom OHLC.")
    
    if df_m5.empty:
        logger.debug("AGGR_SIGNAL: DataFrame kosong setelah pembersihan NaN. Kembali ke HOLD.")
        return {
            "action": "HOLD", "potential_direction": "Sideways", "entry_price_suggestion": float(current_price),
            "stop_loss_suggestion": None, "take_profit_suggestion": None,
            "reasoning": "DataFrame kosong setelah pembersihan data.", "confidence": "Low",
            "tp2_suggestion": None, "tp3_suggestion": None
        }

    df_m5_float = df_m5.copy()
    df_m5_float = df_m5_float.rename(columns={
        'open_price': 'open', 'high_price': 'high', 'low_price': 'low',
        'close_price': 'close', 'tick_volume': 'volume'
    })
    # --- DEBUGGING: Log DataFrame float setelah pembersihan/konversi ---
    logger.debug(f"AGGR_SIGNAL DEBUG: DataFrame float head:\n{df_m5_float.head()}")
    logger.debug(f"AGGR_SIGNAL DEBUG: DataFrame float tail:\n{df_m5_float.tail()}")
    # --- AKHIR DEBUGGING ---


    # --- Kondisi Sinyal Agresif ---
    bullish_conditions_met = []
    bearish_conditions_met = []

    # 1. Momentum Sederhana
    if len(df_m5_float) > momentum_lookback:
        current_close_float = df_m5_float['close'].iloc[-1]
        past_close_float = df_m5_float['close'].iloc[-(momentum_lookback + 1)]
        if current_close_float > past_close_float:
            bullish_conditions_met.append(f"Momentum Bullish (Close saat ini > {momentum_lookback} candle lalu).")
        elif current_close_float < past_close_float:
            bearish_conditions_met.append(f"Momentum Bearish (Close saat ini < {momentum_lookback} candle lalu).")
        logger.debug(f"AGGR_SIGNAL DEBUG: Momentum check: Current Close={float(current_close_float):.5f}, Past Close ({momentum_lookback} ago)={float(past_close_float):.5f}")
    else:
        logger.debug(f"AGGR_SIGNAL: Tidak cukup data untuk analisis momentum ({len(df_m5_float)}).")


    # 2. EMA Crossover
    try:
        ema_short_series = market_data_processor._calculate_ema_internal(df_m5_float['close'], ema_short_period)
        ema_long_series = market_data_processor._calculate_ema_internal(df_m5_float['close'], ema_long_period)

        if not ema_short_series.empty and not ema_long_series.empty and len(ema_short_series) > 1 and len(ema_long_series) > 1:
            if pd.notna(ema_short_series.iloc[-1]) and pd.notna(ema_long_series.iloc[-1]) and \
               pd.notna(ema_short_series.iloc[-2]) and pd.notna(ema_long_series.iloc[-2]):
                
                current_ema_short = ema_short_series.iloc[-1]
                current_ema_long = ema_long_series.iloc[-1]
                prev_ema_short = ema_short_series.iloc[-2]
                prev_ema_long = ema_long_series.iloc[-2]

                logger.debug(f"AGGR_SIGNAL DEBUG: EMA Check: Current Short={float(current_ema_short):.5f}, Long={float(current_ema_long):.5f}. Prev Short={float(prev_ema_short):.5f}, Long={float(prev_ema_long):.5f}")

                if current_ema_short > current_ema_long and prev_ema_short <= prev_ema_long:
                    bullish_conditions_met.append(f"EMA ({ema_short_period}/{ema_long_period}) Bullish Crossover.")
                elif current_ema_short < current_ema_long and prev_ema_short >= prev_ema_long:
                    bearish_conditions_met.append(f"EMA ({ema_short_period}/{ema_long_period}) Bearish Crossover.")
        else:
            logger.debug("AGGR_SIGNAL: EMA series kosong atau tidak cukup data.")
    except Exception as e:
        logger.warning(f"AGGR_SIGNAL: Error calculating EMA: {e}")


    # 3. RSI Overbought/Oversold
    try:
        rsi_series = market_data_processor._calculate_rsi(df_m5_float, rsi_period) 
        if not rsi_series.empty and pd.notna(rsi_series.iloc[-1]):
            current_rsi = rsi_series.iloc[-1]
            logger.debug(f"AGGR_SIGNAL DEBUG: RSI Check: Current RSI={float(current_rsi):.2f}. Oversold={float(rsi_oversold):.2f}, Overbought={float(rsi_overbought):.2f}")
            if current_rsi < rsi_oversold:
                bullish_conditions_met.append(f"RSI ({rsi_period}) Oversold ({float(current_rsi):.2f}).")
            elif current_rsi > rsi_overbought:
                bearish_conditions_met.append(f"RSI ({rsi_period}) Overbought ({float(current_rsi):.2f}).")
        else:
            logger.debug("AGGR_SIGNAL: RSI series kosong atau tidak cukup data.")
    except Exception as e:
        logger.warning(f"AGGR_SIGNAL: Error calculating RSI: {e}")


    # 4. Bullish/Bearish Divergence
    recent_divergences = database_manager.get_divergences(
        symbol=symbol,
        timeframe="M5",
        is_active=True,
        end_time_utc=current_time,
        start_time_utc=current_time - timedelta(hours=recent_divergence_window_hours),
        limit=5
    )
    # --- DEBUGGING: Log divergensi yang ditemukan ---
    logger.debug(f"AGGR_SIGNAL DEBUG: Jumlah divergensi terbaru ditemukan: {len(recent_divergences)}")
    if recent_divergences:
        for div in recent_divergences:
            logger.debug(f"AGGR_SIGNAL DEBUG: Divergence: {div.get('divergence_type')} by {div.get('indicator_type')} at {div.get('price_point_time_utc')}")
    # --- AKHIR DEBUGGING ---

    if recent_divergences:
        for div in recent_divergences:
            # Periksa apakah divergensi masih dalam jendela waktu yang relevan
            if (current_time - div.get('price_point_time_utc', datetime.min)).total_seconds() <= timedelta(hours=recent_divergence_window_hours).total_seconds():
                if "Bullish" in div.get('divergence_type', ''):
                    bullish_conditions_met.append(f"Divergensi Bullish ({div.get('indicator_type', '')}) terbaru.")
                elif "Bearish" in div.get('divergence_type', ''):
                    bearish_conditions_met.append(f"Divergensi Bearish ({div.get('indicator_type', '')}) terbaru.")

    # --- Penentuan Aksi Agresif ---
    action = "HOLD"
    entry_price_sugg = current_price 
    sl_sugg = None
    tp1_sugg = None
    tp2_sugg = None
    tp3_sugg = None
    confidence = "Medium" 

    reasoning_text = ""
    potential_direction = "Sideways"

    logger.debug(f"AGGR_SIGNAL DEBUG: Bullish Conditions Met: {bullish_conditions_met}")
    logger.debug(f"AGGR_SIGNAL DEBUG: Bearish Conditions Met: {bearish_conditions_met}")

    if bullish_conditions_met:
        action = "BUY"
        potential_direction = "Bullish"
        reasoning_text = f"⬆️ Sinyal BUY Agresif: " + "; ".join(bullish_conditions_met)
        confidence = "High" 
    elif bearish_conditions_met:
        action = "SELL"
        potential_direction = "Bearish"
        reasoning_text = f"⬇️ Sinyal SELL Agresif: " + "; ".join(bearish_conditions_met)
        confidence = "High" 
    else:
        action = "HOLD"
        potential_direction = "Sideways"
        reasoning_text = "↔️ Tidak ada sinyal momentum/indikator yang jelas untuk aksi agresif."
        confidence = "Low"

    # Hitung SL/TP (dari config.RuleBasedStrategy)
    entry_price_sugg_dec = utils.to_decimal_or_none(entry_price_sugg)

    if entry_price_sugg_dec is None:
        logger.error("AGGR_SIGNAL: entry_price_sugg tidak valid (None). Tidak dapat menghitung SL/TP.")
        return {
            "action": "HOLD", "potential_direction": "Undefined", "entry_price_suggestion": None,
            "stop_loss_suggestion": None, "take_profit_suggestion": None,
            "reasoning": "Gagal menghitung SL/TP karena harga entry tidak valid.", "confidence": "Low",
            "tp2_suggestion": None, "tp3_suggestion": None
        }

    if action == "BUY":
        sl_sugg = entry_price_sugg_dec - default_sl_pips * xauusd_dollar_per_pip
        tp1_sugg = entry_price_sugg_dec + tp1_pips * xauusd_dollar_per_pip
        tp2_sugg = entry_price_sugg_dec + tp2_pips * xauusd_dollar_per_pip
        tp3_sugg = entry_price_sugg_dec + tp3_pips * xauusd_dollar_per_pip
    elif action == "SELL":
        sl_sugg = entry_price_sugg_dec + default_sl_pips * xauusd_dollar_per_pip
        tp1_sugg = entry_price_sugg_dec - tp1_pips * xauusd_dollar_per_pip
        tp2_sugg = entry_price_sugg_dec - tp2_pips * xauusd_dollar_per_pip
        tp3_sugg = entry_price_sugg_dec - tp3_pips * xauusd_dollar_per_pip
    else:
        sl_sugg = None
        tp1_sugg = None
        tp2_sugg = None
        tp3_sugg = None

    precision = Decimal('0.00001') 
    if sl_sugg is not None: sl_sugg = sl_sugg.quantize(precision)
    if tp1_sugg is not None: tp1_sugg = tp1_sugg.quantize(precision)
    if tp2_sugg is not None: tp2_sugg = tp2_sugg.quantize(precision)
    if tp3_sugg is not None: tp3_sugg = tp3_sugg.quantize(precision)

    logger.debug(f"AGGR_SIGNAL DEBUG: Action: {action}, Entry: {float(entry_price_sugg_dec):.5f}, SL: {float(sl_sugg) if sl_sugg else 'None'}, TP1: {float(tp1_sugg) if tp1_sugg else 'None'}")

    return {
        "action": action,
        "potential_direction": potential_direction,
        "entry_price_suggestion": float(entry_price_sugg_dec),
        "stop_loss_suggestion": float(sl_sugg) if sl_sugg else None,
        "take_profit_suggestion": float(tp1_sugg) if tp1_sugg else None,
        "reasoning": reasoning_text,
        "confidence": confidence,
        "tp2_suggestion": float(tp2_sugg) if tp2_sugg else None,
        "tp3_suggestion": float(tp3_sugg) if tp3_sugg else None,
    }