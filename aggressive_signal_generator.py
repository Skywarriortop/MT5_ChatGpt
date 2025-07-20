import logging
from datetime import datetime, timezone
from decimal import Decimal, getcontext
import pandas as pd
import numpy as np
import os
import sys
import 

# Import config dan utility
import config as app_config # Menggunakan app_config sesuai kesepakatan
import database_manager
import utils # Pastikan utils diimpor untuk fungsi konversi Decimal

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Default ke INFO, bisa disesuaikan jika perlu lebih detail


# Set presisi Decimal untuk perhitungan keuangan secara global.
getcontext().prec = 10 

class AggressiveSignalGenerator:
    def __init__(self):
        # Inisialisasi parameter dari config
        self.min_sl_pips = app_config.Trading.MIN_SL_PIPS
        self.min_tp_pips = app_config.Trading.MIN_TP_PIPS
        self.point_value = app_config.TRADING_SYMBOL_POINT_VALUE
        self.pip_unit_in_dollar = app_config.PIP_UNIT_IN_DOLLAR

        # Parameter detektor dan indikator dari config
        self.atr_period = app_config.MarketData.ATR_PERIOD
        self.atr_multiplier_for_tolerance = app_config.MarketData.ATR_MULTIPLIER_FOR_TOLERANCE
        self.enable_sr_detection = app_config.MarketData.ENABLE_SR_DETECTION
        self.sr_lookback_candles = app_config.MarketData.SR_LOOKBACK_CANDLES
        self.sr_zone_atr_multiplier = app_config.MarketData.SR_ZONE_ATR_MULTIPLIER
        self.min_sr_strength = app_config.MarketData.MIN_SR_STRENGTH

        self.enable_ob_fvg_detection = app_config.MarketData.ENABLE_OB_FVG_DETECTION
        self.fvg_min_atr_multiplier = app_config.MarketData.FVG_MIN_ATR_MULTIPLIER
        self.ob_min_volume_multiplier = app_config.MarketData.OB_MIN_VOLUME_MULTIPLIER
        self.ob_fvg_mitigation_lookback_candles = app_config.MarketData.OB_FVG_MITIGATION_LOOKBACK_CANDLES

        self.enable_liquidity_detection = app_config.MarketData.ENABLE_LIQUIDITY_DETECTION
        self.liquidity_candle_range_percent = app_config.MarketData.LIQUIDITY_CANDLE_RANGE_PERCENT

        self.enable_fibonacci_detection = app_config.MarketData.ENABLE_FIBONACCI_DETECTION
        self.fibo_retraction_levels = app_config.MarketData.FIBO_RETRACTION_LEVELS

        self.enable_market_structure_detection = app_config.MarketData.ENABLE_MARKET_STRUCTURE_DETECTION
        self.bos_choch_min_pips_confirmation = app_config.MarketData.BOS_CHOCH_MIN_PIPS_CONFIRMATION

        self.enable_swing_detection = app_config.MarketData.ENABLE_SWING_DETECTION
        self.swing_lookback_candles = app_config.MarketData.SWING_LOOKBACK_CANDLES

        self.enable_divergence_detection = app_config.MarketData.ENABLE_DIVERGENCE_DETECTION
        self.rsi_divergence_periods = app_config.MarketData.RSI_DIVERGENCE_PERIODS
        self.macd_divergence_fast_period = app_config.MarketData.MACD_DIVERGENCE_FAST_PERIOD
        self.macd_divergence_slow_period = app_config.MarketData.MACD_DIVERGENCE_SLOW_PERIOD
        self.macd_divergence_signal_period = app_config.MarketData.MACD_DIVERGENCE_SIGNAL_PERIOD

        self.enable_rsi_calculation = app_config.MarketData.ENABLE_RSI_CALCULATION
        self.rsi_period = app_config.MarketData.RSI_PERIOD
        self.rsi_overbought_level = app_config.MarketData.RSI_OVERBOUGHT_LEVEL
        self.rsi_oversold_level = app_config.MarketData.RSI_OVERSOLD_LEVEL

        self.enable_macd_calculation = app_config.MarketData.ENABLE_MACD_CALCULATION
        self.macd_fast_period = app_config.MarketData.MACD_FAST_PERIOD
        self.macd_slow_period = app_config.MarketData.MACD_SLOW_PERIOD
        self.macd_signal_period = app_config.MarketData.MACD_SIGNAL_PERIOD

        self.enable_ema_cross_detection = app_config.MarketData.ENABLE_EMA_CROSS_DETECTION
        self.ema_fast_period = app_config.MarketData.EMA_FAST_PERIOD
        self.ema_slow_period = app_config.MarketData.EMA_SLOW_PERIOD

        self.enable_ma_trend_detection = app_config.MarketData.ENABLE_MA_TREND_DETECTION
        self.ma_trend_periods = app_config.MarketData.MA_TREND_PERIODS
        self.ma_trend_timeframes = app_config.MarketData.MA_TREND_TIMEFRAMES
        self.ma_trend_timeframe_weights = app_config.MarketData.MA_TREND_TIMEFRAME_WEIGHTS

        self.enable_volume_profile_detection = app_config.MarketData.ENABLE_VOLUME_PROFILE_DETECTION
        self.enable_previous_high_low_detection = app_config.MarketData.ENABLE_PREVIOUS_HIGH_LOW_DETECTION

        self.confluence_proximity_tolerance_pips = app_config.MarketData.CONFLUENCE_PROXIMITY_TOLERANCE_PIPS
        self.confluence_score_per_level = app_config.MarketData.CONFLUENCE_SCORE_PER_LEVEL

        self.ob_consolidation_tolerance_points = app_config.MarketData.OB_CONSOLIDATION_TOLERANCE_POINTS
        self.ob_shoulder_length = app_config.MarketData.OB_SHOULDER_LENGTH

        self.fvg_min_candle_body_percent_for_strength = app_config.MarketData.FVG_MIN_CANDLE_BODY_PERCENT_FOR_STRENGTH
        self.fvg_volume_factor_for_strength = app_config.MarketData.FVG_VOLUME_FACTOR_FOR_STRENGTH

        # Parameter untuk TP parsial (Rasio Risiko-Reward)
        self.tp1_rr_ratio = Decimal('1.0') # TP1 pada 1x risiko
        self.tp2_rr_ratio = Decimal('2.0') # TP2 pada 2x risiko
        self.tp3_rr_ratio = Decimal('3.0') # TP3 pada 3x risiko (ini bisa diatur di config jika diperlukan)

        logger.info("AggressiveSignalGenerator initialized.")

    def _calculate_atr(self, candles_df: pd.DataFrame) -> Decimal:
        if candles_df.empty or len(candles_df) < self.atr_period:
            return Decimal('0.0')
        
        high = candles_df['high_price'].apply(utils.to_float_or_none).values
        low = candles_df['low_price'].apply(utils.to_float_or_none).values
        close = candles_df['close_price'].apply(utils.to_float_or_none).values

        if len(high) < self.atr_period or len(low) < self.atr_period or len(close) < self.atr_period:
            return Decimal('0.0')

        # Convert to numpy arrays for ta-lib
        high_np = np.array(high, dtype=float)
        low_np = np.array(low, dtype=float)
        close_np = np.array(close, dtype=float)

        # Calculate True Range (TR)
        tr = np.maximum(high_np - low_np, np.abs(high_np - np.roll(close_np, 1)), np.abs(low_np - np.roll(close_np, 1)))
        tr[0] = high_np[0] - low_np[0] # First TR has no previous close

        # Calculate ATR as SMA of TR
        atr_values = np.convolve(tr, np.ones(self.atr_period)/self.atr_period, mode='valid')
        
        if len(atr_values) > 0:
            return Decimal(str(atr_values[-1]))
        return Decimal('0.0')


    def generate_signal(self, symbol: str, current_time: datetime, current_price: Decimal) -> dict:
        """
        Menghasilkan sinyal trading (BUY/SELL/HOLD) untuk strategi agresif.
        Sinyal ini juga akan menyertakan saran TP multiple.
        """
        signal_data = {
            "action": "HOLD",
            "entry_price_suggestion": None,
            "stop_loss_suggestion": None,
            "take_profit_levels_suggestion": [], # Diubah menjadi list untuk multiple TPs
            "strength": 0,
            "reason": [],
            "timestamp": current_time,
            "signal_type": "Aggressive"
        }

        # Mengambil data candle yang relevan untuk analisis
        # Gunakan timeframe default yang diatur di config, misalnya M5 atau M15
        timeframe = app_config.Trading.DEFAULT_TIMEFRAME
        num_candles_needed = max(
            self.sr_lookback_candles,
            self.ob_fvg_mitigation_lookback_candles,
            self.swing_lookback_candles,
            self.atr_period + 5 # Tambahan buffer untuk ATR
        ) + 10 # Tambahan buffer agar tidak error jika data kurang

        candles_data = database_manager.get_historical_candles_from_db(
            symbol=symbol,
            timeframe=timeframe,
            end_time_utc=current_time,
            limit=num_candles_needed,
            order_asc=True # Pastikan urutan menaik
        )

        if not candles_data or len(candles_data) < self.atr_period + 5: # Minimal candle untuk ATR
            logger.warning(f"ASG: Tidak cukup data candle untuk {timeframe} hingga {current_time}. Hanya ada {len(candles_data) if candles_data else 0} candle.")
            return signal_data

        candles_df = pd.DataFrame(candles_data)
        candles_df['open_time_utc'] = pd.to_datetime(candles_df['open_time_utc'])
        
        # Konversi kolom harga ke Decimal
        for col in ['open_price', 'high_price', 'low_price', 'close_price']:
            candles_df[col] = candles_df[col].apply(utils.to_decimal_or_none)

        candles_df.set_index('open_time_utc', inplace=True)
        candles_df.sort_index(inplace=True)

        # Pastikan current_price adalah Decimal
        current_price_dec = utils.to_decimal_or_none(current_price)
        if current_price_dec is None:
            logger.error(f"ASG: Harga saat ini ({current_price}) tidak valid. Melewatkan generasi sinyal.")
            return signal_data
        
        # Hitung ATR
        current_atr = self._calculate_atr(candles_df)
        atr_tolerance_points = current_atr * self.atr_multiplier_for_tolerance
        if current_atr <= Decimal('0.0'):
            logger.warning(f"ASG: ATR tidak valid atau nol ({current_atr}). Tidak dapat menghasilkan sinyal agresif.")
            return signal_data

        # --- Contoh Logika Sinyal Agresif (Sangat Sederhana) ---
        # Ini adalah placeholder. Kamu akan menggantinya dengan logika strategimu yang sebenarnya.
        # Misalnya: Reversal dari SR kuat, atau breakout impulsif.

        last_candle = candles_df.iloc[-1]
        prev_candle = candles_df.iloc[-2] if len(candles_df) >= 2 else None

        # Sinyal Buy (contoh: rejection dari support kuat)
        # Placeholder: Jika candle terakhir adalah bullish engulfing setelah penurunan, dan dekat SR.
        is_bullish_rejection = False
        if prev_candle is not None and last_candle['open_price'] < last_candle['close_price'] and \
           prev_candle['close_price'] > last_candle['open_price'] and \
           last_candle['close_price'] > prev_candle['open_price']: # bullish engulfing
            is_bullish_rejection = True
        
        # Sinyal Sell (contoh: rejection dari resistance kuat)
        # Placeholder: Jika candle terakhir adalah bearish engulfing setelah kenaikan, dan dekat SR.
        is_bearish_rejection = False
        if prev_candle is not None and last_candle['open_price'] > last_candle['close_price'] and \
           prev_candle['close_price'] < last_candle['open_price'] and \
           last_candle['close_price'] < prev_candle['open_price']: # bearish engulfing
            is_bearish_rejection = True

        # Tambahkan logika S&R di sini jika ENABLE_SR_DETECTION=True
        # Untuk contoh sederhana ini, kita akan fokus pada harga saja tanpa detektor kompleks.
        # Implementasi detektor S&R, OB, FVG, dll., akan memerlukan pemanggilan modul atau fungsi analisis terpisah.

        # Logika Sinyal (contoh sederhana)
        if is_bullish_rejection and current_price_dec > prev_candle['high_price']: # Breakout setelah rejection
            signal_data["action"] = "BUY"
            signal_data["entry_price_suggestion"] = current_price_dec
            # SL di bawah low candle terakhir atau di bawah support terdekat
            signal_data["stop_loss_suggestion"] = last_candle['low_price'] - (atr_tolerance_points * self.point_value)
            signal_data["reason"].append("Bullish Rejection + Breakout")
            signal_data["strength"] += 1

        elif is_bearish_rejection and current_price_dec < prev_candle['low_price']: # Breakout setelah rejection
            signal_data["action"] = "SELL"
            signal_data["entry_price_suggestion"] = current_price_dec
            # SL di atas high candle terakhir atau di atas resistance terdekat
            signal_data["stop_loss_suggestion"] = last_candle['high_price'] + (atr_tolerance_points * self.point_value)
            signal_data["reason"].append("Bearish Rejection + Breakdown")
            signal_data["strength"] += 1

        # --- LOGIKA PENENTUAN MULTIPLE TP LEVELS ---
        if signal_data["action"] != "HOLD" and signal_data["entry_price_suggestion"] is not None and signal_data["stop_loss_suggestion"] is not None:
            entry_price = signal_data["entry_price_suggestion"]
            sl_price = signal_data["stop_loss_suggestion"]

            # Hitung jarak risiko dalam dolar per lot (atau pips/points lalu konversi)
            risk_points = abs(entry_price - sl_price) / self.point_value
            risk_dollar_per_lot_0_01 = risk_points * self.point_value * (Decimal('1.0') / Decimal('0.01')) # Dolar per 0.01 lot

            if risk_points <= Decimal('0.0') or risk_dollar_per_lot_0_01 <= Decimal('0.0'):
                logger.warning(f"ASG: Risk points atau risk dollar per lot adalah nol/negatif. Tidak dapat menghitung TP levels. Risk Points: {risk_points}")
                signal_data["action"] = "HOLD" # Batalkan sinyal jika SL invalid
                signal_data["reason"].append("Invalid SL for TP Calculation")
                return signal_data

            # Hitung harga TP berdasarkan rasio Risk-Reward yang sudah ditentukan
            tp_levels_calculated = []

            # TP1
            if self.tp1_rr_ratio > 0:
                tp1_price = Decimal('0.0')
                if signal_data["action"] == "BUY":
                    tp1_price = entry_price + (risk_points * self.tp1_rr_ratio * self.point_value)
                elif signal_data["action"] == "SELL":
                    tp1_price = entry_price - (risk_points * self.tp1_rr_ratio * self.point_value)
                tp_levels_calculated.append({"price": tp1_price.quantize(Decimal('0.00001')), "volume_percentage": Decimal('0.50')}) # Contoh: TP1 tutup 50%

            # TP2
            if self.tp2_rr_ratio > 0:
                tp2_price = Decimal('0.0')
                if signal_data["action"] == "BUY":
                    tp2_price = entry_price + (risk_points * self.tp2_rr_ratio * self.point_value)
                elif signal_data["action"] == "SELL":
                    tp2_price = entry_price - (risk_points * self.tp2_rr_ratio * self.point_value)
                tp_levels_calculated.append({"price": tp2_price.quantize(Decimal('0.00001')), "volume_percentage": Decimal('0.30')}) # Contoh: TP2 tutup 30%

            # TP3
            if self.tp3_rr_ratio > 0:
                tp3_price = Decimal('0.0')
                if signal_data["action"] == "BUY":
                    tp3_price = entry_price + (risk_points * self.tp3_rr_ratio * self.point_value)
                elif signal_data["action"] == "SELL":
                    tp3_price = entry_price - (risk_points * self.tp3_rr_ratio * self.point_value)
                tp_levels_calculated.append({"price": tp3_price.quantize(Decimal('0.00001')), "volume_percentage": Decimal('0.20')}) # Contoh: TP3 tutup 20%

            # Filter TP yang tidak valid (misalnya, TP di sisi SL atau TP lebih rendah dari SL untuk buy)
            valid_tp_levels = []
            for tp in tp_levels_calculated:
                is_tp_valid = False
                if signal_data["action"] == "BUY" and tp['price'] > entry_price:
                    is_tp_valid = True
                elif signal_data["action"] == "SELL" and tp['price'] < entry_price:
                    is_tp_valid = True
                
                if is_tp_valid:
                    valid_tp_levels.append(tp)
                else:
                    logger.warning(f"ASG: TP level {tp['price']} tidak valid untuk {signal_data['action']} sinyal.")

            # Pastikan TP levels terurut dari yang terdekat ke terjauh
            if signal_data["action"] == "BUY":
                valid_tp_levels.sort(key=lambda x: x['price'])
            elif signal_data["action"] == "SELL":
                valid_tp_levels.sort(key=lambda x: x['price'], reverse=True)

            signal_data["take_profit_levels_suggestion"] = valid_tp_levels

            # Validasi akhir untuk SL/TP agar sesuai dengan MIN_SL_PIPS / MIN_TP_PIPS dari app_config
            # SL
            sl_distance_pips = abs(entry_price - sl_price) / self.point_value
            if sl_distance_pips < self.min_sl_pips:
                logger.warning(f"ASG: SL suggestion ({sl_distance_pips:.2f} pips) terlalu dekat dengan entry. Sinyal dibatalkan.")
                signal_data["action"] = "HOLD"
                signal_data["reason"].append(f"SL too close ({sl_distance_pips:.2f} pips < {self.min_sl_pips} pips)")
                return signal_data
            
            # TP1 (jika ada)
            if valid_tp_levels and len(valid_tp_levels) > 0:
                first_tp_price = valid_tp_levels[0]['price']
                tp_distance_pips = abs(entry_price - first_tp_price) / self.point_value
                if tp_distance_pips < self.min_tp_pips:
                    logger.warning(f"ASG: TP1 suggestion ({tp_distance_pips:.2f} pips) terlalu dekat dengan entry. Sinyal dibatalkan.")
                    signal_data["action"] = "HOLD"
                    signal_data["reason"].append(f"TP1 too close ({tp_distance_pips:.2f} pips < {self.min_tp_pips} pips)")
                    return signal_data
            else:
                # Jika tidak ada TP levels yang valid sama sekali, batalkan sinyal
                logger.warning(f"ASG: Tidak ada TP levels yang valid dihasilkan. Sinyal dibatalkan.")
                signal_data["action"] = "HOLD"
                signal_data["reason"].append("No valid TP levels generated.")
                return signal_data

        # --- LOGIKA UNTUK MULTIPLE ENTRY (dari sisi generator sinyal) ---
        # Sesuai diskusi, generator sinyal akan tetap menghasilkan sinyal
        # BUY/SELL setiap kali kondisi terpenuhi, tanpa memedulikan posisi yang sudah terbuka.
        # Keputusan untuk membuka posisi baru (multi-entry) atau menutup posisi lama
        # akan ditangani oleh BacktestManager (untuk backtest) atau TradingEngine (untuk live trading).
        # Jadi, tidak ada perubahan kode di sini untuk "multiple entry".
        # Ini berarti jika kondisi sinyal BUY terus terpenuhi di beberapa candle,
        # generator ini akan terus menyarankan BUY.

        logger.info(f"ASG: Sinyal dihasilkan untuk {symbol} pada {current_time}: {signal_data['action']} (Strength: {signal_data['strength']}).")
        if signal_data['action'] != 'HOLD':
            logger.info(f"ASG: Entry: {float(signal_data['entry_price_suggestion']):.5f}, SL: {float(signal_data['stop_loss_suggestion']):.5f}.")
            for i, tp in enumerate(signal_data['take_profit_levels_suggestion']):
                logger.info(f"ASG: TP{i+1}: {float(tp['price']):.5f} ({float(tp['volume_percentage']*100):.0f}% volume).")

        return signal_data

# Inisialisasi generator sinyal (ini penting agar bisa langsung dipakai saat diimpor)
aggressive_signal_generator_instance = AggressiveSignalGenerator()

# Fungsi publik untuk diakses oleh modul lain
def generate_signal(symbol: str, current_time: datetime, current_price: Decimal) -> dict:
    return aggressive_signal_generator_instance.generate_signal(symbol, current_time, current_price)