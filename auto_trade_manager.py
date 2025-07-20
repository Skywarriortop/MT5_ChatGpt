# auto_trade_manager.py
import logging
import mt5_connector
from config import config
import re
from decimal import Decimal, getcontext # Import getcontext untuk presisi Decimal
from datetime import datetime, timezone, timedelta
import decimal
import data_updater
import notification_service
from utils import to_float_or_none, to_iso_format_or_none , to_decimal_or_none, to_utc_datetime_or_none, _get_scalar_from_possibly_ndarray
import utils
logger = logging.getLogger(__name__)

# Set presisi Decimal untuk perhitungan keuangan
getcontext().prec = 10 

# Variabel global untuk melacak status harian
_daily_limit_hit_today = False
_last_daily_limit_reset_date = None 

# --- MODIFIKASI INI: Variabel global untuk melacak status Take Partials per posisi ---
# Format: {ticket_id: {tp_level_index: True/False}}
_position_partial_tp_status = {}
# --- AKHIR MODIFIKASI ---


def _reset_daily_limit_status():
    """Meriset status limit harian jika hari sudah berganti."""
    global _daily_limit_hit_today, _last_daily_limit_reset_date, _position_partial_tp_status
    current_date_utc = datetime.now(timezone.utc).date()

    if _last_daily_limit_reset_date is None or _last_daily_limit_reset_date < current_date_utc:
        _daily_limit_hit_today = False
        _last_daily_limit_reset_date = current_date_utc
        _position_partial_tp_status = {} # Reset status partial TP setiap hari baru
        logger.info(f"Status limit harian dan partial TP direset untuk hari baru: {current_date_utc}.")
        if not config.Trading.auto_trade_enabled:
            config.Trading.auto_trade_enabled = True
            # Menggunakan notify_app_status dari notification_service
            notification_service.notify_app_status("Auto-trade diaktifkan kembali secara otomatis di awal hari baru.")
            logger.info("Auto-trade diaktifkan kembali karena hari baru.")



def close_all_open_positions(symbol_param: str):
    """
    Menutup semua posisi terbuka untuk simbol tertentu.
    """
    logger.info(f"Menerima permintaan untuk menutup semua posisi terbuka untuk {symbol_param}.")
    
    positions = mt5_connector.get_mt5_positions_raw() # Mengambil posisi sebagai list of dicts
    if not positions:
        logger.info(f"Tidak ada posisi terbuka untuk {symbol_param} yang perlu ditutup.")
        return True

    closed_count = 0
    for pos in positions:
        # Pastikan ini adalah posisi yang relevan dengan simbol yang diminta
        if pos.get('symbol') == symbol_param:
            try:
                ticket = pos.get('ticket')
                volume_to_close = pos.get('volume') # Tutup seluruh volume yang tersisa
                
                if volume_to_close is None or volume_to_close <= Decimal('0.0'):
                    logger.warning(f"Posisi {ticket} untuk {symbol_param} memiliki volume nol atau tidak valid. Melewatkan penutupan.")
                    continue

                logger.info(f"Mencoba menutup posisi {pos.get('type')} {volume_to_close} lot untuk {symbol_param} (Ticket: {ticket}).")
                result = mt5_connector.close_position(ticket, symbol_param, volume_to_close)
                
                if result and result.retcode == mt5_connector.mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Posisi {ticket} berhasil ditutup: {result.comment}.")
                    closed_count += 1
                    # Hapus status partial TP untuk posisi ini
                    if ticket in _position_partial_tp_status:
                        del _position_partial_tp_status[ticket]
                else:
                    logger.error(f"Gagal menutup posisi {ticket}: {result.comment if result else 'Unknown error'}.")
            except Exception as e:
                logger.error(f"Error saat mencoba menutup posisi {pos.get('ticket')}: {e}", exc_info=True)
                continue
    
    if closed_count > 0:
        logger.info(f"Berhasil menutup {closed_count} posisi untuk {symbol_param}.")
        return True
    else:
        logger.warning(f"Tidak ada posisi yang berhasil ditutup untuk {symbol_param}.")
        return False




def check_daily_pnl_and_act(symbol_param: str):
    """
    Memeriksa profit/loss harian yang terealisasi terhadap target/limit.
    Jika target/limit tercapai, akan menutup posisi dan menonaktifkan auto-trade.
    """
    _reset_daily_limit_status() # Reset status di awal setiap pemeriksaan harian

    global _daily_limit_hit_today
    if _daily_limit_hit_today:
        logger.info("Limit profit/loss harian sudah tercapai hari ini. Tidak ada tindakan trading baru.")
        return # Jangan lakukan apa-apa jika sudah tercapai

    daily_pnl, _ = data_updater.get_daily_realized_pnl()
    account_info = mt5_connector.get_mt5_account_info_raw() # Mengambil info akun sebagai dictionary

    if account_info is None: # Periksa objek account_info itu sendiri
        logger.warning("Tidak dapat mengambil info akun MT5 untuk memeriksa P/L harian (account_info is None).")
        return

    # Ambil balance dan equity, dan langsung konversi ke Decimal.
    # Penting: pastikan ini tidak None sebelum digunakan.
    current_balance = utils.to_decimal_or_none(account_info.get('balance'))
    current_equity = utils.to_decimal_or_none(account_info.get('equity'))

    if current_balance is None or current_equity is None:
        logger.warning("Saldo atau ekuitas akun MT5 tidak valid. Tidak dapat memeriksa P/L harian.")
        return

    # Perbaikan untuk TypeError: 'NoneType' and 'decimal.Decimal'
    # initial_balance_today = current_balance - daily_pnl
    # Perhitungan initial_balance_today sebaiknya mencerminkan balance di awal hari.
    # Jika daily_pnl sudah termasuk dalam current_balance (misalnya, daily_pnl adalah selisih dari balance di awal hari),
    # maka initial_balance_today adalah current_balance minus daily_pnl.
    # Tetapi, jika daily_pnl adalah PnL yang *terealisasi* hari ini,
    # dan balance di MT5 adalah balance *saat ini* setelah PnL itu,
    # maka balance *awal* hari adalah balance saat ini dikurangi PnL yang sudah terealisasi.
    # Logika yang Anda miliki `balance - daily_pnl` adalah benar untuk mencari balance awal hari.
    initial_balance_today = current_balance - daily_pnl

    # Hindari pembagian dengan nol atau nilai yang sangat kecil
    # Gunakan absolute value jika initial_balance_today bisa negatif karena kerugian besar yang sudah terealisasi
    if initial_balance_today == Decimal('0.0'):
        logger.warning("Saldo awal harian nol, tidak dapat menghitung persentase P/L. Melewatkan.")
        return

    # Hitung persentase P/L dari saldo awal hari ini
    pnl_percent = (daily_pnl / initial_balance_today) * Decimal('100.0')

    # Ambil target/limit dari config
    daily_profit_target = config.Trading.DAILY_PROFIT_TARGET_PERCENT
    daily_loss_limit = config.Trading.DAILY_LOSS_LIMIT_PERCENT

    action_taken = False

    if daily_profit_target is not None and daily_profit_target > Decimal('0.0') and pnl_percent >= daily_profit_target:
        logger.info(f"TARGET PROFIT HARIAN TERCAPAI! P/L Harian: {float(daily_pnl):.2f} ({float(pnl_percent):.2f}%). Target: {float(daily_profit_target):.2f}%.")
        # PASTIKAN INI ADALAH send_telegram_message
        notification_service.send_telegram_message(
            f"TARGET PROFIT HARIAN TERCAPAI! Profit: {float(daily_pnl):.2f} ({float(pnl_percent):.2f}%). Auto-trade dinonaktifkan.",
            disable_notification=False
        )
        action_taken = True
    elif daily_loss_limit is not None and daily_loss_limit > Decimal('0.0') and pnl_percent <= -daily_loss_limit:
        logger.warning(f"BATAS KERUGIAN HARIAN TERCAPAI! P/L Harian: {float(daily_pnl):.2f} ({float(pnl_percent):.2f}%). Limit: {float(daily_loss_limit):.2f}%.")
        # PASTIKAN INI ADALAH send_telegram_message
        notification_service.send_telegram_message(
            f"BATAS KERUGIAN HARIAN TERCAPAI! Loss: {float(daily_pnl):.2f} ({float(pnl_percent):.2f}%). Auto-trade dinonaktifkan.",
            disable_notification=False
        )
        action_taken = True

    if action_taken:
        logger.info("Menutup semua posisi terbuka karena target/limit harian tercapai.")
        close_all_open_positions(symbol_param)
        config.Trading.auto_trade_enabled = False # Menonaktifkan auto-trade untuk sisa hari ini
        _daily_limit_hit_today = True # Set flag global
        logger.info("Auto-trade dinonaktifkan untuk sisa hari ini.")
    else:
        logger.debug(f"P/L Harian saat ini: {float(daily_pnl):.2f} ({float(pnl_percent):.2f}%). Belum mencapai target/limit.")



# --- MODIFIKASI INI: Fungsi baru untuk memeriksa dan mengeksekusi Take Partials ---
def check_and_execute_take_partials(symbol_param: str, current_price: Decimal):
    """
    Memeriksa semua posisi terbuka dan mengeksekusi take partials jika level TP tercapai.
    Args:
        symbol_param (str): Simbol trading.
        current_price (Decimal): Harga bid/ask saat ini dari tick real-time.
    """
    if not config.Trading.auto_trade_enabled:
        logger.debug("Auto-trade dinonaktifkan, melewatkan pemeriksaan take partials.")
        return

    positions = mt5_connector.get_mt5_positions_raw()
    if not positions:
        logger.debug(f"Tidak ada posisi terbuka untuk {symbol_param}. Melewatkan pemeriksaan take partials.")
        return

    partial_tp_levels_cfg = config.Trading.PARTIAL_TP_LEVELS
    if not partial_tp_levels_cfg:
        logger.debug("Tidak ada level take partials yang dikonfigurasi. Melewatkan pemeriksaan.")
        return

    for pos in positions:
        # Hanya proses posisi untuk simbol yang relevan
        if pos.get('symbol') != symbol_param:
            continue

        ticket = pos.get('ticket')
        pos_type = pos.get('type')
        price_open = pos.get('price_open')
        current_volume = pos.get('volume')
        current_sl = pos.get('sl')
        current_tp = pos.get('tp') # TP awal posisi
        
        if price_open is None or current_volume is None or current_volume <= Decimal('0.0'):
            logger.warning(f"Posisi {ticket} memiliki data tidak valid (price_open/volume). Melewatkan take partials.")
            continue

        # Ambil SL_price_dec (ini adalah SL awal yang disarankan saat open trade)
        # SimInvestasi harusnya sudah punya sl_price di pos jika dibuka oleh bot
        sl_price_dec_initial = utils.to_decimal_or_none(pos.get('sl')) 
        
        if sl_price_dec_initial is None or sl_price_dec_initial == Decimal('0.0'):
            logger.warning(f"Posisi {ticket} tidak memiliki SL awal yang valid. Tidak dapat menghitung TP parsial berbasis RR. Melewatkan.")
            continue

        # Inisialisasi status partial TP untuk posisi ini jika belum ada
        if ticket not in _position_partial_tp_status:
            _position_partial_tp_status[ticket] = {idx: False for idx in range(len(partial_tp_levels_cfg))}

        logger.debug(f"Memeriksa take partials untuk posisi {ticket} ({pos_type}) @ {float(price_open):.5f}, Volume: {float(current_volume):.2f}.")

        # Iterasi melalui level TP parsial yang dikonfigurasi
        for idx, tp_level_cfg in enumerate(partial_tp_levels_cfg):
            if _position_partial_tp_status[ticket][idx]:
                logger.debug(f"TP level {idx+1} untuk posisi {ticket} sudah tercapai. Melewatkan.")
                continue # Level ini sudah tercapai sebelumnya

            price_multiplier = utils.to_decimal_or_none(tp_level_cfg.get('price_multiplier')) # Ini adalah RR
            volume_percentage_to_close = utils.to_decimal_or_none(tp_level_cfg.get('volume_percentage_to_close'))
            move_sl_to_breakeven = tp_level_cfg.get('move_sl_to_breakeven_after_partial')
            move_sl_to_price = utils.to_decimal_or_none(tp_level_cfg.get('move_sl_to_price_after_partial'))

            if price_multiplier is None or volume_percentage_to_close is None or volume_percentage_to_close <= Decimal('0.0'):
                logger.warning(f"Konfigurasi TP level {idx+1} untuk take partials tidak valid (price_multiplier/volume_percentage_to_close). Melewatkan.")
                continue
            
            # Hitung jarak SL awal dalam dolar
            initial_sl_distance_dollars = abs(price_open - sl_price_dec_initial)
            
            if initial_sl_distance_dollars == Decimal('0.0'):
                logger.warning(f"Posisi {ticket} tidak memiliki SL yang valid atau jarak SL nol. Tidak dapat menghitung TP parsial berbasis RR. Melewatkan.")
                continue

            # Hitung target harga TP parsial berdasarkan RR
            # Jarak TP = Jarak SL * Rasio RR
            tp_distance_dollars = initial_sl_distance_dollars * price_multiplier

            target_price = Decimal('0.0')
            if pos_type == mt5_connector.mt5.POSITION_TYPE_BUY:
                target_price = price_open + tp_distance_dollars
                if current_price >= target_price:
                    logger.info(f"TP level {idx+1} (RR {float(price_multiplier):.1f}) TERCAPAI untuk BUY posisi {ticket} di harga {float(current_price):.5f}.")
                    # --- TAMBAHKAN BARIS LOG DEBUG INI ---
                    logger.debug(f"DEBUG_PARTIAL_TRIGGER: Memanggil _execute_partial_close_and_sl_move untuk Posisi {ticket}, TP Level {idx+1} (BUY).")
                    # --- AKHIR TAMBAHAN ---
                    _execute_partial_close_and_sl_move(
                        ticket, symbol_param, pos_type, price_open, current_volume, current_sl, current_tp,
                        target_price, volume_percentage_to_close, move_sl_to_breakeven, move_sl_to_price
                    )
                    _position_partial_tp_status[ticket][idx] = True
                    break
            elif pos_type == mt5_connector.mt5.POSITION_TYPE_SELL:
                target_price = price_open - tp_distance_dollars
                if current_price <= target_price:
                    logger.info(f"TP level {idx+1} (RR {float(price_multiplier):.1f}) TERCAPAI untuk SELL posisi {ticket} di harga {float(current_price):.5f}.")
                    # --- TAMBAHKAN BARIS LOG DEBUG INI ---
                    logger.debug(f"DEBUG_PARTIAL_TRIGGER: Memanggil _execute_partial_close_and_sl_move untuk Posisi {ticket}, TP Level {idx+1} (SELL).")
                    # --- AKHIR TAMBAHAN ---
                    _execute_partial_close_and_sl_move(
                        ticket, symbol_param, pos_type, price_open, current_volume, current_sl, current_tp,
                        target_price, volume_percentage_to_close, move_sl_to_breakeven, move_sl_to_price
                    )
                    _position_partial_tp_status[ticket][idx] = True
                    break



def _execute_partial_close_and_sl_move(ticket: int, symbol_param: str, pos_type: int, price_open: Decimal, current_volume: Decimal, current_sl: Decimal, current_tp: Decimal, partial_tp_price: Decimal, close_percentage: Decimal, move_sl_to_breakeven: bool, move_sl_to_price: Decimal = None):
    """
    Mengeksekusi penutupan sebagian posisi dan memindahkan SL.
    """
    volume_to_close = (current_volume * close_percentage).quantize(Decimal('0.01'), rounding=decimal.ROUND_UP) # Tambahkan rounding=decimal.ROUND_UP
    
    if volume_to_close <= Decimal('0.0'):
        logger.warning(f"Volume untuk partial close posisi {ticket} adalah nol atau tidak valid ({float(volume_to_close):.2f}). Melewatkan.")
        return

    logger.info(f"Mencoba partial close {float(volume_to_close):.2f} lot dari posisi {ticket} ({pos_type}) di harga {float(partial_tp_price):.5f}.")
    result = mt5_connector.close_position(ticket, symbol_param, volume_to_close)

    if result and result.retcode == mt5_connector.mt5.TRADE_RETCODE_DONE:
        logger.info(f"Partial close posisi {ticket} berhasil. Volume tersisa: {float(current_volume - volume_to_close):.2f}.")
        # Menggunakan notify_trade_execution atau notify_trade_status jika ada di notification_service
        # Asumsi ada notify_trade_status yang bisa menerima parameter ini
        notification_service.notify_trade_execution({ 
            'symbol': symbol_param,
            'type': pos_type, # Gunakan tipe posisi asli (0=BUY, 1=SELL)
            'volume': volume_to_close,
            'price': partial_tp_price,
            'retcode': result.retcode,
            'comment': f"Partial TP Hit: {float(close_percentage*100):.1f}% closed",
            'deal': result.deal
        })

        # Hitung SL baru jika perlu
        new_sl_price = current_sl # Default tetap SL saat ini

        # Prioritaskan move_sl_to_price jika ada
        if move_sl_to_price is not None:
            new_sl_price = move_sl_to_price
            logger.info(f"Memindahkan SL posisi {ticket} ke harga spesifik ({float(new_sl_price):.5f}).")
        elif move_sl_to_breakeven:
            new_sl_price = price_open # Pindahkan SL ke harga open
            logger.info(f"Memindahkan SL posisi {ticket} ke Breakeven ({float(new_sl_price):.5f}).")
        
        # Hanya modifikasi SL jika ada perubahan dan SL baru valid
        if new_sl_price is not None and new_sl_price != current_sl:
            # Pastikan new_sl_price tidak terlalu dekat dengan harga saat ini (min_sl_pips)
            # Ini adalah pengecekan tambahan untuk menghindari error broker
            min_sl_distance_dec = config.Trading.MIN_SL_PIPS * config.TRADING_SYMBOL_POINT_VALUE
            
            is_sl_too_close = False
            if pos_type == mt5_connector.mt5.POSITION_TYPE_BUY and (new_sl_price > partial_tp_price - min_sl_distance_dec): # Untuk buy, SL tidak boleh terlalu dekat di bawah harga
                is_sl_too_close = True
            elif pos_type == mt5_connector.mt5.POSITION_TYPE_SELL and (new_sl_price < partial_tp_price + min_sl_distance_dec): # Untuk sell, SL tidak boleh terlalu dekat di atas harga
                is_sl_too_close = True
            
            if is_sl_too_close:
                logger.warning(f"SL baru ({float(new_sl_price):.5f}) terlalu dekat dengan harga saat ini ({float(partial_tp_price):.5f}) untuk posisi {ticket}. Tidak memodifikasi SL.")
            else:
                modify_result = mt5_connector.modify_position_sl_tp(ticket, symbol_param, new_sl_price, current_tp) # TP tetap sama
                if modify_result and modify_result.retcode == mt5_connector.mt5.TRADE_RETCODE_DONE:
                    logger.info(f"SL posisi {ticket} berhasil dimodifikasi ke {float(new_sl_price):.5f}.")
                else:
                    logger.error(f"Gagal memodifikasi SL posisi {ticket}: {modify_result.comment if modify_result else 'Unknown error'}.")
        else:
            logger.debug(f"Tidak ada perubahan SL yang diperlukan atau SL baru tidak valid untuk posisi {ticket}.")

    else:
        logger.error(f"Partial close posisi {ticket} GAGAL: {result.comment if result else 'Unknown error'}.")
        # Menggunakan notify_trade_execution atau notify_trade_status jika ada di notification_service
        notification_service.notify_trade_execution({
            'symbol': symbol_param,
            'type': pos_type, # Gunakan tipe posisi asli (0=BUY, 1=SELL)
            'volume': volume_to_close,
            'price': partial_tp_price,
            'retcode': result.retcode,
            'comment': f"Partial TP Failed: {result.comment if result else 'Unknown error'}",
            'deal': result.deal
        })


def execute_ai_trade(symbol, action, volume, entry_price, stop_loss, take_profit, magic_number, slippage):
    logger.info(f"Menerima permintaan trade: {action} {volume} {symbol} @ Entry:{float(entry_price):.5f}, SL:{float(stop_loss):.5f}, TP:{float(take_profit):.5f}")
    
    global _daily_limit_hit_today
    if not config.Trading.auto_trade_enabled:
        logger.warning("Auto-trade dinonaktifkan. Tidak ada trade yang dieksekusi.")
        notification_service.notify_trade_status(
            symbol, action, volume, entry_price, stop_loss, take_profit,
            "Trade Rejected: Auto-trade Disabled", "Rejected"
        )
        class MockDisabledResult:
            retcode = -1 
            comment = "Auto-trade disabled."
            deal = 0
            volume = 0.0
            price = entry_price
        return MockDisabledResult()
    
    if _daily_limit_hit_today:
        logger.warning("Limit profit/loss harian sudah tercapai. Tidak ada trade baru yang dieksekusi.")
        notification_service.notify_trade_status(
            symbol, action, volume, entry_price, stop_loss, take_profit,
            "Trade Rejected: Daily Limit Hit", "Rejected"
        )
        class MockLimitHitResult:
            retcode = -1 
            comment = "Daily limit hit."
            deal = 0
            volume = 0.0
            price = entry_price
        return MockLimitHitResult()

    trade_result = None

    if action == "BUY":
        trade_result = mt5_connector.open_buy_position(
            symbol_param=symbol,
            volume=volume,
            slippage=slippage,
            magic_number=magic_number,
            comment="AI_BUY_CONSENSUS",
            sl_price=stop_loss, 
            tp_price=take_profit 
        )
    elif action == "SELL":
        trade_result = mt5_connector.open_sell_position(
            symbol_param=symbol,
            volume=volume,
            slippage=slippage,
            magic_number=magic_number,
            comment="AI_SELL_CONSENSUS",
            sl_price=stop_loss, 
            tp_price=take_profit 
        )
    elif action == "HOLD":
        logger.info("Rekomendasi HOLD. Tidak ada trade yang dieksekusi.")
        class MockHoldResult:
            retcode = 10009 
            comment = "HOLD recommendation - no trade executed."
            deal = 0
            volume = 0.0
            price = entry_price
        trade_result = MockHoldResult()
    else:
        logger.warning(f"Aksi trade tidak didukung: {action}. Tidak ada trade yang dieksekusi.")
        class MockInvalidAction:
            retcode = -1 
            comment = f"Aksi tidak didukung: {action}"
            deal = 0
            volume = 0.0
            price = entry_price
        trade_result = MockInvalidAction()
    
    if trade_result and trade_result.retcode == mt5_connector.mt5.TRADE_RETCODE_DONE:
        logger.info(f"Trade {action} {volume} {symbol} berhasil dieksekusi. Deal: {trade_result.deal}, Price: {float(trade_result.price):.5f}.")
        notification_service.notify_trade_status(
            symbol, action, volume, trade_result.price, stop_loss, take_profit,
            trade_result.comment, "Executed", trade_result.deal
        )
    else:
        logger.error(f"Trade {action} {volume} {symbol} GAGAL. Retcode: {trade_result.retcode if trade_result else 'N/A'}, Comment: {trade_result.comment if trade_result else 'N/A'}.")
        notification_service.notify_trade_status(
            symbol, action, volume, entry_price, stop_loss, take_profit,
            trade_result.comment if trade_result else 'Unknown Error', "Failed"
        )
    
    return trade_result

