# notification_service.py

import logging
import requests
import json
import os
import re
import queue
import threading
import asyncio
import telegram
from telegram.error import TelegramError

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from config import config
import utils

logger = logging.getLogger(__name__)

_notification_queue = queue.Queue()
_worker_thread = None
_stop_worker = threading.Event()
_loop_initialized_event = threading.Event()

PIP_UNIT_XAUUSD = Decimal('1.0')

MAX_TELEGRAM_MESSAGE_LENGTH = 4096
MAX_TELEGRAM_LINE_LENGTH = 1000

def send_telegram_message(message: str, disable_notification: bool = False):
    if not message:
        logger.warning("Mencoba mengirim pesan Telegram kosong. Dilewati.")
        return

    escaped_message = utils._escape_markdown(message)

    message_chunks = []
    lines = escaped_message.split('\n')
    current_chunk = ""

    for i, line in enumerate(lines):
        if len(line) > MAX_TELEGRAM_LINE_LENGTH:
            logger.warning(f"Baris ke-{i+1} terlalu panjang ({len(line)} karakter). Memotong baris ini.")
            line = line[:MAX_TELEGRAM_LINE_LENGTH].rsplit(' ', 1)[0] + '...'

        if len(current_chunk) + len(line) + 1 > MAX_TELEGRAM_MESSAGE_LENGTH:
            if current_chunk:
                message_chunks.append(current_chunk.strip())
            current_chunk = line + '\n'
        else:
            current_chunk += line + '\n'
    
    if current_chunk.strip():
        message_chunks.append(current_chunk.strip())

    if not message_chunks:
        message_chunks.append(escaped_message[:MAX_TELEGRAM_MESSAGE_LENGTH])
        logger.warning("Pemecahan pesan menghasilkan bagian kosong, menggunakan pemotongan kasar sebagai fallback.")

    logger.debug(f"Pesan asli dipecah menjadi {len(message_chunks)} bagian.")
    for chunk in message_chunks:
        try:
            _notification_queue.put({
                'type': 'text',
                'content': chunk,
                'disable_notification': disable_notification
            })
            logger.debug(f"Pesan teks (bagian) ditambahkan ke antrean notifikasi. Ukuran: {len(chunk)}.")
        except queue.Full:
            logger.warning("Antrean notifikasi penuh, tidak dapat menambahkan pesan teks baru.")
        except Exception as e:
            logger.error(f"Gagal menambahkan pesan teks ke antrean notifikasi: {e}", exc_info=True)

def send_telegram_photo(photo_path: str, caption: str = "", disable_notification: bool = False):
    if not photo_path or not os.path.exists(photo_path):
        logger.error(f"Mencoba mengirim foto yang tidak ada: {photo_path}. Dilewati.")
        return

    escaped_caption = utils._escape_markdown(caption)

    try:
        logger.debug(f"Menambahkan foto '{photo_path}' dengan caption '{escaped_caption[:50]}...' ke antrean notifikasi.")
        _notification_queue.put({
            'type': 'photo',
            'content': photo_path,
            'caption': escaped_caption,
            'disable_notification': disable_notification
        })
        logger.debug(f"Foto '{photo_path}' berhasil ditambahkan ke antrean notifikasi.")
    except queue.Full:
        logger.warning("Antrean notifikasi penuh, tidak dapat menambahkan foto baru.")
    except Exception as e:
        logger.error(f"Gagal menambahkan foto ke antrean notifikasi: {e}", exc_info=True)

def notify_new_ai_signal(signal_data: dict, min_confidence: str = "Low", is_individual_analyst_signal: bool = False):
    if is_individual_analyst_signal:
        if not config.Telegram.SEND_INDIVIDUAL_ANALYST_SIGNALS:
            logger.info(f"Notifikasi sinyal individual dari {signal_data.get('analyst_id', 'N/A')} dinonaktifkan.")
            return
        confidence_levels = {"Low": 1, "Medium": 2, "High": 3}
        if confidence_levels.get(signal_data.get('ai_confidence', 'Low')) < confidence_levels.get(min_confidence, 0):
            logger.info(f"Sinyal individual dari {signal_data.get('analyst_id', 'N/A')} dilewati karena confidence rendah ({signal_data.get('ai_confidence')}).")
            return
    else:
        if not config.Telegram.SEND_SIGNAL_NOTIFICATIONS:
            logger.info("Notifikasi sinyal konsensus/final Telegram dinonaktifkan.")
            return

    message_lines = [f"📊 *SINYAL TRADING BARU untuk {signal_data.get('symbol', 'N/A')}*"]
    
    action = signal_data.get('recommendation_action')
    direction_emoji = "📈" if action == "BUY" else ("📉" if action == "SELL" else "↔️")
    message_lines.append(f"{direction_emoji} *Aksi: {action}*")
    
    entry_price = signal_data.get('trading_recommendation', {}).get('entry_price_suggestion')
    stop_loss = signal_data.get('trading_recommendation', {}).get('stop_loss_suggestion')
    take_profit = signal_data.get('trading_recommendation', {}).get('take_profit_suggestion')
    
    if entry_price is not None:
        message_lines.append(f"Entri: `${float(entry_price):.2f}`")
    if stop_loss is not None:
        message_lines.append(f"SL: `${float(stop_loss):.2f}`")
    if take_profit is not None:
        message_lines.append(f"TP: `${float(take_profit):.2f}`")

    tp2 = signal_data.get('tp2_suggestion')
    tp3 = signal_data.get('tp3_suggestion')
    if tp2 is not None:
        message_lines.append(f"TP2: `${float(tp2):.2f}`")
    if tp3 is not None:
        message_lines.append(f"TP3: `${float(tp3):.2f}`")
    
    if signal_data.get('reasoning'):
        message_lines.append(f"Alasan: {signal_data['reasoning']}")

    send_telegram_message("\n".join(message_lines))
    logger.info(f"Notifikasi sinyal {action} untuk {signal_data.get('symbol', 'N/A')} dikirim.")

def notify_trade_execution(trade_result: dict):
    if not config.Telegram.SEND_TRADE_NOTIFICATIONS:
        logger.info("Notifikasi eksekusi trade Telegram dinonaktifkan.")
        return
    
    symbol = trade_result.get('symbol', 'N/A')
    order_type_raw = trade_result.get('type')
    order_type = "BUY" if order_type_raw == 0 else ("SELL" if order_type_raw == 1 else "HOLD")
    volume = trade_result.get('volume', 0.0)
    price = trade_result.get('price', 0.0)
    
    status_emoji = "✅" if trade_result.get('retcode') == 10009 else "❌"
    message = (
        f"{status_emoji} *Trade Executed: {order_type} {volume} lots of {symbol}*\n"
        f"  Price: `${float(price):.2f}`\n"
        f"  Status: `{trade_result.get('comment', 'Unknown')}` (`retcode: {trade_result.get('retcode')}`)\n"
        f"  Deal: `{trade_result.get('deal')}`"
    )
    send_telegram_message(message)
    logger.info(f"Notifikasi eksekusi trade {order_type} {symbol} dikirim.")

def notify_position_update(position_data: dict):
    if not config.Telegram.SEND_TRADE_NOTIFICATIONS:
        logger.info("Notifikasi pembaruan posisi Telegram dinonaktifkan.")
        return

    message = (
        f"📝 *Position Update for {position_data['symbol']}*\n"
        f"  Ticket: `{position_data['ticket']}` (`{position_data['type']}` {position_data['volume']} lots)\n"
        f"  Open Price: `${float(position_data['price_open']):.2f}`\n"
        f"  Current Price: `${float(position_data['current_price']):.2f}`\n"
        f"  Profit: `${float(position_data['profit']):.2f}`\n"
        f"  SL: `${float(position_data['sl_price']):.2f}` | TP: `${float(position_data['tp_price']):.2f}`"
    )
    send_telegram_message(message, disable_notification=True)
    logger.info(f"Notifikasi update posisi {position_data['symbol']} dikirim.")

def notify_account_info(account_info_data: dict):
    if not config.Telegram.SEND_ACCOUNT_NOTIFICATIONS:
        logger.info("Notifikasi info akun Telegram dinonaktifkan.")
        return

    message = (
        f"💰 *MT5 Account Info ({account_info_data.get('login')})*\n"
        f"  Balance: `${account_info_data.get('balance'):.2f}`\n"
        f"  Equity: `${account_info_data.get('equity'):.2f}`\n"
        f"  Profit: `${account_info_data.get('profit'):.2f}`\n"
        f"  Free Margin: `${account_info_data.get('free_margin'):.2f}`\n"
        f"  Currency: `{account_info_data.get('currency')}`"
    )
    send_telegram_message(message, disable_notification=True)
    logger.info("Notifikasi info akun MT5 dikirim.")

def notify_daily_summary(symbol: str):
    if not config.Telegram.SEND_DAILY_SUMMARY:
        logger.info("Notifikasi ringkasan harian Telegram dinonaktifkan.")
        return

    try:
        from database_manager import get_ai_analysis_results
        latest_consensus = get_ai_analysis_results(symbol=symbol, analyst_id="Consensus_Executor", limit=1)
        
        summary_data = {
            "date": datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            "net_profit": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_percent": "0.00%",
            "max_drawdown": 0.0
        }

    except Exception as e:
        logger.error(f"Gagal mendapatkan data untuk ringkasan harian: {e}", exc_info=True)
        summary_data = {
            "date": datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            "net_profit": 0.0,
            "total_trades": 0,
            "win_rate_percent": "N/A",
            "max_drawdown": 0.0,
            "message": "Error getting summary data."
        }

    message = (
        f"📈 *Daily Performance Summary ({summary_data.get('date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))})*\n"
        f"  Net Profit: `${summary_data.get('net_profit', 0.0):.2f}`\n"
        f"  Total Trades: `{summary_data.get('total_trades', 0)}`\n"
        f"  Win Rate: `{summary_data.get('win_rate_percent', '0.00%')}`\n"
        f"  Max Drawdown: `${summary_data.get('max_drawdown', 0.0):.2f}`"
    )
    send_telegram_message(message)
    logger.info("Notifikasi ringkasan harian dikirim.")

def notify_error(error_message: str, error_type: str = "Application Error"):
    if not config.Telegram.SEND_ERROR_NOTIFICATIONS:
        logger.info("Notifikasi error Telegram dinonaktifkan.")
        return

    message = f"❗ *ERROR: {error_type}*\n\n`{error_message}`"
    send_telegram_message(message)
    logger.error(f"Notifikasi error '{error_type}' dikirim.")

def notify_app_start():
    if not config.Telegram.SEND_APP_STATUS_NOTIFICATIONS:
        logger.info("Notifikasi status aplikasi Telegram dinonaktifkan.")
        return
    
    part1 = "🚀 "
    part2_raw = "Trading Bot Berhasil Dimulai!"
    part3_raw = " Aplikasi telah berhasil diinisialisasi dan siap beroperasi."

    escaped_part1 = utils._escape_markdown(part1)
    escaped_part3 = utils._escape_markdown(part3_raw)
    escaped_part2_content = utils._escape_markdown(part2_raw)
    formatted_part2 = f"*{escaped_part2_content}*"

    final_message_content = escaped_part1 + formatted_part2 + escaped_part3
    
    send_telegram_message(final_message_content)
    logger.info("Notifikasi aplikasi dimulai dikirim.")

def notify_app_stop():
    if not config.Telegram.SEND_APP_STATUS_NOTIFICATIONS:
        logger.info("Notifikasi status aplikasi Telegram dinonaktifkan.")
        return

    part1 = "🛑 "
    part2_raw = "Trading Bot Dihentikan!"
    part3_raw = " Aplikasi telah dihentikan dengan aman."

    escaped_part1 = utils._escape_markdown(part1)
    escaped_part3 = utils._escape_markdown(part3_raw)
    escaped_part2_content = utils._escape_markdown(part2_raw)
    formatted_part2 = f"*{escaped_part2_content}*"

    final_message_content = escaped_part1 + formatted_part2 + escaped_part3
    
    send_telegram_message(final_message_content)
    logger.info("Notifikasi aplikasi dihentikan dikirim.")

def notify_fundamental_data_summary(
    economic_events_list: list, 
    news_articles_list: list, 
    total_scraped_events: int, 
    total_scraped_articles: int,
    min_impact_filter_level: str = "Medium", 
    news_topics_filter: list = None,
    include_ai_analysis_status: bool = False
):
    if not config.Telegram.SEND_FUNDAMENTAL_NOTIFICATIONS:
        logger.info("Notifikasi fundamental Telegram dinonaktifkan di konfigurasi.")
        return

    message_parts = []
    
    impact_levels = {"Low": 1, "Medium": 2, "High": 3}
    min_impact_value_num = impact_levels.get(min_impact_filter_level, 0)

    filtered_events = [
        event for event in economic_events_list 
        if impact_levels.get(event.get("impact"), 0) >= min_impact_value_num
    ]
    events_to_show = sorted(filtered_events, key=lambda x: x.get('event_time_utc', datetime.min))
    
    message_parts.append("📅 *Event Kalender Ekonomi Terbaru:*")
    if events_to_show:
        display_count_events = 0
        for event in events_to_show:
            if config.Telegram.MAX_EVENTS_TO_NOTIFY != 0 and display_count_events >= config.Telegram.MAX_EVENTS_TO_NOTIFY:
                break
            
            time_utc = event.get('event_time_utc')
            time_str = time_utc.strftime('%Y-%m-%d %H:%M UTC') if isinstance(time_utc, datetime) else 'N/A'
            
            current_event_impact = event.get('impact', 'N/A')
            current_event_currency = event.get('currency', 'N/A')
            current_event_name = event.get('name', 'N/A')

            impact_emoji_char = impact_emoji.get(current_event_impact, '⚪')
            
            event_detail_lines = []
            event_detail_lines.append(f"{impact_emoji_char} *{utils._escape_markdown(current_event_name)}* (`{utils._escape_markdown(current_event_currency)}`)")
            if time_str != 'N/A':
                event_detail_lines.append(f"   Waktu: `{utils._escape_markdown(time_str)}`")
            event_detail_lines.append(f"   Dampak: `{utils._escape_markdown(current_event_impact)}`")
            
            actual_val = event.get('actual_value') if event.get('actual_value') is not None else 'N/A'
            forecast_val = event.get('forecast_value') if event.get('forecast_value') is not None else 'N/A'
            previous_val = event.get('previous_value') if event.get('previous_value') is not None else 'N/A'

            if str(actual_val) != 'N/A' or str(forecast_val) != 'N/A' or str(previous_val) != 'N/A':
                message_parts.append(f"   Actual: `{utils._escape_markdown(str(actual_val))}` | Forecast: `{utils._escape_markdown(str(forecast_val))}` | Previous: `{utils._escape_markdown(str(previous_val))}`")

            message_parts.append("\n".join(event_detail_lines))

            display_count_events += 1
            if (config.Telegram.MAX_EVENTS_TO_NOTIFY == 0 and display_count_events < len(events_to_show)) or \
               (config.Telegram.MAX_EVENTS_TO_NOTIFY != 0 and display_count_events < config.Telegram.MAX_EVENTS_TO_NOTIFY and display_count_events < len(events_to_show)):
                 message_parts.append("")
    else:
        message_parts.append("_Tidak ada event kalender yang memenuhi kriteria._")

    message_parts.append("")

    filtered_articles = []
    if news_topics_filter and len(news_topics_filter) > 0:
        for article in news_articles_list:
            is_relevant = False
            title = article.get('title', '')
            summary = article.get('summary', '')
            for topic in news_topics_filter:
                if topic.lower() in title.lower() or (summary and topic.lower() in summary.lower()):
                    is_relevant = True
                    break
            if is_relevant:
                filtered_articles.append(article)
            else:
                logger.debug(f"Melewatkan artikel '{article.get('title')}' karena tidak relevan dengan topik: {news_topics_filter}.")
    else:
        filtered_articles = news_articles_list[:]

    articles_to_show = sorted(filtered_articles, key=lambda x: x.get('published_time_utc', datetime.min), reverse=True)

    message_parts.append("📰 *Berita Fundamental Terbaru:*")
    if articles_to_show:
        display_count_articles = 0
        for i, article in enumerate(articles_to_show):
            if config.Telegram.MAX_ARTICLES_TO_NOTIFY != 0 and display_count_articles >= config.Telegram.MAX_ARTICLES_TO_NOTIFY:
                break

            title = article.get('title', 'N/A')
            source = article.get('source', 'N/A')
            url = article.get('url', '#')
            published_time_utc = article.get('published_time_utc')
            
            time_str = published_time_utc.strftime('%Y-%m-%d %H:%M UTC') if isinstance(published_time_utc, datetime) else 'N/A'

            article_detail_lines = []
            article_detail_lines.append(f"*{i+1}\\. {utils._escape_markdown(title)}*")
            article_detail_lines.append(f"   Sumber: `{utils._escape_markdown(source)}`")
            article_detail_lines.append(f"   Waktu: `{utils._escape_markdown(time_str)}`")
            article_detail_lines.append(f"   \\[Baca Selengkapnya\\]({utils._escape_markdown(url)})")
            
            message_parts.append("\n".join(article_detail_lines))

            display_count_articles += 1
            if (config.Telegram.MAX_ARTICLES_TO_NOTIFY == 0 and display_count_articles < len(articles_to_show)) or \
               (config.Telegram.MAX_ARTICLES_TO_NOTIFY != 0 and display_count_articles < config.Telegram.MAX_ARTICLES_TO_NOTIFY and display_count_articles < len(articles_to_show)):
                message_parts.append("")
    else:
        message_parts.append("_Tidak ada artikel berita yang memenuhi kriteria._")

    message_parts.append("")
    
    message_parts.append(
        f"✅ *Pengumpulan Data Fundamental Selesai:*\n"
        f"Data terbaru \\({total_scraped_articles} berita, {total_scraped_events} event\\) telah dikumpulkan dan disimpan ke database."
    )

    if include_ai_analysis_status:
        message_parts.append("Analisis AI fundamental berhasil dijalankan.")
    else:
        message_parts.append("Analisis AI fundamental dilewati sesuai permintaan\\.")
    
    send_telegram_message("\n".join(message_parts))
    logger.info(f"Notifikasi fundamental komprehensif dikirim (Total Event: {total_scraped_events}, Berita: {total_scraped_articles}).")

def notify_only_economic_calendar(economic_events_list: list, min_impact: str = "Low"):
    if not config.Telegram.SEND_FUNDAMENTAL_NOTIFICATIONS:
        logger.info("Notifikasi fundamental (kalender) Telegram dinonaktifkan.")
        return
    
    if not economic_events_list:
        logger.info("Tidak ada event kalender untuk dinotifikasi.")
        return

    message_lines = ["📅 *Event Kalender Ekonomi Terbaru (Hanya Kalender):*"]
    
    impact_levels = {"Low": 1, "Medium": 2, "High": 3}
    min_impact_value_num = impact_levels.get(min_impact, 0)

    filtered_events = [
        event for event in economic_events_list 
        if impact_levels.get(event.get("impact"), 0) >= min_impact_value_num
    ]
    events_to_show = sorted(filtered_events, key=lambda x: x.get('event_time_utc', datetime.min))
    
    max_events_to_notify = config.Telegram.MAX_EVENTS_TO_NOTIFY

    # NOTE: impact_emoji diimpor di ai_consensus_manager.py, bukan di sini.
    # Jika Anda ingin menggunakannya di sini, Anda harus mengimpornya atau mendefinisikannya di sini.
    # Untuk tujuan notifikasi ini, saya akan mendefinisikannya di sini.
    local_impact_emoji = {
        "Low": "⚪",
        "Medium": "🟡",
        "High": "🔴"
    }

    display_count = 0
    if events_to_show:
        for i, event in enumerate(events_to_show):
            if max_events_to_notify != 0 and display_count >= max_events_to_notify:
                break

            time_utc = event.get('event_time_utc')
            time_str = time_utc.strftime('%Y-%m-%d %H:%M UTC') if isinstance(time_utc, datetime) else 'N/A'
            
            current_event_impact = event.get('impact', 'N/A')
            current_event_currency = event.get('currency', 'N/A')
            current_event_name = event.get('name', 'N/A')

            impact_emoji_char = local_impact_emoji.get(current_event_impact, '⚪')
            
            event_detail_lines = []
            event_detail_lines.append(f"{impact_emoji_char} *{utils._escape_markdown(current_event_name)}* (`{utils._escape_markdown(current_event_currency)}`)")
            if time_str != 'N/A':
                event_detail_lines.append(f"   Waktu: `{utils._escape_markdown(time_str)}`")
            event_detail_lines.append(f"   Dampak: `{utils._escape_markdown(current_event_impact)}`")
            
            actual_val = event.get('actual_value') if event.get('actual_value') is not None else 'N/A'
            forecast_val = event.get('forecast_value') if event.get('forecast_value') is not None else 'N/A'
            previous_val = event.get('previous_value') if event.get('previous_value') is not None else 'N/A'

            if str(actual_val) != 'N/A' or str(forecast_val) != 'N/A' or str(previous_val) != 'N/A':
                event_detail_lines.append(f"   Actual: `{utils._escape_markdown(str(actual_val))}` | Forecast: `{utils._escape_markdown(str(forecast_val))}` | Previous: `{utils._escape_markdown(str(previous_val))}`")

            message_lines.append("\n".join(event_detail_lines))

            display_count += 1
            if (max_events_to_notify == 0 and display_count < len(events_to_show)) or \
               (max_events_to_notify != 0 and display_count < max_events_to_notify and display_count < len(events_to_show)):
                 message_lines.append("")
    else:
        message_lines.append("_Tidak ada event kalender yang memenuhi kriteria._")

    message_lines.append("")

    message_lines.append(f"✅ *Pengumpulan Kalender Selesai:*")
    message_lines.append(f"Total {len(filtered_events)} event kalender ditemukan dan disimpan ke database.")

    send_telegram_message("\n".join(message_lines))
    logger.info(f"Notifikasi {display_count} event kalender dikirim (dari {len(economic_events_list)} total yang diterima).")


def notify_only_news_articles(news_articles_list: list, include_topics: list = None):
    if not config.Telegram.SEND_FUNDAMENTAL_NOTIFICATIONS:
        logger.info("Notifikasi fundamental (berita) Telegram dinonaktifkan.")
        return
        
    if not news_articles_list:
        logger.info("Tidak ada artikel berita untuk dinotifikasi.")
        return

    message_lines = ["📰 *Berita Fundamental Terbaru (Hanya Berita):*"]
    
    filtered_articles = []
    if include_topics and len(include_topics) > 0:
        for article in news_articles_list:
            is_relevant = False
            title = article.get('title', '')
            summary = article.get('summary', '')
            for topic in include_topics:
                if topic.lower() in title.lower() or (summary and topic.lower() in summary.lower()):
                    is_relevant = True
                    break
            if is_relevant:
                filtered_articles.append(article)
            else:
                logger.debug(f"Melewatkan artikel '{article.get('title')}' karena tidak relevan dengan topik: {include_topics}.")
    else:
        filtered_articles = news_articles_list[:]

    articles_to_show = sorted(filtered_articles, key=lambda x: x.get('published_time_utc', datetime.min), reverse=True)
    
    max_articles_to_notify = config.Telegram.MAX_ARTICLES_TO_NOTIFY

    display_count = 0
    if articles_to_show:
        for i, article in enumerate(articles_to_show):
            if max_articles_to_notify != 0 and display_count >= max_articles_to_notify:
                break

            title = article.get('title', 'N/A')
            source = article.get('source', 'N/A')
            url = article.get('url', '#')
            published_time_utc = article.get('published_time_utc')
            
            time_str = published_time_utc.strftime('%Y-%m-%d %H:%M UTC') if isinstance(published_time_utc, datetime) else 'N/A'

            article_detail_lines = []
            article_detail_lines.append(f"*{i+1}\\. {utils._escape_markdown(title)}*")
            article_detail_lines.append(f"   Sumber: `{utils._escape_markdown(source)}`")
            article_detail_lines.append(f"   Waktu: `{utils._escape_markdown(time_str)}`")
            article_detail_lines.append(f"   \\[Baca Selengkapnya\\]({utils._escape_markdown(url)})")
            
            message_lines.append("\n".join(article_detail_lines))

            display_count += 1
            if (max_articles_to_notify == 0 and display_count < len(articles_to_show)) or \
               (max_articles_to_notify != 0 and display_count < max_articles_to_notify and display_count < len(articles_to_show)):
                message_lines.append("")
    else:
        message_lines.append("_Tidak ada artikel berita yang memenuhi kriteria._")

    message_lines.append("")

    message_lines.append(f"✅ *Pengumpulan Berita Selesai:*")
    message_lines.append(f"Total {len(filtered_articles)} artikel berita ditemukan dan disimpan ke database.")

    send_telegram_message("\n".join(message_lines))
    logger.info(f"Notifikasi {display_count} artikel berita dikirim (dari {len(news_articles_list)} total yang diterima).")


async def _async_telegram_sender_worker():
    custom_request = telegram.request.HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=20.0,
        write_timeout=20.0,
    )
    local_telegram_bot = telegram.Bot(token=config.APIKeys.TELEGRAM_BOT_TOKEN, request=custom_request)

    _loop_initialized_event.set()

    logger.info("Telegram sender worker loop dimulai.")

    while not _stop_worker.is_set() or not _notification_queue.empty():
        notification_item = None
        try:
            notification_item = _notification_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
        except Exception as e:
            logger.error(f"Error saat mencoba mengambil item dari antrean: {e}", exc_info=True)
            await asyncio.sleep(0.1)
            continue

        try:
            notif_type = notification_item['type']
            content = notification_item['content']
            disable_notif = notification_item.get('disable_notification', False)

            if notif_type == 'text':
                print("\n" + "="*50)
                print(">>> DEBUG: PESAN TELEGRAM FINAL YANG DIKIRIM <<<")
                print("="*50)
                print(content)
                print("="*50 + "\n")
                await local_telegram_bot.send_message(
                    chat_id=config.APIKeys.TELEGRAM_CHAT_ID,
                    text=content,
                    parse_mode='MarkdownV2',
                    disable_notification=disable_notif
                )
                logger.info("Notifikasi Telegram teks berhasil terkirim.")
            elif notif_type == 'photo':
                caption = notification_item.get('caption', '')
                
                print("\n" + "="*50)
                print(">>> DEBUG: CAPTION TELEGRAM FINAL YANG DIKIRIM <<<")
                print("="*50)
                print(caption)
                print("="*50 + "\n")

                with open(content, 'rb') as photo_file:
                    await local_telegram_bot.send_photo(
                        chat_id=config.APIKeys.TELEGRAM_CHAT_ID,
                        photo=photo_file,
                        caption=utils._escape_markdown(caption),
                        parse_mode='MarkdownV2',
                        disable_notification=disable_notif
                    )
                logger.info(f"Notifikasi Telegram foto '{content}' berhasil terkirim.")
            _notification_queue.task_done()

        except TelegramError as e:
            logger.error(f"Gagal mengirim notifikasi Telegram (TelegramError): {e}", exc_info=True)
            _notification_queue.task_done()
        except Exception as e:
            logger.error(f"Error tak terduga di Telegram sender worker setelah mengambil item: {e}", exc_info=True)
            _notification_queue.task_done()

    logger.info("Telegram sender worker loop dihentikan.")
    while not _notification_queue.empty():
        try:
            _notification_queue.get_nowait()
            _notification_queue.task_done()
        except queue.Empty:
            pass
    logger.info("Telegram notification queue dibersihkan saat shutdown.")

def start_notification_service():
    """
    Memulai worker thread untuk mengirim notifikasi Telegram secara asinkron.
    """
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _stop_worker.clear()
        _loop_initialized_event.clear()
        
        _worker_thread = threading.Thread(target=lambda: asyncio.run(_async_telegram_sender_worker()), name="TelegramSenderThread", daemon=True)
        _worker_thread.start()
        
        # Tunggu hingga event loop di worker thread terinisialisasi
        _loop_initialized_event.wait(timeout=10) # Beri waktu 10 detik
        if not _loop_initialized_event.is_set():
            logger.error("Gagal memulai Telegram sender worker event loop dalam batas waktu.")
            return False
        
        logger.info("Telegram notification service dimulai.")
        return True
    logger.info("Telegram notification service sudah berjalan.")
    return False

def stop_notification_service():
    """
    Menghentikan worker thread pengiriman notifikasi Telegram.
    """
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        logger.info("Menghentikan Telegram notification service...")
        _stop_worker.set()
        try:
            # Masukkan sinyal 'None' ke antrean agar worker keluar dari get()
            _notification_queue.put(None, timeout=1) 
        except queue.Full:
            logger.warning("Notification queue penuh saat shutdown, tidak dapat mengirim sinyal berhenti.")
        except Exception as e:
            logger.error(f"Error mengirim sinyal berhenti ke notif queue: {e}", exc_info=True)

        _worker_thread.join(timeout=5)
        if _worker_thread.is_alive():
            logger.warning("Telegram notification worker thread tidak berhenti dalam batas waktu.")
        else:
            logger.info("Telegram notification worker thread telah dihentikan.")
        _worker_thread = None
    else:
        logger.info("Telegram notification service tidak berjalan atau sudah berhenti.")

if not config.APIKeys.TELEGRAM_BOT_TOKEN or not config.APIKeys.TELEGRAM_CHAT_ID:
    logger.critical("Token Bot Telegram atau Chat ID tidak dikonfigurasi! Notifikasi Telegram akan dinonaktifkan.")
    config.Telegram.SEND_SIGNAL_NOTIFICATIONS = False
    config.Telegram.SEND_TRADE_NOTIFICATIONS = False
    config.Telegram.SEND_ACCOUNT_NOTIFICATIONS = False
    config.Telegram.SEND_DAILY_SUMMARY = False
    config.Telegram.SEND_ERROR_NOTIFICATIONS = False
    config.Telegram.SEND_APP_STATUS_NOTIFICATIONS = False
    config.Telegram.SEND_FUNDAMENTAL_NOTIFICATIONS = False
    config.Telegram.SEND_INDIVIDUAL_ANALYST_SIGNALS = False 
else:
    logger.info("Konfigurasi Telegram API ditemukan.")