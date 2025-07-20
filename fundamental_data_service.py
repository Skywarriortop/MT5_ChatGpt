# fundamental_data_service.py

import logging
from datetime import datetime, date, timedelta, timezone
import requests
from bs4 import BeautifulSoup
import re
import time
import json
import os
import threading
import queue

# --- Import scraper yang terpisah ---
import news_calender_fxverify as calendar_scraper
import news_scraper_myfxbook
import news_scraper_yahoo
import news_scraper_google
import news_scalper_cnbc

# Import config (untuk API keys seperti cookie string)
import config
import database_manager
import notification_service

logger = logging.getLogger(__name__)


class FundamentalDataService:
    def __init__(self):
        pass

    def _run_scraper_in_thread(self, scraper_func, result_queue, start_filter_dt: datetime, end_filter_dt: datetime, **kwargs):
        """
        Fungsi pembantu untuk menjalankan scraper di thread terpisah dan menyimpan hasilnya.
        Melakukan filtering data berdasarkan rentang waktu yang diberikan.
        """
        read_from_cache_only = kwargs.pop('read_from_cache_only', False)

        results_raw = []
        item_type_to_assign = None

        if scraper_func.__name__ == calendar_scraper.scrape_fxverify_calendar_selenium.__name__:
            item_type_to_assign = "economic_calendar"
        elif scraper_func.__name__ == news_scraper_myfxbook.scrape_myfxbook_news.__name__:
            item_type_to_assign = "news_article"
        elif scraper_func.__name__ == news_scraper_yahoo.scrape_yahoo_finance_headlines_direct.__name__:
            item_type_to_assign = "news_article"
        elif scraper_func.__name__ == news_scraper_google.scrape_google_news.__name__:
            item_type_to_assign = "news_article"
        elif scraper_func.__name__ == news_scalper_cnbc.scrape_cnbc_news.__name__:
            item_type_to_assign = "news_article"

        try:
            temp_results = []
            if read_from_cache_only:
                logger.info(f"FundaDataService: Membaca dari cache untuk scraper {scraper_func.__name__} (tanpa scraping ulang).")

                cache_file = None
                if scraper_func.__name__ == calendar_scraper.scrape_fxverify_calendar_selenium.__name__:
                    cache_file = calendar_scraper.FXVERIFY_CACHE_FILE
                elif scraper_func.__name__ == news_scraper_myfxbook.scrape_myfxbook_news.__name__:
                    cache_file = news_scraper_myfxbook.MYFXBOOK_CACHE_FILE
                elif scraper_func.__name__ == news_scraper_yahoo.scrape_yahoo_finance_headlines_direct.__name__:
                    cache_file = news_scraper_yahoo.YAHOO_CACHE_FILE
                elif scraper_func.__name__ == news_scraper_google.scrape_google_news.__name__:
                    cache_file = news_scraper_google.GOOGLE_CACHE_FILE
                elif scraper_func.__name__ == news_scalper_cnbc.scrape_cnbc_news.__name__:
                    cache_file = news_scalper_cnbc.CNBC_CACHE_FILE

                if cache_file and os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)

                        data_key = None
                        if item_type_to_assign == "economic_calendar":
                            data_key = "events"
                        elif item_type_to_assign == "news_article":
                            data_key = "news"

                        if data_key and data_key in cache_data:
                            raw_items = cache_data[data_key]
                            for item in raw_items:
                                item_copy = item.copy()
                                item_copy['type'] = item_type_to_assign
                                if item_type_to_assign == "economic_calendar":
                                    # START OF MODIFICATION: Corrected item.copy.get to item_copy.get
                                    if item_copy.get('event_time_utc') and isinstance(item_copy.get('event_time_utc'), str):
                                        try:
                                            item_copy['event_time_utc'] = datetime.fromisoformat(item_copy['event_time_utc']).replace(tzinfo=timezone.utc)
                                        except ValueError:
                                            item_copy['event_time_utc'] = None
                                    if item_copy['event_time_utc'] and start_filter_dt <= item_copy['event_time_utc'] <= end_filter_dt:
                                        temp_results.append(item_copy)
                                    else:
                                        logger.debug(f"FundaDataService: Melewatkan event dari cache '{item_copy.get('name')}' ({item_copy.get('event_time_utc')}) di luar rentang filter ({start_filter_dt} - {end_filter_dt}).")
                                elif item_type_to_assign == "news_article":
                                    if item_copy.get('published_time_utc') and isinstance(item_copy.get('published_time_utc'), str):
                                        try:
                                            item_copy['published_time_utc'] = datetime.fromisoformat(item_copy['published_time_utc']).replace(tzinfo=timezone.utc)
                                        except ValueError:
                                            item_copy['published_time_utc'] = None
                                    if item_copy['published_time_utc'] and start_filter_dt <= item_copy['published_time_utc'] <= end_filter_dt:
                                        temp_results.append(item_copy)
                                    else:
                                        logger.debug(f"FundaDataService: Melewatkan artikel dari cache '{item_copy.get('title')}' ({item_copy.get('published_time_utc')}) di luar rentang filter ({start_filter_dt} - {end_filter_dt}).")
                                    # END OF MODIFICATION
                        else:
                            logger.warning(f"FundaDataService: Cache {cache_file} ditemukan tetapi kunci '{data_key}' tidak ada atau format tidak dikenal.")
                    except json.JSONDecodeError as jde:
                        logger.error(f"FundaDataService: Gagal membaca JSON dari cache {cache_file} untuk {scraper_func.__name__}: {jde}. Mungkin file rusak.")
                else:
                    logger.warning(f"FundaDataService: Cache {cache_file} tidak ditemukan atau kosong untuk {scraper_func.__name__}.")
                
                results_raw = temp_results

            else: # Ini adalah bagian untuk LIVE SCRAPE (normal mode)
                local_kwargs = kwargs.copy()
                if scraper_func.__name__ == calendar_scraper.scrape_fxverify_calendar_selenium.__name__:
                    if 'days_past' in local_kwargs:
                        del local_kwargs['days_past']
                    if 'days_future' in local_kwargs:
                        del local_kwargs['days_future']
                    time.sleep(2) 

                logger.info(f"FundaDataService: Menjalankan scraper {scraper_func.__name__} (normal mode)...")
                results_from_scraper = scraper_func(**local_kwargs)
                logger.info(f"FundaDataService: Scraper {scraper_func.__name__} selesai. Ditemukan {len(results_from_scraper)} item.")

                temp_results = [] # Ini harus didefinisikan di sini
                for item in results_from_scraper: # Item di sini adalah dictionary
                    item_copy = item.copy() # Membuat salinan item agar tidak mengubah item asli dalam list results_from_scraper
                    item_copy['type'] = item_type_to_assign

                    time_obj = None
                    if item_type_to_assign == "economic_calendar":
                        # START OF MODIFICATION: Corrected item.copy.get to item_copy.get
                        if item_copy.get('event_time_utc') and isinstance(item_copy.get('event_time_utc'), str):
                            try:
                                time_obj = datetime.fromisoformat(item_copy['event_time_utc']).replace(tzinfo=timezone.utc)
                            except ValueError:
                                time_obj = None
                        elif isinstance(item_copy.get('event_time_utc'), datetime):
                            time_obj = item_copy.get('event_time_utc')
                    elif item_type_to_assign == "news_article":
                        if item_copy.get('published_time_utc') and isinstance(item_copy.get('published_time_utc'), str):
                            try:
                                time_obj = datetime.fromisoformat(item_copy['published_time_utc']).replace(tzinfo=timezone.utc)
                            except ValueError:
                                time_obj = None
                        elif isinstance(item_copy.get('published_time_utc'), datetime):
                            time_obj = item_copy.get('published_time_utc')
                    # END OF MODIFICATION
                    
                    if time_obj and start_filter_dt <= time_obj <= end_filter_dt:
                        if item_type_to_assign == "economic_calendar":
                            item_copy['event_time_utc'] = time_obj
                        elif item_type_to_assign == "news_article":
                            item_copy['published_time_utc'] = time_obj
                        temp_results.append(item_copy)
                    else:
                        logger.debug(f"FundaDataService: Melewatkan item dari scraper '{item_copy.get('name', item_copy.get('title'))}' ({time_obj}) di luar rentang filter ({start_filter_dt} - {end_filter_dt}).")
                
                results_raw = temp_results
            
                result_queue.put((scraper_func.__name__, results_raw))

        except Exception as e:
            logger.error(f"FundaDataService: Error saat menjalankan scraper {scraper_func.__name__}: {e}", exc_info=True)
            result_queue.put((scraper_func.__name__, []))


    def get_comprehensive_fundamental_data(self, days_past=1, days_future=0, start_date: str = None, end_date: str = None, min_impact_level="Low", target_currency=None, include_news_topics=None, read_from_cache_only: bool = False):
        """
        Mengumpulkan data fundamental (kalender ekonomi dan artikel berita) dari berbagai sumber.
        Jika read_from_cache_only adalah True, hanya akan membaca dari file cache JSON yang ada.
        Dukungan untuk start_date dan end_date untuk filter yang presisi.
        """
        scraper_queue = queue.Queue()
        threads = []

        now_utc = datetime.now(timezone.utc)

        effective_start_filter_dt = None
        effective_end_filter_dt = None

        if start_date and end_date:
            try:
                effective_start_filter_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                effective_end_filter_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).replace(hour=23, minute=59, second=59, microsecond=999999)
                logger.info(f"FundaDataService: Filter tanggal eksplisit: {effective_start_filter_dt.isoformat()} - {effective_end_filter_dt.isoformat()}")
            except ValueError as e:
                logger.error(f"FundaDataService: Format tanggal start_date/end_date tidak valid: {e}. Menggunakan days_past/days_future sebagai fallback.")
                effective_start_filter_dt = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_past)
                effective_end_filter_dt = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999) + timedelta(days=days_future)
        else:
            effective_start_filter_dt = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_past)
            effective_end_filter_dt = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999) + timedelta(days=days_future)
            logger.info(f"FundaDataService: Filter tanggal berdasarkan days_past/days_future: {effective_start_filter_dt.isoformat()} - {effective_end_filter_dt.isoformat()}")


        myfxbook_cookies = config.APIKeys.MYFXBOOK_COOKIES_STRING

        threads.append(threading.Thread(target=self._run_scraper_in_thread,
                                        args=(calendar_scraper.scrape_fxverify_calendar_selenium, scraper_queue, effective_start_filter_dt, effective_end_filter_dt),
                                        kwargs={'read_from_cache_only': read_from_cache_only, 'days_past': days_past, 'days_future': days_future}))

        threads.append(threading.Thread(target=self._run_scraper_in_thread,
                                        args=(news_scraper_myfxbook.scrape_myfxbook_news, scraper_queue, effective_start_filter_dt, effective_end_filter_dt),
                                        kwargs={'cookie_string': myfxbook_cookies, 'days_past': days_past, 'read_from_cache_only': read_from_cache_only}))

        threads.append(threading.Thread(target=self._run_scraper_in_thread,
                                        args=(news_scraper_yahoo.scrape_yahoo_finance_headlines_direct, scraper_queue, effective_start_filter_dt, effective_end_filter_dt),
                                        kwargs={'days_past': days_past, 'read_from_cache_only': read_from_cache_only}))

        threads.append(threading.Thread(target=self._run_scraper_in_thread,
                                        args=(news_scraper_google.scrape_google_news, scraper_queue, effective_start_filter_dt, effective_end_filter_dt),
                                        kwargs={'query_topics': include_news_topics, 'days_past': days_past, 'read_from_cache_only': read_from_cache_only}))

        threads.append(threading.Thread(target=self._run_scraper_in_thread,
                                        args=(news_scalper_cnbc.scrape_cnbc_news, scraper_queue, effective_start_filter_dt, effective_end_filter_dt),
                                        kwargs={'query_topics': include_news_topics, 'days_past': days_past, 'read_from_cache_only': read_from_cache_only}))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        all_raw_results = []
        while not scraper_queue.empty():
            scraper_name, scraper_results = scraper_queue.get()
            all_raw_results.extend(scraper_results)
        
        verified_economic_events = [item for item in all_raw_results if item.get("type") == "economic_calendar"]
        verified_news_articles = [item for item in all_raw_results if item.get("type") == "news_article"]
        
        temp_verified_economic_events = []
        for item in verified_economic_events:
            event_id_val = item.get("event_id")
            event_name_val = item.get("name")
            event_time_obj_final = item.get("event_time_utc")
            missing_fields = []
            if not event_id_val or (isinstance(event_id_val, str) and not event_id_val.strip()):
                missing_fields.append("event_id")
            if not event_name_val or (isinstance(event_name_val, str) and not event_name_val.strip()):
                missing_fields.append("name")
            if event_time_obj_final is None: 
                missing_fields.append("event_time_utc (parsed)")
            if missing_fields:
                logger.warning(f"FundaDataService: Melewatkan event kalender tidak lengkap (bidang hilang: {', '.join(missing_fields)}). Item Asli: {item}")
                continue
            if not item.get("currency"):
                item["currency"] = "N/A"
            if not item.get("impact"):
                item["impact"] = "Low"
            temp_verified_economic_events.append(item)
        verified_economic_events = temp_verified_economic_events

        temp_verified_news_articles = []
        for item in verified_news_articles:
            if 'name' in item and 'title' not in item:
                item['title'] = item.pop('name')
            elif 'name' in item and 'title' in item:
                del item['name']
            
            published_time_obj = item.get("published_time_utc")
            if not item.get("id") or not item.get("title") or not item.get("url") or published_time_obj is None:
                logger.warning(f"FundaDataService: Melewatkan artikel berita tidak lengkap: {item}")
                continue
            if not item.get('summary'):
                item['summary'] = item['title']
            temp_verified_news_articles.append(item)
        verified_news_articles = temp_verified_news_articles

        return {
            "economic_calendar": verified_economic_events,
            "news_article": verified_news_articles
        }

    def get_economic_calendar_only(self, days_past=1, days_future=0, start_date: str = None, end_date: str = None, min_impact_level="Low", target_currency=None, read_from_cache_only: bool = False):
        """
        Mengumpulkan hanya data kalender ekonomi dari sumber yang relevan.
        """
        scraper_queue = queue.Queue()
        threads = []

        now_utc = datetime.now(timezone.utc)

        effective_start_filter_dt = None
        effective_end_filter_dt = None

        if start_date and end_date:
            try:
                effective_start_filter_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                effective_end_filter_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).replace(hour=23, minute=59, second=59, microsecond=999999)
                logger.info(f"FundaDataService: Filter tanggal eksplisit untuk kalender: {effective_start_filter_dt.isoformat()} - {effective_end_filter_dt.isoformat()}")
            except ValueError as e:
                logger.error(f"FundaDataService: Format tanggal start_date/end_date tidak valid untuk kalender: {e}. Menggunakan days_past/days_future sebagai fallback.")
                effective_start_filter_dt = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_past)
                effective_end_filter_dt = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999) + timedelta(days=days_future)
        else:
            effective_start_filter_dt = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_past)
            effective_end_filter_dt = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999) + timedelta(days=days_future)
            logger.info(f"FundaDataService: Filter tanggal berdasarkan days_past/days_future untuk kalender: {effective_start_filter_dt.isoformat()} - {effective_end_filter_dt.isoformat()}")


        # Hanya panggil scraper kalender
        threads.append(threading.Thread(target=self._run_scraper_in_thread,
                                        args=(calendar_scraper.scrape_fxverify_calendar_selenium, scraper_queue, effective_start_filter_dt, effective_end_filter_dt),
                                        kwargs={'read_from_cache_only': read_from_cache_only, 'days_past': days_past, 'days_future': days_future})) # days_past/future tetap diteruskan ke scraper, tapi filtering utama di _run_scraper_in_thread

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_raw_results = []
        while not scraper_queue.empty():
            scraper_name, scraper_results = scraper_queue.get()
            all_raw_results.extend(scraper_results)

        verified_economic_events = []
        for item in all_raw_results:
            if item.get("type") == "economic_calendar":
                published_time_obj = item.get("event_time_utc") 
                if isinstance(published_time_obj, str):
                    try: published_time_obj = datetime.fromisoformat(published_time_obj).replace(tzinfo=timezone.utc)
                    except ValueError: published_time_obj = None
                
                if not (published_time_obj and effective_start_filter_dt <= published_time_obj <= effective_end_filter_dt):
                    continue

                item['event_time_utc'] = published_time_obj # Ensure the datetime object is assigned back
                if not item.get("currency"): item["currency"] = "N/A"
                if not item.get("impact"): item["impact"] = "Low"
                verified_economic_events.append(item)
            else:
                logger.warning(f"FundaDataService: Item tak terduga ditemukan di get_economic_calendar_only: {item.get('type')}. Dilewati.")

        return verified_economic_events


    def get_news_articles_only(self, days_past=1, days_future=0, start_date: str = None, end_date: str = None, include_news_topics: list = None, read_from_cache_only: bool = False):
        """
        Mengumpulkan hanya data artikel berita dari sumber yang relevan.
        """
        scraper_queue = queue.Queue()
        threads = []

        now_utc = datetime.now(timezone.utc)

        effective_start_filter_dt = None
        effective_end_filter_dt = None

        if start_date and end_date:
            try:
                effective_start_filter_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                effective_end_filter_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).replace(hour=23, minute=59, second=59, microsecond=999999)
                logger.info(f"FundaDataService: Filter tanggal eksplisit untuk berita: {effective_start_filter_dt.isoformat()} - {effective_end_filter_dt.isoformat()}")
            except ValueError as e:
                logger.error(f"FundaDataService: Format tanggal start_date/end_date tidak valid untuk berita: {e}. Menggunakan days_past/days_future sebagai fallback.")
                effective_start_filter_dt = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_past)
                effective_end_filter_dt = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999) + timedelta(days=days_future)
        else:
            effective_start_filter_dt = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_past)
            effective_end_filter_dt = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999) + timedelta(days=days_future)
            logger.info(f"FundaDataService: Filter tanggal berdasarkan days_past/days_future untuk berita: {effective_start_filter_dt.isoformat()} - {effective_end_filter_dt.isoformat()}")

        myfxbook_cookies = config.APIKeys.MYFXBOOK_COOKIES_STRING

        threads.append(threading.Thread(target=self._run_scraper_in_thread,
                                        args=(news_scraper_myfxbook.scrape_myfxbook_news, scraper_queue, effective_start_filter_dt, effective_end_filter_dt),
                                        kwargs={'cookie_string': myfxbook_cookies, 'days_past': days_past, 'read_from_cache_only': read_from_cache_only}))

        threads.append(threading.Thread(target=self._run_scraper_in_thread,
                                        args=(news_scraper_yahoo.scrape_yahoo_finance_headlines_direct, scraper_queue, effective_start_filter_dt, effective_end_filter_dt),
                                        kwargs={'days_past': days_past, 'read_from_cache_only': read_from_cache_only}))

        threads.append(threading.Thread(target=self._run_scraper_in_thread,
                                        args=(news_scraper_google.scrape_google_news, scraper_queue, effective_start_filter_dt, effective_end_filter_dt),
                                        kwargs={'query_topics': include_news_topics, 'days_past': days_past, 'read_from_cache_only': read_from_cache_only}))

        threads.append(threading.Thread(target=self._run_scraper_in_thread,
                                        args=(news_scalper_cnbc.scrape_cnbc_news, scraper_queue, effective_start_filter_dt, effective_end_filter_dt),
                                        kwargs={'query_topics': include_news_topics, 'days_past': days_past, 'read_from_cache_only': read_from_cache_only}))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_raw_results = []
        while not scraper_queue.empty():
            scraper_name, scraper_results = scraper_queue.get()
            all_raw_results.extend(scraper_results)

        verified_news_articles = []
        for item in all_raw_results:
            if item.get("type") == "news_article":
                published_time_obj = item.get("published_time_utc")
                if isinstance(published_time_obj, str):
                    try: published_time_obj = datetime.fromisoformat(published_time_obj).replace(tzinfo=timezone.utc)
                    except ValueError: published_time_obj = None

                if not (published_time_obj and effective_start_filter_dt <= published_time_obj <= effective_end_filter_dt):
                    continue

                item['published_time_utc'] = published_time_obj # Ensure the datetime object is assigned back
                if 'name' in item and 'title' not in item: item['title'] = item.pop('name')
                elif 'name' in item and 'title' in item: del item['name']
                if not item.get('summary'): item['summary'] = item['title']
                verified_news_articles.append(item)
            else:
                logger.warning(f"FundaDataService: Item tak terduga ditemukan di get_news_articles_only: {item.get('type')}. Dilewati.")
        
        return verified_news_articles