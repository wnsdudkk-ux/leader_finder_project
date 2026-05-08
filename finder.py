"""Market Regime Analyzer
========================
S&P 500 + Sector ETF 통합 분석 — RS Stack + Breadth + 52W + Gemini AI

Usage
-----
    pip install -r requirements.txt
    set GEMINI_API_KEY=your_api_key
    python finder.py
"""

import argparse
import datetime
import json
import os
import re
import sqlite3
import ssl
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _make_session():
    try:
        from curl_cffi.requests import Session as CfSession
        return CfSession(verify=False, impersonate="chrome131")
    except ImportError:
        import requests as _rq
        s = _rq.Session()
        s.verify = False
        return s

_YF_SESSION = _make_session()

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template_string, redirect, jsonify

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from google import genai as new_genai
    from google.genai import types as new_genai_types
    HAS_NEW_GENAI = True
except ImportError:
    HAS_NEW_GENAI = False


def _log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode(), flush=True)


# ═══════════════════════════════════════════════════════════════════
# API Key Loader — local file 우선, 없는 키는 palantir_project/api_keys.txt
# 에서 보강. (예: NAVER_CLIENT_ID/SECRET 은 palantir_project 에 있음)
# ═══════════════════════════════════════════════════════════════════
def _parse_keys_file(path):
    out = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip()
    except Exception as e:
        _log(f"[!] api_keys load error ({path}): {e}")
    return out


def _load_api_keys(filename="api_keys.txt"):
    keys = {}
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    sib_palantir = os.path.join(parent, "palantir_project", filename)

    fallback_files = []
    if os.path.isfile(sib_palantir):
        fallback_files.append(sib_palantir)

    primary_files = []
    if getattr(sys, "frozen", False):
        p = os.path.join(os.path.dirname(sys.executable), filename)
        if os.path.isfile(p):
            primary_files.append(p)
    p_local = os.path.join(here, filename)
    if os.path.isfile(p_local):
        primary_files.append(p_local)
    p_cwd = os.path.join(os.getcwd(), filename)
    if os.path.isfile(p_cwd) and p_cwd not in primary_files:
        primary_files.append(p_cwd)

    for f in fallback_files + primary_files:
        for k, v in _parse_keys_file(f).items():
            keys[k] = v
    return keys


_API_KEYS = _load_api_keys()
NAVER_CLIENT_ID = _API_KEYS.get("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = _API_KEYS.get("NAVER_CLIENT_SECRET", "")


# ═══════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════
BENCH = "SPY"  # legacy default (US). 각 시장별 bench는 MARKETS 참조
W_FAST = 5
W_SLOW = 10
W_JUDGE = 20
W_FILTER = 60
W_BETA = 252
K_MOM = 20
LOG_RET = True
BETA_ADJ = True
ANN = 252.0
INITIAL_PERIOD = "3y"
W_52WEEK = 252
MIN_BARS = W_BETA + 2 * K_MOM + 10
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sector_prices.db")
DL_DELAY = 0.35
GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_FLASH_MODEL = "gemini-3-flash-preview"
BATCH_SIZE = 50

NARRATIVE_LOOKBACK_DAYS = 20

# ═══════════════════════════════════════════════════════════════════
# Markets (US / KR)
# ═══════════════════════════════════════════════════════════════════
MARKETS = {
    "US": {
        "code": "US",
        "label": "US",
        "title": "US Market — S&P 500 + Sector ETF",
        "bench": "SPY",
        "bench_name": "SPY (S&P 500)",
        "currency": "$",
        "language": "en",
        "constituents_label": "S&P 500",
        "etf_label": "Sector / Theme ETF",
    },
    "KR": {
        "code": "KR",
        "label": "KR",
        "title": "KR Market — KOSPI + 테마 ETF",
        "bench": "069500.KS",
        "bench_name": "KODEX 200 (069500.KS)",
        "currency": "₩",
        "language": "ko",
        "constituents_label": "KOSPI",
        "etf_label": "테마 / 섹터 ETF",
    },
}

# ═══════════════════════════════════════════════════════════════════
# Sector ETF Definitions
# ═══════════════════════════════════════════════════════════════════
SECTOR_ETFS = {
    "IBUY":  {"name": "Amplify Online Retail ETF",              "category": "Online Retail"},
    "KBE":   {"name": "SPDR S&P Bank ETF",                     "category": "Banking"},
    "QTUM":  {"name": "Defiance Quantum ETF",                   "category": "Quantum Computing"},
    "ONLN":  {"name": "ProShares Online Retail ETF",            "category": "Online Retail"},
    "PEJ":   {"name": "Invesco Leisure & Entertainment ETF",    "category": "Leisure & Entertainment"},
    "HERO":  {"name": "Global X Video Games & Esports ETF",     "category": "Video Games & Esports"},
    "ESPO":  {"name": "VanEck Video Gaming & eSports ETF",      "category": "Video Games & Esports"},
    "VOX":   {"name": "Vanguard Communication Services ETF",    "category": "Communication Services"},
    "VNQ":   {"name": "Vanguard Real Estate ETF",               "category": "Real Estate"},
    "IDRV":  {"name": "iShares Self-Driving EV & Tech ETF",     "category": "EV & Autonomous Driving"},
    "DRIV":  {"name": "Global X Autonomous & EV ETF",           "category": "EV & Autonomous Driving"},
    "LIT":   {"name": "Global X Lithium & Battery Tech ETF",    "category": "Lithium & Battery"},
    "PAVE":  {"name": "Global X U.S. Infrastructure ETF",       "category": "Infrastructure"},
    "JETS":  {"name": "U.S. Global Jets ETF",                   "category": "Airlines"},
    "IYT":   {"name": "iShares U.S. Transportation ETF",        "category": "Transportation"},
    "PPA":   {"name": "Invesco Aerospace & Defense ETF",        "category": "Aerospace & Defense"},
    "ITA":   {"name": "iShares U.S. Aerospace & Defense ETF",   "category": "Aerospace & Defense"},
    "SIL":   {"name": "Global X Silver Miners ETF",             "category": "Silver Mining"},
    "GDX":   {"name": "VanEck Gold Miners ETF",                 "category": "Gold Mining"},
    "GDXJ":  {"name": "VanEck Junior Gold Miners ETF",          "category": "Gold Mining"},
    "MOO":   {"name": "VanEck Agribusiness ETF",                "category": "Agribusiness"},
    "SLX":   {"name": "VanEck Steel ETF",                       "category": "Steel"},
    "XME":   {"name": "SPDR S&P Metals & Mining ETF",           "category": "Metals & Mining"},
    "NLR":   {"name": "VanEck Uranium+Nuclear Energy ETF",      "category": "Nuclear Energy"},
    "URA":   {"name": "Global X Uranium ETF",                   "category": "Uranium"},
    "FAN":   {"name": "First Trust Global Wind Energy ETF",     "category": "Wind Energy"},
    "TAN":   {"name": "Invesco Solar ETF",                      "category": "Solar Energy"},
    "ICLN":  {"name": "iShares Global Clean Energy ETF",        "category": "Clean Energy"},
    "AMLP":  {"name": "Alerian MLP ETF",                        "category": "MLP / Midstream"},
    "OIH":   {"name": "VanEck Oil Services ETF",                "category": "Oil Services"},
    "XOP":   {"name": "SPDR S&P Oil & Gas E&P ETF",             "category": "Oil & Gas E&P"},
    "IBB":   {"name": "iShares Biotechnology ETF",              "category": "Biotechnology"},
    "XBI":   {"name": "SPDR S&P Biotech ETF",                   "category": "Biotechnology"},
    "IPAY":  {"name": "ETFMG Prime Mobile Payments ETF",        "category": "Mobile Payments"},
    "FINX":  {"name": "Global X FinTech ETF",                   "category": "FinTech"},
    "CIBR":  {"name": "First Trust NASDAQ Cybersecurity ETF",   "category": "Cybersecurity"},
    "WCLD":  {"name": "WisdomTree Cloud Computing ETF",         "category": "Cloud Computing"},
    "BOTZ":  {"name": "Global X Robotics & AI ETF",             "category": "Robotics & AI"},
    "SMH":   {"name": "VanEck Semiconductor ETF",               "category": "Semiconductors"},
    "SOXX":  {"name": "iShares Semiconductor ETF",              "category": "Semiconductors"},
}


def get_sector_etfs():
    _log("  Loading sector ETF list (US) ...")
    tickers = list(SECTOR_ETFS.keys())
    info = {tk: {"name": v["name"], "sector": v["category"], "industry": v["category"]}
            for tk, v in SECTOR_ETFS.items()}
    _log(f"    {len(info)} US sector ETFs")
    return tickers, info


# ═══════════════════════════════════════════════════════════════════
# KR Sector / Theme ETFs (yfinance 형식: 6자리 코드 + .KS)
# ═══════════════════════════════════════════════════════════════════
KR_SECTOR_ETFS = {
    "102110.KS": {"name": "TIGER 200",                    "category": "KOSPI 200 (대형주)"},
    "069660.KS": {"name": "KOSEF 200",                    "category": "KOSPI 200 (대형주)"},
    "229200.KS": {"name": "KODEX 코스닥150",              "category": "코스닥 150"},
    "232080.KS": {"name": "TIGER 코스닥150",              "category": "코스닥 150"},
    "091160.KS": {"name": "KODEX 반도체",                 "category": "반도체"},
    "091230.KS": {"name": "TIGER 반도체",                 "category": "반도체"},
    "091170.KS": {"name": "KODEX 은행",                   "category": "은행"},
    "139270.KS": {"name": "TIGER 200 금융",               "category": "금융"},
    "138540.KS": {"name": "KODEX 보험",                   "category": "보험"},
    "139220.KS": {"name": "TIGER 200 건설",               "category": "건설"},
    "117460.KS": {"name": "KODEX 에너지화학",             "category": "에너지/화학"},
    "117680.KS": {"name": "KODEX 철강",                   "category": "철강"},
    "139260.KS": {"name": "TIGER 200 산업재",             "category": "산업재"},
    "098560.KS": {"name": "TIGER 방송통신",               "category": "방송/통신"},
    "227540.KS": {"name": "TIGER 200 헬스케어",           "category": "헬스케어"},
    "143860.KS": {"name": "TIGER 헬스케어",               "category": "헬스케어"},
    "244620.KS": {"name": "KODEX 바이오",                 "category": "바이오"},
    "266390.KS": {"name": "KODEX 200ESG",                 "category": "ESG"},
    "091180.KS": {"name": "KODEX 자동차",                 "category": "자동차"},
    "139250.KS": {"name": "TIGER 200 중공업",             "category": "조선/중공업"},
    "140710.KS": {"name": "KODEX 운송",                   "category": "운송/물류"},
    "266370.KS": {"name": "KODEX 미디어&엔터",            "category": "미디어/엔터"},
    "139290.KS": {"name": "TIGER 200 IT",                 "category": "IT"},
    "157490.KS": {"name": "TIGER 소프트웨어",             "category": "소프트웨어"},
    "228810.KS": {"name": "TIGER 화장품",                 "category": "화장품"},
    "266410.KS": {"name": "KODEX 필수소비재",             "category": "필수소비재"},
    "266420.KS": {"name": "KODEX 경기소비재",             "category": "경기소비재"},
    "228800.KS": {"name": "TIGER 여행레저",               "category": "여행/레저"},
    "305720.KS": {"name": "KODEX 2차전지산업",            "category": "2차전지"},
    "305540.KS": {"name": "TIGER 2차전지테마",            "category": "2차전지"},
    "364980.KS": {"name": "TIGER KRX BBIG K-뉴딜",        "category": "BBIG (배터리·바이오·인터넷·게임)"},
    "364990.KS": {"name": "TIGER KRX 2차전지 K-뉴딜",     "category": "2차전지"},
    "365040.KS": {"name": "TIGER KRX 게임 K-뉴딜",        "category": "게임"},
    "364970.KS": {"name": "TIGER KRX 인터넷 K-뉴딜",      "category": "인터넷"},
    "364960.KS": {"name": "TIGER KRX 바이오 K-뉴딜",      "category": "바이오"},
    "395160.KS": {"name": "TIGER KRX BBIG K-뉴딜레버리지", "category": "BBIG (레버리지)"},
    "449450.KS": {"name": "PLUS K방산",                   "category": "방산"},
    "449180.KS": {"name": "KODEX K-방산",                 "category": "방산"},
    "445290.KS": {"name": "KODEX 미국AI테크TOP10",        "category": "AI/테크"},
    "381180.KS": {"name": "TIGER 미국필라델피아반도체",   "category": "반도체 (해외)"},
    "367380.KS": {"name": "ARIRANG 신흥국MSCI(합성 H)",   "category": "해외 — 신흥국"},
    "411420.KS": {"name": "KODEX 코스닥150레버리지",      "category": "코스닥 (레버리지)"},
    "228790.KS": {"name": "TIGER 화장품",                 "category": "화장품"},
    "291130.KS": {"name": "KODEX 한국대만IT프리미어",     "category": "IT (한국·대만)"},
    "459580.KS": {"name": "KODEX CD금리액티브(합성)",     "category": "단기금리/현금성"},
    "385600.KS": {"name": "KBSTAR 비메모리반도체액티브",  "category": "반도체 (비메모리)"},
}


def get_kr_sector_etfs():
    _log("  Loading sector ETF list (KR) ...")
    tickers = list(KR_SECTOR_ETFS.keys())
    info = {tk: {"name": v["name"], "sector": v["category"], "industry": v["category"]}
            for tk, v in KR_SECTOR_ETFS.items()}
    _log(f"    {len(info)} KR sector ETFs")
    return tickers, info


# ═══════════════════════════════════════════════════════════════════
# KOSPI 전 종목 (보통주 ~807개, Naver Finance 시가총액 + WICS 업종 분류)
# yfinance 형식: 6자리.KS
# sector 는 broad 분류 (IT/금융/헬스케어/소재/산업재/경기소비재/필수소비재/에너지/유틸리티/통신·미디어/부동산/기타)
# industry 는 Naver/WICS 의 세부 업종
# ═══════════════════════════════════════════════════════════════════
KOSPI_STOCKS = {
    "000020.KS": {"name": "동화약품", "sector": "헬스케어", "industry": "제약"},
    "000040.KS": {"name": "KR모터스", "sector": "경기소비재", "industry": "자동차"},
    "000050.KS": {"name": "경방", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "000070.KS": {"name": "삼양홀딩스", "sector": "소재", "industry": "화학"},
    "000080.KS": {"name": "하이트진로", "sector": "필수소비재", "industry": "음료"},
    "000100.KS": {"name": "유한양행", "sector": "헬스케어", "industry": "제약"},
    "000120.KS": {"name": "CJ대한통운", "sector": "산업재", "industry": "항공화물운송과물류"},
    "000140.KS": {"name": "하이트진로홀딩스", "sector": "필수소비재", "industry": "음료"},
    "000150.KS": {"name": "두산", "sector": "산업재", "industry": "복합기업"},
    "000180.KS": {"name": "성창기업지주", "sector": "소재", "industry": "종이와목재"},
    "000210.KS": {"name": "DL", "sector": "소재", "industry": "화학"},
    "000220.KS": {"name": "유유제약", "sector": "헬스케어", "industry": "제약"},
    "000230.KS": {"name": "일동홀딩스", "sector": "헬스케어", "industry": "제약"},
    "000240.KS": {"name": "한국앤컴퍼니", "sector": "경기소비재", "industry": "자동차부품"},
    "000270.KS": {"name": "기아", "sector": "경기소비재", "industry": "자동차"},
    "000300.KS": {"name": "DH오토넥스", "sector": "경기소비재", "industry": "자동차부품"},
    "000320.KS": {"name": "노루홀딩스", "sector": "소재", "industry": "건축자재"},
    "000370.KS": {"name": "한화손해보험", "sector": "금융", "industry": "손해보험"},
    "000390.KS": {"name": "SP삼화", "sector": "소재", "industry": "건축자재"},
    "000400.KS": {"name": "롯데손해보험", "sector": "금융", "industry": "손해보험"},
    "000430.KS": {"name": "대원강업", "sector": "경기소비재", "industry": "자동차부품"},
    "000480.KS": {"name": "CR홀딩스", "sector": "소재", "industry": "비철금속"},
    "000490.KS": {"name": "대동", "sector": "산업재", "industry": "기계"},
    "000500.KS": {"name": "가온전선", "sector": "소재", "industry": "전기장비"},
    "000520.KS": {"name": "삼일제약", "sector": "헬스케어", "industry": "제약"},
    "000540.KS": {"name": "흥국화재", "sector": "금융", "industry": "손해보험"},
    "000590.KS": {"name": "CS홀딩스", "sector": "소재", "industry": "철강"},
    "000640.KS": {"name": "동아쏘시오홀딩스", "sector": "헬스케어", "industry": "제약"},
    "000650.KS": {"name": "천일고속", "sector": "산업재", "industry": "도로와철도운송"},
    "000660.KS": {"name": "SK하이닉스", "sector": "IT", "industry": "반도체와반도체장비"},
    "000670.KS": {"name": "영풍", "sector": "통신·미디어", "industry": "핸드셋"},
    "000680.KS": {"name": "LS네트웍스", "sector": "금융", "industry": "증권"},
    "000700.KS": {"name": "유수홀딩스", "sector": "산업재", "industry": "항공화물운송과물류"},
    "000720.KS": {"name": "현대건설", "sector": "산업재", "industry": "건설"},
    "000760.KS": {"name": "이화산업", "sector": "소재", "industry": "화학"},
    "000810.KS": {"name": "삼성화재", "sector": "금융", "industry": "손해보험"},
    "000850.KS": {"name": "화천기공", "sector": "산업재", "industry": "기계"},
    "000860.KS": {"name": "강남제비스코", "sector": "소재", "industry": "건축자재"},
    "000880.KS": {"name": "한화", "sector": "산업재", "industry": "복합기업"},
    "000890.KS": {"name": "보해양조", "sector": "필수소비재", "industry": "음료"},
    "000910.KS": {"name": "유니온", "sector": "소재", "industry": "건축자재"},
    "000950.KS": {"name": "전방", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "000970.KS": {"name": "한국주철관", "sector": "소재", "industry": "철강"},
    "000990.KS": {"name": "DB하이텍", "sector": "IT", "industry": "반도체와반도체장비"},
    "001020.KS": {"name": "페이퍼코리아", "sector": "소재", "industry": "종이와목재"},
    "001040.KS": {"name": "CJ", "sector": "산업재", "industry": "복합기업"},
    "001060.KS": {"name": "JW중외제약", "sector": "헬스케어", "industry": "제약"},
    "001070.KS": {"name": "대한방직", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "001080.KS": {"name": "만호제강", "sector": "소재", "industry": "철강"},
    "001120.KS": {"name": "LX인터내셔널", "sector": "산업재", "industry": "무역회사와판매업체"},
    "001130.KS": {"name": "대한제분", "sector": "필수소비재", "industry": "식품"},
    "001200.KS": {"name": "유진투자증권", "sector": "금융", "industry": "증권"},
    "001210.KS": {"name": "금호전기", "sector": "IT", "industry": "디스플레이장비및부품"},
    "001230.KS": {"name": "동국홀딩스", "sector": "소재", "industry": "철강"},
    "001250.KS": {"name": "GS글로벌", "sector": "산업재", "industry": "무역회사와판매업체"},
    "001260.KS": {"name": "남광토건", "sector": "산업재", "industry": "건설"},
    "001270.KS": {"name": "부국증권", "sector": "금융", "industry": "증권"},
    "001290.KS": {"name": "상상인증권", "sector": "금융", "industry": "증권"},
    "001340.KS": {"name": "PKC", "sector": "소재", "industry": "화학"},
    "001360.KS": {"name": "삼성제약", "sector": "헬스케어", "industry": "제약"},
    "001380.KS": {"name": "SG글로벌", "sector": "경기소비재", "industry": "자동차부품"},
    "001390.KS": {"name": "KG케미칼", "sector": "경기소비재", "industry": "자동차"},
    "001420.KS": {"name": "태원물산", "sector": "경기소비재", "industry": "자동차부품"},
    "001430.KS": {"name": "세아베스틸지주", "sector": "소재", "industry": "철강"},
    "001440.KS": {"name": "대한전선", "sector": "소재", "industry": "전기장비"},
    "001450.KS": {"name": "현대해상", "sector": "금융", "industry": "손해보험"},
    "001460.KS": {"name": "BYC", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "001470.KS": {"name": "삼부토건", "sector": "산업재", "industry": "건설"},
    "001500.KS": {"name": "현대차증권", "sector": "금융", "industry": "증권"},
    "001510.KS": {"name": "SK증권", "sector": "금융", "industry": "증권"},
    "001520.KS": {"name": "동양", "sector": "소재", "industry": "건축자재"},
    "001530.KS": {"name": "DI동일", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "001550.KS": {"name": "조비", "sector": "소재", "industry": "화학"},
    "001560.KS": {"name": "제일연마", "sector": "소재", "industry": "비철금속"},
    "001570.KS": {"name": "금양", "sector": "소재", "industry": "화학"},
    "001620.KS": {"name": "케이비아이동국실업", "sector": "경기소비재", "industry": "자동차부품"},
    "001630.KS": {"name": "종근당홀딩스", "sector": "헬스케어", "industry": "제약"},
    "001680.KS": {"name": "대상", "sector": "필수소비재", "industry": "식품"},
    "001720.KS": {"name": "신영증권", "sector": "금융", "industry": "증권"},
    "001740.KS": {"name": "SK네트웍스", "sector": "산업재", "industry": "복합기업"},
    "001750.KS": {"name": "한양증권", "sector": "금융", "industry": "증권"},
    "001770.KS": {"name": "SHD", "sector": "소재", "industry": "철강"},
    "001780.KS": {"name": "알루코", "sector": "소재", "industry": "비철금속"},
    "001790.KS": {"name": "대한제당", "sector": "필수소비재", "industry": "식품"},
    "001800.KS": {"name": "오리온홀딩스", "sector": "필수소비재", "industry": "식품"},
    "001820.KS": {"name": "삼화콘덴서", "sector": "IT", "industry": "전기제품"},
    "001940.KS": {"name": "KISCO홀딩스", "sector": "소재", "industry": "철강"},
    "002020.KS": {"name": "코오롱", "sector": "산업재", "industry": "복합기업"},
    "002030.KS": {"name": "아세아", "sector": "소재", "industry": "건축자재"},
    "002070.KS": {"name": "비비안", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "002100.KS": {"name": "경농", "sector": "소재", "industry": "화학"},
    "002140.KS": {"name": "고려산업", "sector": "필수소비재", "industry": "식품"},
    "002150.KS": {"name": "도화엔지니어링", "sector": "산업재", "industry": "건설"},
    "002170.KS": {"name": "SYTS", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "002200.KS": {"name": "한국수출포장", "sector": "소재", "industry": "포장재"},
    "002210.KS": {"name": "동성제약", "sector": "헬스케어", "industry": "제약"},
    "002220.KS": {"name": "한일철강", "sector": "소재", "industry": "철강"},
    "002240.KS": {"name": "고려제강", "sector": "소재", "industry": "철강"},
    "002310.KS": {"name": "아세아제지", "sector": "소재", "industry": "포장재"},
    "002320.KS": {"name": "한진", "sector": "산업재", "industry": "항공화물운송과물류"},
    "002350.KS": {"name": "넥센타이어", "sector": "경기소비재", "industry": "자동차부품"},
    "002360.KS": {"name": "SH에너지화학", "sector": "소재", "industry": "화학"},
    "002380.KS": {"name": "KCC", "sector": "소재", "industry": "건축자재"},
    "002390.KS": {"name": "한독", "sector": "헬스케어", "industry": "제약"},
    "002410.KS": {"name": "범양건영", "sector": "산업재", "industry": "건설"},
    "002420.KS": {"name": "세기상사", "sector": "경기소비재", "industry": "전문소매"},
    "002450.KS": {"name": "삼익악기", "sector": "경기소비재", "industry": "레저용장비와제품"},
    "002460.KS": {"name": "HS화성", "sector": "산업재", "industry": "건설"},
    "002600.KS": {"name": "조흥", "sector": "필수소비재", "industry": "식품"},
    "002620.KS": {"name": "제일파마홀딩스", "sector": "헬스케어", "industry": "제약"},
    "002630.KS": {"name": "오리엔트바이오", "sector": "헬스케어", "industry": "생물공학"},
    "002690.KS": {"name": "동일제강", "sector": "소재", "industry": "철강"},
    "002700.KS": {"name": "신일전자", "sector": "경기소비재", "industry": "가정용기기와용품"},
    "002710.KS": {"name": "TCC스틸", "sector": "소재", "industry": "철강"},
    "002720.KS": {"name": "국제약품", "sector": "헬스케어", "industry": "제약"},
    "002760.KS": {"name": "보락", "sector": "필수소비재", "industry": "식품"},
    "002780.KS": {"name": "진흥기업", "sector": "산업재", "industry": "건설"},
    "002790.KS": {"name": "아모레퍼시픽홀딩스", "sector": "필수소비재", "industry": "화장품"},
    "002810.KS": {"name": "삼영무역", "sector": "소재", "industry": "화학"},
    "002820.KS": {"name": "SUN&L", "sector": "소재", "industry": "종이와목재"},
    "002840.KS": {"name": "미원상사", "sector": "소재", "industry": "화학"},
    "002870.KS": {"name": "신풍", "sector": "소재", "industry": "포장재"},
    "002880.KS": {"name": "디와이에이", "sector": "경기소비재", "industry": "자동차부품"},
    "002900.KS": {"name": "TYM", "sector": "산업재", "industry": "기계"},
    "002920.KS": {"name": "유성기업", "sector": "경기소비재", "industry": "자동차부품"},
    "002960.KS": {"name": "한국쉘석유", "sector": "소재", "industry": "화학"},
    "002990.KS": {"name": "금호건설", "sector": "산업재", "industry": "건설"},
    "003000.KS": {"name": "부광약품", "sector": "헬스케어", "industry": "제약"},
    "003010.KS": {"name": "혜인", "sector": "산업재", "industry": "기계"},
    "003030.KS": {"name": "세아제강지주", "sector": "소재", "industry": "철강"},
    "003060.KS": {"name": "에이프로젠바이오로직스", "sector": "헬스케어", "industry": "제약"},
    "003070.KS": {"name": "코오롱글로벌", "sector": "산업재", "industry": "건설"},
    "003080.KS": {"name": "SB성보", "sector": "소재", "industry": "화학"},
    "003090.KS": {"name": "대웅", "sector": "헬스케어", "industry": "제약"},
    "003120.KS": {"name": "일성아이에스", "sector": "헬스케어", "industry": "제약"},
    "003160.KS": {"name": "디아이", "sector": "IT", "industry": "반도체와반도체장비"},
    "003200.KS": {"name": "일신방직", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "003220.KS": {"name": "대원제약", "sector": "헬스케어", "industry": "제약"},
    "003230.KS": {"name": "삼양식품", "sector": "필수소비재", "industry": "식품"},
    "003240.KS": {"name": "태광산업", "sector": "소재", "industry": "화학"},
    "003280.KS": {"name": "흥아해운", "sector": "산업재", "industry": "해운사"},
    "003300.KS": {"name": "한일홀딩스", "sector": "소재", "industry": "건축자재"},
    "003350.KS": {"name": "한국화장품제조", "sector": "필수소비재", "industry": "화장품"},
    "003460.KS": {"name": "유화증권", "sector": "금융", "industry": "증권"},
    "003470.KS": {"name": "유안타증권", "sector": "금융", "industry": "증권"},
    "003480.KS": {"name": "한진중공업홀딩스", "sector": "유틸리티", "industry": "가스유틸리티"},
    "003490.KS": {"name": "대한항공", "sector": "산업재", "industry": "항공사"},
    "003520.KS": {"name": "영진약품", "sector": "헬스케어", "industry": "제약"},
    "003530.KS": {"name": "한화투자증권", "sector": "금융", "industry": "증권"},
    "003540.KS": {"name": "대신증권", "sector": "금융", "industry": "증권"},
    "003550.KS": {"name": "LG", "sector": "산업재", "industry": "복합기업"},
    "003570.KS": {"name": "SNT다이내믹스", "sector": "경기소비재", "industry": "자동차부품"},
    "003580.KS": {"name": "HLB글로벌", "sector": "산업재", "industry": "판매업체"},
    "003610.KS": {"name": "방림", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "003620.KS": {"name": "KG모빌리티", "sector": "경기소비재", "industry": "자동차"},
    "003650.KS": {"name": "미창석유", "sector": "소재", "industry": "화학"},
    "003670.KS": {"name": "포스코퓨처엠", "sector": "소재", "industry": "화학"},
    "003680.KS": {"name": "한성기업", "sector": "필수소비재", "industry": "식품"},
    "003690.KS": {"name": "코리안리", "sector": "금융", "industry": "손해보험"},
    "003720.KS": {"name": "삼영", "sector": "소재", "industry": "화학"},
    "003780.KS": {"name": "진양산업", "sector": "소재", "industry": "화학"},
    "003830.KS": {"name": "대한화섬", "sector": "소재", "industry": "화학"},
    "003850.KS": {"name": "보령", "sector": "헬스케어", "industry": "제약"},
    "003920.KS": {"name": "남양유업", "sector": "필수소비재", "industry": "식품"},
    "003960.KS": {"name": "사조대림", "sector": "필수소비재", "industry": "식품"},
    "004000.KS": {"name": "롯데정밀화학", "sector": "소재", "industry": "화학"},
    "004020.KS": {"name": "현대제철", "sector": "소재", "industry": "철강"},
    "004060.KS": {"name": "SG세계물산", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "004080.KS": {"name": "신흥", "sector": "헬스케어", "industry": "건강관리장비와용품"},
    "004090.KS": {"name": "한국석유", "sector": "소재", "industry": "화학"},
    "004100.KS": {"name": "태양금속", "sector": "경기소비재", "industry": "자동차부품"},
    "004140.KS": {"name": "동방", "sector": "산업재", "industry": "항공화물운송과물류"},
    "004150.KS": {"name": "한솔홀딩스", "sector": "산업재", "industry": "복합기업"},
    "004170.KS": {"name": "신세계", "sector": "경기소비재", "industry": "백화점과일반상점"},
    "004250.KS": {"name": "NPC", "sector": "소재", "industry": "포장재"},
    "004270.KS": {"name": "남성", "sector": "IT", "industry": "전자장비와기기"},
    "004310.KS": {"name": "현대약품", "sector": "헬스케어", "industry": "제약"},
    "004360.KS": {"name": "세방", "sector": "산업재", "industry": "항공화물운송과물류"},
    "004370.KS": {"name": "농심", "sector": "필수소비재", "industry": "식품"},
    "004380.KS": {"name": "삼익THK", "sector": "산업재", "industry": "기계"},
    "004410.KS": {"name": "서울식품", "sector": "필수소비재", "industry": "식품"},
    "004430.KS": {"name": "송원산업", "sector": "소재", "industry": "화학"},
    "004440.KS": {"name": "삼일씨엔에스", "sector": "소재", "industry": "건축자재"},
    "004450.KS": {"name": "삼화왕관", "sector": "소재", "industry": "포장재"},
    "004490.KS": {"name": "세방전지", "sector": "경기소비재", "industry": "자동차부품"},
    "004540.KS": {"name": "깨끗한나라", "sector": "소재", "industry": "종이와목재"},
    "004560.KS": {"name": "현대비앤지스틸", "sector": "소재", "industry": "비철금속"},
    "004690.KS": {"name": "삼천리", "sector": "유틸리티", "industry": "가스유틸리티"},
    "004700.KS": {"name": "조광피혁", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "004710.KS": {"name": "한솔테크닉스", "sector": "IT", "industry": "전자장비와기기"},
    "004720.KS": {"name": "팜젠사이언스", "sector": "헬스케어", "industry": "제약"},
    "004770.KS": {"name": "써니전자", "sector": "IT", "industry": "전자장비와기기"},
    "004800.KS": {"name": "효성", "sector": "산업재", "industry": "복합기업"},
    "004830.KS": {"name": "덕성", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "004840.KS": {"name": "DRB동일", "sector": "소재", "industry": "화학"},
    "004870.KS": {"name": "티웨이홀딩스", "sector": "소재", "industry": "건축자재"},
    "004890.KS": {"name": "동일산업", "sector": "소재", "industry": "철강"},
    "004910.KS": {"name": "조광페인트", "sector": "소재", "industry": "건축자재"},
    "004920.KS": {"name": "씨아이테크", "sector": "IT", "industry": "컴퓨터와주변기기"},
    "004960.KS": {"name": "한신공영", "sector": "산업재", "industry": "건설"},
    "004970.KS": {"name": "신라교역", "sector": "필수소비재", "industry": "식품"},
    "004980.KS": {"name": "성신양회", "sector": "소재", "industry": "건축자재"},
    "004990.KS": {"name": "롯데지주", "sector": "산업재", "industry": "복합기업"},
    "005010.KS": {"name": "휴스틸", "sector": "소재", "industry": "철강"},
    "005030.KS": {"name": "부산주공", "sector": "경기소비재", "industry": "자동차부품"},
    "005070.KS": {"name": "코스모신소재", "sector": "소재", "industry": "화학"},
    "005090.KS": {"name": "SGC에너지", "sector": "산업재", "industry": "에너지장비및서비스"},
    "005110.KS": {"name": "한창", "sector": "부동산", "industry": "부동산"},
    "005180.KS": {"name": "빙그레", "sector": "필수소비재", "industry": "식품"},
    "005250.KS": {"name": "녹십자홀딩스", "sector": "헬스케어", "industry": "제약"},
    "005300.KS": {"name": "롯데칠성", "sector": "필수소비재", "industry": "음료"},
    "005320.KS": {"name": "온타이드", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "005360.KS": {"name": "모나미", "sector": "경기소비재", "industry": "문구류"},
    "005380.KS": {"name": "현대차", "sector": "경기소비재", "industry": "자동차"},
    "005420.KS": {"name": "코스모화학", "sector": "소재", "industry": "화학"},
    "005430.KS": {"name": "한국공항", "sector": "산업재", "industry": "운송인프라"},
    "005440.KS": {"name": "현대지에프홀딩스", "sector": "산업재", "industry": "복합기업"},
    "005490.KS": {"name": "POSCO홀딩스", "sector": "소재", "industry": "철강"},
    "005500.KS": {"name": "삼진제약", "sector": "헬스케어", "industry": "제약"},
    "005610.KS": {"name": "삼립", "sector": "필수소비재", "industry": "식품"},
    "005680.KS": {"name": "삼영전자", "sector": "IT", "industry": "전기제품"},
    "005690.KS": {"name": "파미셀", "sector": "소재", "industry": "화학"},
    "005720.KS": {"name": "넥센", "sector": "경기소비재", "industry": "자동차부품"},
    "005740.KS": {"name": "크라운해태홀딩스", "sector": "필수소비재", "industry": "식품"},
    "005750.KS": {"name": "대림바스", "sector": "소재", "industry": "건축제품"},
    "005800.KS": {"name": "신영와코루", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "005810.KS": {"name": "풍산홀딩스", "sector": "소재", "industry": "비철금속"},
    "005820.KS": {"name": "원림", "sector": "소재", "industry": "포장재"},
    "005830.KS": {"name": "DB손해보험", "sector": "금융", "industry": "손해보험"},
    "005850.KS": {"name": "에스엘", "sector": "경기소비재", "industry": "자동차부품"},
    "005870.KS": {"name": "휴니드", "sector": "산업재", "industry": "우주항공과국방"},
    "005880.KS": {"name": "대한해운", "sector": "산업재", "industry": "해운사"},
    "005930.KS": {"name": "삼성전자", "sector": "IT", "industry": "반도체와반도체장비"},
    "005940.KS": {"name": "NH투자증권", "sector": "금융", "industry": "증권"},
    "005950.KS": {"name": "이수화학", "sector": "소재", "industry": "화학"},
    "005960.KS": {"name": "동부건설", "sector": "산업재", "industry": "건설"},
    "006040.KS": {"name": "동원산업", "sector": "필수소비재", "industry": "식품"},
    "006060.KS": {"name": "화승인더", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "006090.KS": {"name": "사조오양", "sector": "필수소비재", "industry": "식품"},
    "006110.KS": {"name": "삼아알미늄", "sector": "소재", "industry": "비철금속"},
    "006120.KS": {"name": "SK디스커버리", "sector": "에너지", "industry": "석유와가스"},
    "006200.KS": {"name": "한국전자홀딩스", "sector": "IT", "industry": "반도체와반도체장비"},
    "006220.KS": {"name": "제주은행", "sector": "금융", "industry": "은행"},
    "006260.KS": {"name": "LS", "sector": "소재", "industry": "전기장비"},
    "006280.KS": {"name": "녹십자", "sector": "헬스케어", "industry": "제약"},
    "006340.KS": {"name": "대원전선", "sector": "소재", "industry": "전기장비"},
    "006360.KS": {"name": "GS건설", "sector": "산업재", "industry": "건설"},
    "006370.KS": {"name": "대구백화점", "sector": "경기소비재", "industry": "백화점과일반상점"},
    "006380.KS": {"name": "카프로", "sector": "소재", "industry": "화학"},
    "006400.KS": {"name": "삼성SDI", "sector": "IT", "industry": "전기제품"},
    "006490.KS": {"name": "인스코비", "sector": "통신·미디어", "industry": "무선통신서비스"},
    "006570.KS": {"name": "대림통상", "sector": "소재", "industry": "건축제품"},
    "006650.KS": {"name": "대한유화", "sector": "소재", "industry": "화학"},
    "006660.KS": {"name": "삼성공조", "sector": "경기소비재", "industry": "자동차부품"},
    "006740.KS": {"name": "블루산업개발", "sector": "소재", "industry": "종이와목재"},
    "006800.KS": {"name": "미래에셋증권", "sector": "금융", "industry": "증권"},
    "006840.KS": {"name": "AK홀딩스", "sector": "소재", "industry": "화학"},
    "006880.KS": {"name": "신송홀딩스", "sector": "필수소비재", "industry": "식품과기본식료품소매"},
    "006890.KS": {"name": "태경케미컬", "sector": "소재", "industry": "화학"},
    "006980.KS": {"name": "우성", "sector": "필수소비재", "industry": "식품"},
    "007070.KS": {"name": "GS리테일", "sector": "경기소비재", "industry": "백화점과일반상점"},
    "007110.KS": {"name": "일신석재", "sector": "소재", "industry": "건축자재"},
    "007120.KS": {"name": "미래아이앤지", "sector": "IT", "industry": "IT서비스"},
    "007160.KS": {"name": "사조산업", "sector": "필수소비재", "industry": "식품"},
    "007210.KS": {"name": "벽산", "sector": "소재", "industry": "건축자재"},
    "007280.KS": {"name": "한국특강", "sector": "소재", "industry": "철강"},
    "007310.KS": {"name": "오뚜기", "sector": "필수소비재", "industry": "식품"},
    "007340.KS": {"name": "DN오토모티브", "sector": "경기소비재", "industry": "자동차부품"},
    "007460.KS": {"name": "에이프로젠", "sector": "헬스케어", "industry": "제약"},
    "007540.KS": {"name": "샘표", "sector": "필수소비재", "industry": "식품"},
    "007570.KS": {"name": "일양약품", "sector": "헬스케어", "industry": "제약"},
    "007590.KS": {"name": "동방아그로", "sector": "소재", "industry": "화학"},
    "007610.KS": {"name": "선도전기", "sector": "소재", "industry": "전기장비"},
    "007660.KS": {"name": "이수페타시스", "sector": "IT", "industry": "전자장비와기기"},
    "007690.KS": {"name": "국도화학", "sector": "소재", "industry": "화학"},
    "007700.KS": {"name": "F&F홀딩스", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "007810.KS": {"name": "코리아써키트", "sector": "IT", "industry": "전자장비와기기"},
    "007860.KS": {"name": "서연", "sector": "경기소비재", "industry": "자동차부품"},
    "007980.KS": {"name": "TP", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "008040.KS": {"name": "사조동아원", "sector": "필수소비재", "industry": "식품"},
    "008060.KS": {"name": "대덕", "sector": "IT", "industry": "전자장비와기기"},
    "008250.KS": {"name": "이건산업", "sector": "소재", "industry": "건축자재"},
    "008260.KS": {"name": "NI스틸", "sector": "소재", "industry": "건축자재"},
    "008350.KS": {"name": "남선알미늄", "sector": "소재", "industry": "비철금속"},
    "008420.KS": {"name": "문배철강", "sector": "소재", "industry": "철강"},
    "008490.KS": {"name": "서흥", "sector": "헬스케어", "industry": "제약"},
    "008500.KS": {"name": "일정실업", "sector": "경기소비재", "industry": "자동차부품"},
    "008600.KS": {"name": "윌비스", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "008700.KS": {"name": "아남전자", "sector": "IT", "industry": "전자제품"},
    "008730.KS": {"name": "율촌화학", "sector": "소재", "industry": "포장재"},
    "008770.KS": {"name": "호텔신라", "sector": "경기소비재", "industry": "백화점과일반상점"},
    "008870.KS": {"name": "금비", "sector": "소재", "industry": "포장재"},
    "008930.KS": {"name": "한미사이언스", "sector": "헬스케어", "industry": "제약"},
    "008970.KS": {"name": "KBI동양철관", "sector": "소재", "industry": "철강"},
    "009070.KS": {"name": "KCTC", "sector": "산업재", "industry": "항공화물운송과물류"},
    "009140.KS": {"name": "경인전자", "sector": "IT", "industry": "전자제품"},
    "009150.KS": {"name": "삼성전기", "sector": "IT", "industry": "전자장비와기기"},
    "009160.KS": {"name": "SIMPAC", "sector": "소재", "industry": "비철금속"},
    "009180.KS": {"name": "한솔로지스틱스", "sector": "산업재", "industry": "항공화물운송과물류"},
    "009190.KS": {"name": "대양금속", "sector": "소재", "industry": "철강"},
    "009200.KS": {"name": "무림페이퍼", "sector": "소재", "industry": "종이와목재"},
    "009240.KS": {"name": "한샘", "sector": "경기소비재", "industry": "가구"},
    "009270.KS": {"name": "신원", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "009290.KS": {"name": "광동제약", "sector": "헬스케어", "industry": "제약"},
    "009310.KS": {"name": "참엔지니어링", "sector": "IT", "industry": "디스플레이장비및부품"},
    "009320.KS": {"name": "아진전자부품", "sector": "경기소비재", "industry": "자동차부품"},
    "009410.KS": {"name": "태영건설", "sector": "산업재", "industry": "건설"},
    "009420.KS": {"name": "한올바이오파마", "sector": "헬스케어", "industry": "제약"},
    "009440.KS": {"name": "KC그린홀딩스", "sector": "산업재", "industry": "상업서비스와공급품"},
    "009450.KS": {"name": "경동나비엔", "sector": "경기소비재", "industry": "가정용기기와용품"},
    "009460.KS": {"name": "한창제지", "sector": "소재", "industry": "포장재"},
    "009470.KS": {"name": "삼화전기", "sector": "IT", "industry": "전기제품"},
    "009540.KS": {"name": "HD한국조선해양", "sector": "산업재", "industry": "조선"},
    "009580.KS": {"name": "무림P&P", "sector": "소재", "industry": "종이와목재"},
    "009680.KS": {"name": "모토닉", "sector": "경기소비재", "industry": "자동차부품"},
    "009770.KS": {"name": "삼정펄프", "sector": "필수소비재", "industry": "가정용품"},
    "009810.KS": {"name": "플레이그램", "sector": "산업재", "industry": "판매업체"},
    "009830.KS": {"name": "한화솔루션", "sector": "산업재", "industry": "에너지장비및서비스"},
    "009900.KS": {"name": "명신산업", "sector": "경기소비재", "industry": "자동차부품"},
    "009970.KS": {"name": "영원무역홀딩스", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "010040.KS": {"name": "한국내화", "sector": "소재", "industry": "비철금속"},
    "010060.KS": {"name": "OCI홀딩스", "sector": "소재", "industry": "화학"},
    "010100.KS": {"name": "한국무브넥스", "sector": "경기소비재", "industry": "자동차부품"},
    "010120.KS": {"name": "LS ELECTRIC", "sector": "소재", "industry": "전기장비"},
    "010130.KS": {"name": "고려아연", "sector": "소재", "industry": "비철금속"},
    "010140.KS": {"name": "삼성중공업", "sector": "산업재", "industry": "조선"},
    "010400.KS": {"name": "우진아이엔에스", "sector": "산업재", "industry": "건설"},
    "010580.KS": {"name": "에스엠벡셀", "sector": "경기소비재", "industry": "자동차부품"},
    "010640.KS": {"name": "진양폴리", "sector": "소재", "industry": "화학"},
    "010660.KS": {"name": "화천기계", "sector": "산업재", "industry": "기계"},
    "010690.KS": {"name": "화신", "sector": "경기소비재", "industry": "자동차부품"},
    "010770.KS": {"name": "평화홀딩스", "sector": "경기소비재", "industry": "자동차부품"},
    "010780.KS": {"name": "아이에스동서", "sector": "산업재", "industry": "건설"},
    "010820.KS": {"name": "퍼스텍", "sector": "산업재", "industry": "우주항공과국방"},
    "010950.KS": {"name": "S-Oil", "sector": "에너지", "industry": "석유와가스"},
    "010960.KS": {"name": "삼호개발", "sector": "산업재", "industry": "건설"},
    "011000.KS": {"name": "진원생명과학", "sector": "헬스케어", "industry": "생물공학"},
    "011070.KS": {"name": "LG이노텍", "sector": "IT", "industry": "전자장비와기기"},
    "011090.KS": {"name": "에넥스", "sector": "경기소비재", "industry": "가구"},
    "011150.KS": {"name": "CJ씨푸드", "sector": "필수소비재", "industry": "식품"},
    "011170.KS": {"name": "롯데케미칼", "sector": "소재", "industry": "화학"},
    "011200.KS": {"name": "HMM", "sector": "산업재", "industry": "해운사"},
    "011210.KS": {"name": "현대위아", "sector": "경기소비재", "industry": "자동차부품"},
    "011230.KS": {"name": "삼화전자", "sector": "IT", "industry": "전자장비와기기"},
    "011280.KS": {"name": "태림포장", "sector": "소재", "industry": "포장재"},
    "011300.KS": {"name": "우성머티리얼스", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "011330.KS": {"name": "유니켐", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "011390.KS": {"name": "부산산업", "sector": "소재", "industry": "건축자재"},
    "011420.KS": {"name": "갤럭시아에스엠", "sector": "경기소비재", "industry": "레저용장비와제품"},
    "011500.KS": {"name": "한농화성", "sector": "소재", "industry": "화학"},
    "011690.KS": {"name": "와이투솔루션", "sector": "IT", "industry": "전자장비와기기"},
    "011700.KS": {"name": "한신기계", "sector": "산업재", "industry": "기계"},
    "011760.KS": {"name": "현대코퍼레이션", "sector": "산업재", "industry": "무역회사와판매업체"},
    "011780.KS": {"name": "금호석유화학", "sector": "소재", "industry": "화학"},
    "011790.KS": {"name": "SKC", "sector": "소재", "industry": "화학"},
    "011810.KS": {"name": "STX", "sector": "산업재", "industry": "무역회사와판매업체"},
    "011930.KS": {"name": "신성이엔지", "sector": "IT", "industry": "반도체와반도체장비"},
    "012030.KS": {"name": "DB", "sector": "IT", "industry": "IT서비스"},
    "012160.KS": {"name": "영흥", "sector": "소재", "industry": "철강"},
    "012170.KS": {"name": "아센디오", "sector": "통신·미디어", "industry": "방송과엔터테인먼트"},
    "012200.KS": {"name": "계양전기", "sector": "경기소비재", "industry": "자동차부품"},
    "012280.KS": {"name": "영화금속", "sector": "경기소비재", "industry": "자동차부품"},
    "012320.KS": {"name": "경동인베스트", "sector": "소재", "industry": "비철금속"},
    "012330.KS": {"name": "현대모비스", "sector": "경기소비재", "industry": "자동차부품"},
    "012450.KS": {"name": "한화에어로스페이스", "sector": "산업재", "industry": "우주항공과국방"},
    "012510.KS": {"name": "더존비즈온", "sector": "IT", "industry": "소프트웨어"},
    "012610.KS": {"name": "경인양행", "sector": "소재", "industry": "화학"},
    "012630.KS": {"name": "HDC", "sector": "산업재", "industry": "건설"},
    "012690.KS": {"name": "모나리자", "sector": "필수소비재", "industry": "가정용품"},
    "012750.KS": {"name": "에스원", "sector": "산업재", "industry": "상업서비스와공급품"},
    "012800.KS": {"name": "대창", "sector": "소재", "industry": "비철금속"},
    "013000.KS": {"name": "세우글로벌", "sector": "소재", "industry": "화학"},
    "013360.KS": {"name": "일성건설", "sector": "산업재", "industry": "건설"},
    "013520.KS": {"name": "화승코퍼레이션", "sector": "경기소비재", "industry": "자동차부품"},
    "013570.KS": {"name": "디와이", "sector": "경기소비재", "industry": "자동차부품"},
    "013580.KS": {"name": "계룡건설", "sector": "산업재", "industry": "건설"},
    "013700.KS": {"name": "까뮤이앤씨", "sector": "산업재", "industry": "건설"},
    "013870.KS": {"name": "지엠비코리아", "sector": "경기소비재", "industry": "자동차부품"},
    "013890.KS": {"name": "지누스", "sector": "경기소비재", "industry": "가구"},
    "014130.KS": {"name": "한익스프레스", "sector": "산업재", "industry": "항공화물운송과물류"},
    "014160.KS": {"name": "대영포장", "sector": "소재", "industry": "포장재"},
    "014280.KS": {"name": "금강공업", "sector": "소재", "industry": "건축제품"},
    "014440.KS": {"name": "영보화학", "sector": "소재", "industry": "화학"},
    "014530.KS": {"name": "극동유화", "sector": "에너지", "industry": "석유와가스"},
    "014580.KS": {"name": "태경비케이", "sector": "소재", "industry": "비철금속"},
    "014680.KS": {"name": "한솔케미칼", "sector": "소재", "industry": "화학"},
    "014710.KS": {"name": "사조씨푸드", "sector": "필수소비재", "industry": "식품"},
    "014790.KS": {"name": "HL D&I", "sector": "산업재", "industry": "건설"},
    "014820.KS": {"name": "동원시스템즈", "sector": "소재", "industry": "포장재"},
    "014830.KS": {"name": "유니드", "sector": "소재", "industry": "화학"},
    "014910.KS": {"name": "성문전자", "sector": "IT", "industry": "전기제품"},
    "014990.KS": {"name": "인디에프", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "015020.KS": {"name": "이스타코", "sector": "부동산", "industry": "부동산"},
    "015230.KS": {"name": "대창단조", "sector": "산업재", "industry": "기계"},
    "015260.KS": {"name": "에이엔피", "sector": "IT", "industry": "전자장비와기기"},
    "015360.KS": {"name": "INVENI", "sector": "유틸리티", "industry": "가스유틸리티"},
    "015590.KS": {"name": "DKME", "sector": "산업재", "industry": "기계"},
    "015760.KS": {"name": "한국전력", "sector": "유틸리티", "industry": "전기유틸리티"},
    "015860.KS": {"name": "일진홀딩스", "sector": "소재", "industry": "전기장비"},
    "015890.KS": {"name": "태경산업", "sector": "소재", "industry": "비철금속"},
    "016090.KS": {"name": "대현", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "016360.KS": {"name": "삼성증권", "sector": "금융", "industry": "증권"},
    "016380.KS": {"name": "KG스틸", "sector": "소재", "industry": "철강"},
    "016450.KS": {"name": "한세예스24홀딩스", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "016580.KS": {"name": "환인제약", "sector": "헬스케어", "industry": "제약"},
    "016590.KS": {"name": "신대양제지", "sector": "소재", "industry": "포장재"},
    "016610.KS": {"name": "DB증권", "sector": "금융", "industry": "증권"},
    "016710.KS": {"name": "대성홀딩스", "sector": "유틸리티", "industry": "가스유틸리티"},
    "016740.KS": {"name": "두올", "sector": "경기소비재", "industry": "자동차부품"},
    "016800.KS": {"name": "퍼시스", "sector": "경기소비재", "industry": "가구"},
    "016880.KS": {"name": "웅진", "sector": "통신·미디어", "industry": "출판"},
    "017040.KS": {"name": "광명전기", "sector": "소재", "industry": "전기장비"},
    "017180.KS": {"name": "명문제약", "sector": "헬스케어", "industry": "제약"},
    "017370.KS": {"name": "우신시스템", "sector": "경기소비재", "industry": "자동차부품"},
    "017390.KS": {"name": "서울가스", "sector": "유틸리티", "industry": "가스유틸리티"},
    "017550.KS": {"name": "수산세보틱스", "sector": "산업재", "industry": "기계"},
    "017670.KS": {"name": "SK텔레콤", "sector": "통신·미디어", "industry": "무선통신서비스"},
    "017800.KS": {"name": "현대엘리베이터", "sector": "산업재", "industry": "기계"},
    "017810.KS": {"name": "풀무원", "sector": "필수소비재", "industry": "식품"},
    "017860.KS": {"name": "DS단석", "sector": "소재", "industry": "화학"},
    "017900.KS": {"name": "광전자", "sector": "IT", "industry": "전자장비와기기"},
    "017940.KS": {"name": "E1", "sector": "유틸리티", "industry": "가스유틸리티"},
    "017960.KS": {"name": "한국카본", "sector": "산업재", "industry": "조선"},
    "018250.KS": {"name": "애경산업", "sector": "필수소비재", "industry": "화장품"},
    "018260.KS": {"name": "삼성에스디에스", "sector": "IT", "industry": "IT서비스"},
    "018470.KS": {"name": "조일알미늄", "sector": "소재", "industry": "비철금속"},
    "018500.KS": {"name": "동원금속", "sector": "경기소비재", "industry": "자동차부품"},
    "018670.KS": {"name": "SK가스", "sector": "유틸리티", "industry": "가스유틸리티"},
    "018880.KS": {"name": "한온시스템", "sector": "경기소비재", "industry": "자동차부품"},
    "019170.KS": {"name": "신풍제약", "sector": "헬스케어", "industry": "제약"},
    "019180.KS": {"name": "티에이치엔", "sector": "경기소비재", "industry": "자동차부품"},
    "019490.KS": {"name": "엑시큐어하이트론", "sector": "통신·미디어", "industry": "통신장비"},
    "019680.KS": {"name": "대교", "sector": "경기소비재", "industry": "교육서비스"},
    "020000.KS": {"name": "한섬", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "020120.KS": {"name": "키다리스튜디오", "sector": "통신·미디어", "industry": "양방향미디어와서비스"},
    "020150.KS": {"name": "롯데에너지머티리얼즈", "sector": "IT", "industry": "전자장비와기기"},
    "020560.KS": {"name": "아시아나항공", "sector": "산업재", "industry": "항공사"},
    "020760.KS": {"name": "일진디스플", "sector": "IT", "industry": "디스플레이패널"},
    "021050.KS": {"name": "서원", "sector": "소재", "industry": "비철금속"},
    "021240.KS": {"name": "코웨이", "sector": "경기소비재", "industry": "가정용기기와용품"},
    "021820.KS": {"name": "세원정공", "sector": "경기소비재", "industry": "자동차부품"},
    "022100.KS": {"name": "포스코DX", "sector": "IT", "industry": "IT서비스"},
    "023000.KS": {"name": "삼원강재", "sector": "경기소비재", "industry": "자동차부품"},
    "023150.KS": {"name": "MH에탄올", "sector": "필수소비재", "industry": "음료"},
    "023350.KS": {"name": "한국종합기술", "sector": "산업재", "industry": "건설"},
    "023450.KS": {"name": "동남합성", "sector": "소재", "industry": "화학"},
    "023530.KS": {"name": "롯데쇼핑", "sector": "경기소비재", "industry": "백화점과일반상점"},
    "023590.KS": {"name": "다우기술", "sector": "금융", "industry": "증권"},
    "023800.KS": {"name": "인지컨트롤스", "sector": "경기소비재", "industry": "자동차부품"},
    "023810.KS": {"name": "인팩", "sector": "경기소비재", "industry": "자동차부품"},
    "023960.KS": {"name": "에쓰씨엔지니어링", "sector": "산업재", "industry": "건설"},
    "024070.KS": {"name": "WISCOM", "sector": "소재", "industry": "화학"},
    "024090.KS": {"name": "디씨엠", "sector": "소재", "industry": "비철금속"},
    "024110.KS": {"name": "기업은행", "sector": "금융", "industry": "은행"},
    "024720.KS": {"name": "콜마홀딩스", "sector": "필수소비재", "industry": "화장품"},
    "024890.KS": {"name": "대원화성", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "024900.KS": {"name": "디와이덕양", "sector": "경기소비재", "industry": "자동차부품"},
    "025000.KS": {"name": "KPX케미칼", "sector": "소재", "industry": "화학"},
    "025530.KS": {"name": "SJM홀딩스", "sector": "경기소비재", "industry": "자동차부품"},
    "025540.KS": {"name": "한국단자", "sector": "경기소비재", "industry": "자동차부품"},
    "025560.KS": {"name": "미래산업", "sector": "IT", "industry": "반도체와반도체장비"},
    "025620.KS": {"name": "차AI헬스케어", "sector": "필수소비재", "industry": "화장품"},
    "025750.KS": {"name": "한솔홈데코", "sector": "소재", "industry": "건축자재"},
    "025820.KS": {"name": "이구산업", "sector": "소재", "industry": "비철금속"},
    "025860.KS": {"name": "남해화학", "sector": "소재", "industry": "화학"},
    "025890.KS": {"name": "한국주강", "sector": "소재", "industry": "철강"},
    "026890.KS": {"name": "스틱인베스트먼트", "sector": "금융", "industry": "창업투자"},
    "026940.KS": {"name": "부국철강", "sector": "소재", "industry": "철강"},
    "026960.KS": {"name": "동서", "sector": "필수소비재", "industry": "식품"},
    "027410.KS": {"name": "BGF", "sector": "소재", "industry": "화학"},
    "027740.KS": {"name": "마니커", "sector": "필수소비재", "industry": "식품"},
    "027970.KS": {"name": "한국제지", "sector": "소재", "industry": "종이와목재"},
    "028050.KS": {"name": "삼성E&A", "sector": "산업재", "industry": "건설"},
    "028100.KS": {"name": "동아지질", "sector": "산업재", "industry": "건설"},
    "028260.KS": {"name": "삼성물산", "sector": "산업재", "industry": "복합기업"},
    "028670.KS": {"name": "팬오션", "sector": "산업재", "industry": "해운사"},
    "029460.KS": {"name": "케이씨", "sector": "IT", "industry": "반도체와반도체장비"},
    "029530.KS": {"name": "신도리코", "sector": "IT", "industry": "사무용전자제품"},
    "029780.KS": {"name": "삼성카드", "sector": "금융", "industry": "카드"},
    "030000.KS": {"name": "제일기획", "sector": "통신·미디어", "industry": "광고"},
    "030190.KS": {"name": "NICE평가정보", "sector": "산업재", "industry": "상업서비스와공급품"},
    "030200.KS": {"name": "KT", "sector": "통신·미디어", "industry": "다각화된통신서비스"},
    "030210.KS": {"name": "다올투자증권", "sector": "금융", "industry": "증권"},
    "030610.KS": {"name": "교보증권", "sector": "금융", "industry": "증권"},
    "030720.KS": {"name": "동원수산", "sector": "필수소비재", "industry": "식품"},
    "031210.KS": {"name": "서울보증보험", "sector": "금융", "industry": "손해보험"},
    "031430.KS": {"name": "신세계인터내셔날", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "031440.KS": {"name": "신세계푸드", "sector": "경기소비재", "industry": "호텔,레스토랑,레저"},
    "031820.KS": {"name": "아이티센씨티에스", "sector": "IT", "industry": "IT서비스"},
    "032350.KS": {"name": "롯데관광개발", "sector": "경기소비재", "industry": "호텔,레스토랑,레저"},
    "032560.KS": {"name": "황금에스티", "sector": "소재", "industry": "비철금속"},
    "032640.KS": {"name": "LG유플러스", "sector": "통신·미디어", "industry": "무선통신서비스"},
    "032830.KS": {"name": "삼성생명", "sector": "금융", "industry": "생명보험"},
    "033240.KS": {"name": "자화전자", "sector": "통신·미디어", "industry": "통신장비"},
    "033250.KS": {"name": "체시스", "sector": "경기소비재", "industry": "자동차부품"},
    "033270.KS": {"name": "유나이티드제약", "sector": "헬스케어", "industry": "제약"},
    "033530.KS": {"name": "SJG세종", "sector": "경기소비재", "industry": "자동차부품"},
    "033780.KS": {"name": "KT&G", "sector": "필수소비재", "industry": "담배"},
    "033920.KS": {"name": "무학", "sector": "필수소비재", "industry": "음료"},
    "034020.KS": {"name": "두산에너빌리티", "sector": "산업재", "industry": "기계"},
    "034120.KS": {"name": "SBS", "sector": "통신·미디어", "industry": "방송과엔터테인먼트"},
    "034220.KS": {"name": "LG디스플레이", "sector": "IT", "industry": "디스플레이패널"},
    "034230.KS": {"name": "파라다이스", "sector": "경기소비재", "industry": "호텔,레스토랑,레저"},
    "034310.KS": {"name": "NICE", "sector": "IT", "industry": "IT서비스"},
    "034590.KS": {"name": "인천도시가스", "sector": "유틸리티", "industry": "가스유틸리티"},
    "034730.KS": {"name": "SK", "sector": "에너지", "industry": "석유와가스"},
    "034830.KS": {"name": "한국토지신탁", "sector": "부동산", "industry": "부동산"},
    "035000.KS": {"name": "HS애드", "sector": "통신·미디어", "industry": "광고"},
    "035150.KS": {"name": "백산", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "035250.KS": {"name": "강원랜드", "sector": "경기소비재", "industry": "호텔,레스토랑,레저"},
    "035420.KS": {"name": "NAVER", "sector": "통신·미디어", "industry": "양방향미디어와서비스"},
    "035510.KS": {"name": "신세계 I&C", "sector": "IT", "industry": "IT서비스"},
    "035720.KS": {"name": "카카오", "sector": "통신·미디어", "industry": "양방향미디어와서비스"},
    "036420.KS": {"name": "콘텐트리중앙", "sector": "통신·미디어", "industry": "방송과엔터테인먼트"},
    "036460.KS": {"name": "한국가스공사", "sector": "유틸리티", "industry": "가스유틸리티"},
    "036530.KS": {"name": "SNT홀딩스", "sector": "경기소비재", "industry": "자동차부품"},
    "036570.KS": {"name": "NC", "sector": "IT", "industry": "게임엔터테인먼트"},
    "036580.KS": {"name": "팜스코", "sector": "필수소비재", "industry": "식품"},
    "037270.KS": {"name": "YG PLUS", "sector": "통신·미디어", "industry": "방송과엔터테인먼트"},
    "037560.KS": {"name": "LG헬로비전", "sector": "통신·미디어", "industry": "방송과엔터테인먼트"},
    "037710.KS": {"name": "광주신세계", "sector": "경기소비재", "industry": "백화점과일반상점"},
    "039130.KS": {"name": "하나투어", "sector": "경기소비재", "industry": "호텔,레스토랑,레저"},
    "039490.KS": {"name": "키움증권", "sector": "금융", "industry": "증권"},
    "039570.KS": {"name": "HDC랩스", "sector": "산업재", "industry": "건설"},
    "041650.KS": {"name": "상신브레이크", "sector": "경기소비재", "industry": "자동차부품"},
    "042660.KS": {"name": "한화오션", "sector": "산업재", "industry": "조선"},
    "042700.KS": {"name": "한미반도체", "sector": "IT", "industry": "반도체와반도체장비"},
    "044380.KS": {"name": "주연테크", "sector": "IT", "industry": "컴퓨터와주변기기"},
    "044450.KS": {"name": "KSS해운", "sector": "산업재", "industry": "해운사"},
    "044820.KS": {"name": "코스맥스비티아이", "sector": "헬스케어", "industry": "건강관리업체및서비스"},
    "047040.KS": {"name": "대우건설", "sector": "산업재", "industry": "건설"},
    "047050.KS": {"name": "포스코인터내셔널", "sector": "산업재", "industry": "무역회사와판매업체"},
    "047400.KS": {"name": "유니온머티리얼", "sector": "경기소비재", "industry": "자동차부품"},
    "047810.KS": {"name": "한국항공우주", "sector": "산업재", "industry": "우주항공과국방"},
    "049800.KS": {"name": "우진플라임", "sector": "산업재", "industry": "기계"},
    "051600.KS": {"name": "한전KPS", "sector": "유틸리티", "industry": "전기유틸리티"},
    "051630.KS": {"name": "진양화학", "sector": "소재", "industry": "화학"},
    "051900.KS": {"name": "LG생활건강", "sector": "필수소비재", "industry": "화장품"},
    "051910.KS": {"name": "LG화학", "sector": "소재", "industry": "화학"},
    "052690.KS": {"name": "한전기술", "sector": "유틸리티", "industry": "전기유틸리티"},
    "053210.KS": {"name": "스카이라이프", "sector": "통신·미디어", "industry": "방송과엔터테인먼트"},
    "053690.KS": {"name": "한미글로벌", "sector": "산업재", "industry": "건설"},
    "055490.KS": {"name": "테이팩스", "sector": "IT", "industry": "전자장비와기기"},
    "055550.KS": {"name": "신한지주", "sector": "금융", "industry": "은행"},
    "057050.KS": {"name": "현대홈쇼핑", "sector": "IT", "industry": "인터넷과카탈로그소매"},
    "058430.KS": {"name": "포스코스틸리온", "sector": "소재", "industry": "비철금속"},
    "058650.KS": {"name": "세아홀딩스", "sector": "소재", "industry": "철강"},
    "058730.KS": {"name": "다스코", "sector": "소재", "industry": "건축자재"},
    "058850.KS": {"name": "KTcs", "sector": "산업재", "industry": "상업서비스와공급품"},
    "058860.KS": {"name": "KTis", "sector": "산업재", "industry": "상업서비스와공급품"},
    "060980.KS": {"name": "HL홀딩스", "sector": "경기소비재", "industry": "자동차부품"},
    "062040.KS": {"name": "산일전기", "sector": "소재", "industry": "전기장비"},
    "063160.KS": {"name": "종근당바이오", "sector": "헬스케어", "industry": "제약"},
    "064350.KS": {"name": "현대로템", "sector": "산업재", "industry": "우주항공과국방"},
    "064400.KS": {"name": "LG씨엔에스", "sector": "IT", "industry": "IT서비스"},
    "064960.KS": {"name": "SNT모티브", "sector": "경기소비재", "industry": "자동차부품"},
    "066570.KS": {"name": "LG전자", "sector": "IT", "industry": "전자제품"},
    "066970.KS": {"name": "엘앤에프", "sector": "IT", "industry": "전기제품"},
    "067830.KS": {"name": "세이브존I&C", "sector": "경기소비재", "industry": "백화점과일반상점"},
    "068270.KS": {"name": "셀트리온", "sector": "헬스케어", "industry": "제약"},
    "068290.KS": {"name": "삼성출판사", "sector": "통신·미디어", "industry": "출판"},
    "069260.KS": {"name": "TKG휴켐스", "sector": "소재", "industry": "화학"},
    "069460.KS": {"name": "대호에이엘", "sector": "소재", "industry": "비철금속"},
    "069620.KS": {"name": "대웅제약", "sector": "헬스케어", "industry": "제약"},
    "069640.KS": {"name": "한세엠케이", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "069730.KS": {"name": "DSR제강", "sector": "소재", "industry": "철강"},
    "069960.KS": {"name": "현대백화점", "sector": "경기소비재", "industry": "백화점과일반상점"},
    "070960.KS": {"name": "모나용평", "sector": "경기소비재", "industry": "호텔,레스토랑,레저"},
    "071050.KS": {"name": "한국금융지주", "sector": "금융", "industry": "증권"},
    "071090.KS": {"name": "하이스틸", "sector": "소재", "industry": "철강"},
    "071320.KS": {"name": "지역난방공사", "sector": "유틸리티", "industry": "복합유틸리티"},
    "071950.KS": {"name": "코아스", "sector": "경기소비재", "industry": "가구"},
    "071970.KS": {"name": "HD현대마린엔진", "sector": "산업재", "industry": "조선"},
    "072130.KS": {"name": "유엔젤", "sector": "IT", "industry": "IT서비스"},
    "072710.KS": {"name": "농심홀딩스", "sector": "필수소비재", "industry": "식품"},
    "073240.KS": {"name": "금호타이어", "sector": "경기소비재", "industry": "자동차부품"},
    "074610.KS": {"name": "이엔플러스", "sector": "산업재", "industry": "기계"},
    "075180.KS": {"name": "새론오토모티브", "sector": "경기소비재", "industry": "자동차부품"},
    "075580.KS": {"name": "세진중공업", "sector": "산업재", "industry": "조선"},
    "077500.KS": {"name": "유니퀘스트", "sector": "IT", "industry": "반도체와반도체장비"},
    "077970.KS": {"name": "STX엔진", "sector": "산업재", "industry": "조선"},
    "078000.KS": {"name": "텔코웨어", "sector": "IT", "industry": "소프트웨어"},
    "078520.KS": {"name": "에이블씨엔씨", "sector": "필수소비재", "industry": "화장품"},
    "078930.KS": {"name": "GS", "sector": "에너지", "industry": "석유와가스"},
    "079160.KS": {"name": "CJ CGV", "sector": "통신·미디어", "industry": "방송과엔터테인먼트"},
    "079430.KS": {"name": "현대리바트", "sector": "경기소비재", "industry": "가구"},
    "079550.KS": {"name": "LIG디펜스앤에어로스페이스", "sector": "산업재", "industry": "우주항공과국방"},
    "079900.KS": {"name": "전진건설로봇", "sector": "산업재", "industry": "기계"},
    "079980.KS": {"name": "휴비스", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "081000.KS": {"name": "일진다이아", "sector": "소재", "industry": "비철금속"},
    "081660.KS": {"name": "미스토홀딩스", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "082640.KS": {"name": "동양생명", "sector": "금융", "industry": "생명보험"},
    "082740.KS": {"name": "한화엔진", "sector": "산업재", "industry": "조선"},
    "083420.KS": {"name": "그린케미칼", "sector": "소재", "industry": "화학"},
    "084010.KS": {"name": "대한제강", "sector": "소재", "industry": "철강"},
    "084670.KS": {"name": "동양고속", "sector": "산업재", "industry": "도로와철도운송"},
    "084680.KS": {"name": "이월드", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "084690.KS": {"name": "대상홀딩스", "sector": "필수소비재", "industry": "식품"},
    "084870.KS": {"name": "TBH글로벌", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "085310.KS": {"name": "엔케이", "sector": "산업재", "industry": "조선"},
    "085620.KS": {"name": "미래에셋생명", "sector": "금융", "industry": "생명보험"},
    "086280.KS": {"name": "현대글로비스", "sector": "산업재", "industry": "항공화물운송과물류"},
    "086790.KS": {"name": "하나금융지주", "sector": "금융", "industry": "은행"},
    "088350.KS": {"name": "한화생명", "sector": "금융", "industry": "생명보험"},
    "088790.KS": {"name": "진도", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "089470.KS": {"name": "HDC현대EP", "sector": "경기소비재", "industry": "자동차부품"},
    "089590.KS": {"name": "제주항공", "sector": "산업재", "industry": "항공사"},
    "089860.KS": {"name": "롯데렌탈", "sector": "산업재", "industry": "도로와철도운송"},
    "090080.KS": {"name": "평화산업", "sector": "경기소비재", "industry": "자동차부품"},
    "090350.KS": {"name": "노루페인트", "sector": "소재", "industry": "건축자재"},
    "090370.KS": {"name": "메타랩스", "sector": "산업재", "industry": "상업서비스와공급품"},
    "090430.KS": {"name": "아모레퍼시픽", "sector": "필수소비재", "industry": "화장품"},
    "090460.KS": {"name": "비에이치", "sector": "IT", "industry": "전자장비와기기"},
    "091810.KS": {"name": "트리니티항공", "sector": "산업재", "industry": "항공사"},
    "092200.KS": {"name": "디아이씨", "sector": "경기소비재", "industry": "자동차부품"},
    "092220.KS": {"name": "KEC", "sector": "IT", "industry": "반도체와반도체장비"},
    "092230.KS": {"name": "KPX홀딩스", "sector": "소재", "industry": "화학"},
    "092440.KS": {"name": "기신정기", "sector": "산업재", "industry": "기계"},
    "092780.KS": {"name": "DYP", "sector": "경기소비재", "industry": "자동차부품"},
    "092790.KS": {"name": "넥스틸", "sector": "소재", "industry": "철강"},
    "093050.KS": {"name": "LF", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "093240.KS": {"name": "형지엘리트", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "093370.KS": {"name": "후성", "sector": "소재", "industry": "화학"},
    "094280.KS": {"name": "효성ITX", "sector": "산업재", "industry": "상업서비스와공급품"},
    "095570.KS": {"name": "AJ네트웍스", "sector": "산업재", "industry": "도로와철도운송"},
    "095720.KS": {"name": "웅진씽크빅", "sector": "경기소비재", "industry": "교육서비스"},
    "096760.KS": {"name": "JW홀딩스", "sector": "헬스케어", "industry": "제약"},
    "096770.KS": {"name": "SK이노베이션", "sector": "에너지", "industry": "석유와가스"},
    "097230.KS": {"name": "HJ중공업", "sector": "산업재", "industry": "건설"},
    "097520.KS": {"name": "엠씨넥스", "sector": "통신·미디어", "industry": "핸드셋"},
    "097950.KS": {"name": "CJ제일제당", "sector": "필수소비재", "industry": "식품"},
    "100090.KS": {"name": "SK오션플랜트", "sector": "산업재", "industry": "에너지장비및서비스"},
    "100220.KS": {"name": "비상교육", "sector": "경기소비재", "industry": "교육서비스"},
    "100250.KS": {"name": "진양홀딩스", "sector": "경기소비재", "industry": "자동차부품"},
    "100840.KS": {"name": "SNT에너지", "sector": "산업재", "industry": "에너지장비및서비스"},
    "101140.KS": {"name": "인바이오젠", "sector": "IT", "industry": "전자장비와기기"},
    "101530.KS": {"name": "해태제과식품", "sector": "필수소비재", "industry": "식품"},
    "102260.KS": {"name": "동성케미컬", "sector": "소재", "industry": "화학"},
    "102460.KS": {"name": "이연제약", "sector": "헬스케어", "industry": "제약"},
    "103140.KS": {"name": "풍산", "sector": "소재", "industry": "비철금속"},
    "103590.KS": {"name": "일진전기", "sector": "소재", "industry": "전기장비"},
    "104700.KS": {"name": "한국철강", "sector": "소재", "industry": "철강"},
    "105560.KS": {"name": "KB금융", "sector": "금융", "industry": "은행"},
    "105630.KS": {"name": "한세실업", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "105840.KS": {"name": "우진", "sector": "산업재", "industry": "기계"},
    "107590.KS": {"name": "미원홀딩스", "sector": "소재", "industry": "화학"},
    "108320.KS": {"name": "LX세미콘", "sector": "IT", "industry": "디스플레이장비및부품"},
    "108670.KS": {"name": "LX하우시스", "sector": "소재", "industry": "건축자재"},
    "109070.KS": {"name": "주성코퍼레이션", "sector": "산업재", "industry": "해운사"},
    "111110.KS": {"name": "호전실업", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "111380.KS": {"name": "동인기연", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "111770.KS": {"name": "영원무역", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "112610.KS": {"name": "씨에스윈드", "sector": "산업재", "industry": "에너지장비및서비스"},
    "114090.KS": {"name": "GKL", "sector": "경기소비재", "industry": "호텔,레스토랑,레저"},
    "117580.KS": {"name": "대성에너지", "sector": "유틸리티", "industry": "가스유틸리티"},
    "118000.KS": {"name": "메타케어", "sector": "헬스케어", "industry": "제약"},
    "119650.KS": {"name": "KC코트렐", "sector": "산업재", "industry": "상업서비스와공급품"},
    "120030.KS": {"name": "조선선재", "sector": "소재", "industry": "철강"},
    "120110.KS": {"name": "코오롱인더", "sector": "소재", "industry": "화학"},
    "122900.KS": {"name": "아이마켓코리아", "sector": "산업재", "industry": "상업서비스와공급품"},
    "123690.KS": {"name": "한국화장품", "sector": "필수소비재", "industry": "화장품"},
    "123700.KS": {"name": "SJM", "sector": "경기소비재", "industry": "자동차부품"},
    "123890.KS": {"name": "한국자산신탁", "sector": "부동산", "industry": "부동산"},
    "126560.KS": {"name": "현대퓨처넷", "sector": "필수소비재", "industry": "화장품"},
    "126720.KS": {"name": "수산인더스트리", "sector": "산업재", "industry": "에너지장비및서비스"},
    "128820.KS": {"name": "대성산업", "sector": "에너지", "industry": "석유와가스"},
    "128940.KS": {"name": "한미약품", "sector": "헬스케어", "industry": "제약"},
    "129260.KS": {"name": "인터지스", "sector": "산업재", "industry": "해운사"},
    "130660.KS": {"name": "한전산업", "sector": "유틸리티", "industry": "전기유틸리티"},
    "133820.KS": {"name": "화인베스틸", "sector": "소재", "industry": "철강"},
    "134380.KS": {"name": "미원화학", "sector": "소재", "industry": "화학"},
    "134790.KS": {"name": "시디즈", "sector": "경기소비재", "industry": "가구"},
    "136490.KS": {"name": "선진", "sector": "필수소비재", "industry": "식품"},
    "137310.KS": {"name": "에스디바이오센서", "sector": "헬스케어", "industry": "건강관리장비와용품"},
    "138040.KS": {"name": "메리츠금융지주", "sector": "금융", "industry": "증권"},
    "138930.KS": {"name": "BNK금융지주", "sector": "금융", "industry": "은행"},
    "139130.KS": {"name": "iM금융지주", "sector": "금융", "industry": "은행"},
    "139480.KS": {"name": "이마트", "sector": "경기소비재", "industry": "백화점과일반상점"},
    "139990.KS": {"name": "아주스틸", "sector": "소재", "industry": "철강"},
    "143210.KS": {"name": "핸즈코퍼레이션", "sector": "경기소비재", "industry": "자동차부품"},
    "145210.KS": {"name": "다이나믹디자인", "sector": "경기소비재", "industry": "자동차부품"},
    "145720.KS": {"name": "덴티움", "sector": "헬스케어", "industry": "건강관리장비와용품"},
    "145990.KS": {"name": "삼양사", "sector": "필수소비재", "industry": "식품"},
    "155660.KS": {"name": "DSR", "sector": "소재", "industry": "비철금속"},
    "161000.KS": {"name": "애경케미칼", "sector": "소재", "industry": "화학"},
    "161390.KS": {"name": "한국타이어앤테크놀로지", "sector": "경기소비재", "industry": "자동차부품"},
    "161890.KS": {"name": "한국콜마", "sector": "필수소비재", "industry": "화장품"},
    "163560.KS": {"name": "동일고무벨트", "sector": "소재", "industry": "화학"},
    "170900.KS": {"name": "동아에스티", "sector": "헬스케어", "industry": "제약"},
    "175330.KS": {"name": "JB금융지주", "sector": "금융", "industry": "은행"},
    "178920.KS": {"name": "PI첨단소재", "sector": "소재", "industry": "화학"},
    "180640.KS": {"name": "한진칼", "sector": "산업재", "industry": "항공사"},
    "181710.KS": {"name": "NHN", "sector": "IT", "industry": "IT서비스"},
    "183190.KS": {"name": "아세아시멘트", "sector": "소재", "industry": "건축자재"},
    "185750.KS": {"name": "종근당", "sector": "헬스케어", "industry": "제약"},
    "192080.KS": {"name": "더블유게임즈", "sector": "IT", "industry": "게임엔터테인먼트"},
    "192400.KS": {"name": "쿠쿠홀딩스", "sector": "경기소비재", "industry": "가정용기기와용품"},
    "192650.KS": {"name": "드림텍", "sector": "통신·미디어", "industry": "핸드셋"},
    "192820.KS": {"name": "코스맥스", "sector": "필수소비재", "industry": "화장품"},
    "194370.KS": {"name": "제이에스코퍼레이션", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "195870.KS": {"name": "해성디에스", "sector": "IT", "industry": "반도체와반도체장비"},
    "200880.KS": {"name": "서연이화", "sector": "경기소비재", "industry": "자동차부품"},
    "204320.KS": {"name": "HL만도", "sector": "경기소비재", "industry": "자동차부품"},
    "207940.KS": {"name": "삼성바이오로직스", "sector": "헬스케어", "industry": "제약"},
    "210540.KS": {"name": "디와이파워", "sector": "산업재", "industry": "기계"},
    "210980.KS": {"name": "SK디앤디", "sector": "부동산", "industry": "부동산"},
    "213500.KS": {"name": "한솔제지", "sector": "소재", "industry": "종이와목재"},
    "214320.KS": {"name": "이노션", "sector": "통신·미디어", "industry": "광고"},
    "214330.KS": {"name": "금호에이치티", "sector": "경기소비재", "industry": "자동차부품"},
    "214390.KS": {"name": "경보제약", "sector": "헬스케어", "industry": "제약"},
    "214420.KS": {"name": "토니모리", "sector": "필수소비재", "industry": "화장품"},
    "217590.KS": {"name": "티엠씨", "sector": "산업재", "industry": "조선"},
    "226320.KS": {"name": "잇츠한불", "sector": "필수소비재", "industry": "화장품"},
    "227840.KS": {"name": "현대코퍼레이션홀딩스", "sector": "필수소비재", "industry": "식품과기본식료품소매"},
    "229640.KS": {"name": "LS에코에너지", "sector": "소재", "industry": "전기장비"},
    "234080.KS": {"name": "JW생명과학", "sector": "헬스케어", "industry": "제약"},
    "241560.KS": {"name": "두산밥캣", "sector": "산업재", "industry": "기계"},
    "241590.KS": {"name": "화승엔터프라이즈", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "244920.KS": {"name": "에이플러스에셋", "sector": "금융", "industry": "손해보험"},
    "248070.KS": {"name": "솔루엠", "sector": "IT", "industry": "전자장비와기기"},
    "248170.KS": {"name": "샘표식품", "sector": "필수소비재", "industry": "식품"},
    "249420.KS": {"name": "일동제약", "sector": "헬스케어", "industry": "제약"},
    "251270.KS": {"name": "넷마블", "sector": "IT", "industry": "게임엔터테인먼트"},
    "259960.KS": {"name": "크래프톤", "sector": "IT", "industry": "게임엔터테인먼트"},
    "264900.KS": {"name": "크라운제과", "sector": "필수소비재", "industry": "식품"},
    "267250.KS": {"name": "HD현대", "sector": "산업재", "industry": "조선"},
    "267260.KS": {"name": "HD현대일렉트릭", "sector": "소재", "industry": "전기장비"},
    "267270.KS": {"name": "HD건설기계", "sector": "산업재", "industry": "기계"},
    "267290.KS": {"name": "경동도시가스", "sector": "유틸리티", "industry": "가스유틸리티"},
    "267850.KS": {"name": "아시아나IDT", "sector": "IT", "industry": "IT서비스"},
    "268280.KS": {"name": "미원에스씨", "sector": "소재", "industry": "화학"},
    "271560.KS": {"name": "오리온", "sector": "필수소비재", "industry": "식품"},
    "271940.KS": {"name": "일진하이솔루스", "sector": "경기소비재", "industry": "자동차부품"},
    "271980.KS": {"name": "제일약품", "sector": "헬스케어", "industry": "제약"},
    "272210.KS": {"name": "한화시스템", "sector": "산업재", "industry": "우주항공과국방"},
    "272450.KS": {"name": "진에어", "sector": "산업재", "industry": "항공사"},
    "272550.KS": {"name": "삼양패키징", "sector": "소재", "industry": "포장재"},
    "278470.KS": {"name": "에이피알", "sector": "필수소비재", "industry": "화장품"},
    "279570.KS": {"name": "케이뱅크", "sector": "금융", "industry": "은행"},
    "280360.KS": {"name": "롯데웰푸드", "sector": "필수소비재", "industry": "식품"},
    "281820.KS": {"name": "케이씨텍", "sector": "IT", "industry": "반도체와반도체장비"},
    "282330.KS": {"name": "BGF리테일", "sector": "경기소비재", "industry": "백화점과일반상점"},
    "284740.KS": {"name": "쿠쿠홈시스", "sector": "경기소비재", "industry": "가정용기기와용품"},
    "285130.KS": {"name": "SK케미칼", "sector": "소재", "industry": "화학"},
    "286940.KS": {"name": "롯데이노베이트", "sector": "IT", "industry": "IT서비스"},
    "293480.KS": {"name": "하나제약", "sector": "헬스케어", "industry": "제약"},
    "294870.KS": {"name": "IPARK현대산업개발", "sector": "산업재", "industry": "건설"},
    "298000.KS": {"name": "효성화학", "sector": "소재", "industry": "화학"},
    "298020.KS": {"name": "효성티앤씨", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "298040.KS": {"name": "효성중공업", "sector": "소재", "industry": "전기장비"},
    "298050.KS": {"name": "HS효성첨단소재", "sector": "소재", "industry": "화학"},
    "298690.KS": {"name": "에어부산", "sector": "산업재", "industry": "항공사"},
    "300720.KS": {"name": "한일시멘트", "sector": "소재", "industry": "건축자재"},
    "302440.KS": {"name": "SK바이오사이언스", "sector": "헬스케어", "industry": "제약"},
    "306200.KS": {"name": "세아제강", "sector": "소재", "industry": "철강"},
    "307950.KS": {"name": "현대오토에버", "sector": "IT", "industry": "IT서비스"},
    "308170.KS": {"name": "씨티알모빌리티", "sector": "경기소비재", "industry": "자동차부품"},
    "316140.KS": {"name": "우리금융지주", "sector": "금융", "industry": "은행"},
    "317400.KS": {"name": "자이에스앤디", "sector": "산업재", "industry": "건설"},
    "317450.KS": {"name": "명인제약", "sector": "헬스케어", "industry": "제약"},
    "322000.KS": {"name": "HD현대에너지솔루션", "sector": "산업재", "industry": "에너지장비및서비스"},
    "323410.KS": {"name": "카카오뱅크", "sector": "금융", "industry": "은행"},
    "326030.KS": {"name": "SK바이오팜", "sector": "헬스케어", "industry": "제약"},
    "329180.KS": {"name": "HD현대중공업", "sector": "산업재", "industry": "조선"},
    "336260.KS": {"name": "두산퓨얼셀", "sector": "IT", "industry": "전기제품"},
    "336370.KS": {"name": "솔루스첨단소재", "sector": "IT", "industry": "전자장비와기기"},
    "339770.KS": {"name": "교촌에프앤비", "sector": "필수소비재", "industry": "식품"},
    "344820.KS": {"name": "KCC글라스", "sector": "소재", "industry": "건축자재"},
    "352820.KS": {"name": "하이브", "sector": "통신·미디어", "industry": "방송과엔터테인먼트"},
    "353200.KS": {"name": "대덕전자", "sector": "IT", "industry": "전자장비와기기"},
    "361610.KS": {"name": "SK아이이테크놀로지", "sector": "IT", "industry": "전기제품"},
    "363280.KS": {"name": "티와이홀딩스", "sector": "경기소비재", "industry": "호텔,레스토랑,레저"},
    "372910.KS": {"name": "한컴라이프케어", "sector": "헬스케어", "industry": "건강관리장비와용품"},
    "373220.KS": {"name": "LG에너지솔루션", "sector": "IT", "industry": "전기제품"},
    "375500.KS": {"name": "DL이앤씨", "sector": "산업재", "industry": "건설"},
    "377300.KS": {"name": "카카오페이", "sector": "IT", "industry": "IT서비스"},
    "377740.KS": {"name": "바이오노트", "sector": "헬스케어", "industry": "건강관리장비와용품"},
    "378850.KS": {"name": "화승알앤에이", "sector": "경기소비재", "industry": "자동차부품"},
    "381970.KS": {"name": "케이카", "sector": "경기소비재", "industry": "자동차"},
    "383220.KS": {"name": "F&F", "sector": "경기소비재", "industry": "섬유,의류,신발,호화품"},
    "383800.KS": {"name": "LX홀딩스", "sector": "산업재", "industry": "무역회사와판매업체"},
    "402340.KS": {"name": "SK스퀘어", "sector": "산업재", "industry": "복합기업"},
    "403550.KS": {"name": "쏘카", "sector": "산업재", "industry": "도로와철도운송"},
    "439260.KS": {"name": "대한조선", "sector": "산업재", "industry": "조선"},
    "443060.KS": {"name": "HD현대마린솔루션", "sector": "산업재", "industry": "조선"},
    "446070.KS": {"name": "유니드비티플러스", "sector": "소재", "industry": "건축자재"},
    "450080.KS": {"name": "에코프로머티", "sector": "IT", "industry": "전기제품"},
    "452260.KS": {"name": "한화갤러리아", "sector": "경기소비재", "industry": "백화점과일반상점"},
    "453340.KS": {"name": "현대그린푸드", "sector": "필수소비재", "industry": "식품"},
    "454910.KS": {"name": "두산로보틱스", "sector": "산업재", "industry": "기계"},
    "456040.KS": {"name": "OCI", "sector": "소재", "industry": "화학"},
    "457190.KS": {"name": "이수스페셜티케미컬", "sector": "소재", "industry": "화학"},
    "460850.KS": {"name": "동국씨엠", "sector": "소재", "industry": "철강"},
    "460860.KS": {"name": "동국제강", "sector": "소재", "industry": "철강"},
    "462520.KS": {"name": "조선내화", "sector": "소재", "industry": "비철금속"},
    "462870.KS": {"name": "시프트업", "sector": "IT", "industry": "게임엔터테인먼트"},
    "465770.KS": {"name": "STX그린로지스", "sector": "산업재", "industry": "해운사"},
    "475150.KS": {"name": "SK이터닉스", "sector": "산업재", "industry": "에너지장비및서비스"},
    "475560.KS": {"name": "더본코리아", "sector": "경기소비재", "industry": "호텔,레스토랑,레저"},
    "480370.KS": {"name": "씨케이솔루션", "sector": "산업재", "industry": "기계"},
    "483650.KS": {"name": "달바글로벌", "sector": "필수소비재", "industry": "화장품"},
    "484870.KS": {"name": "엠앤씨솔루션", "sector": "산업재", "industry": "우주항공과국방"},
    "487570.KS": {"name": "HS효성", "sector": "산업재", "industry": "복합기업"},
    "489790.KS": {"name": "한화비전", "sector": "통신·미디어", "industry": "통신장비"},
    "499790.KS": {"name": "GS피앤엘", "sector": "경기소비재", "industry": "호텔,레스토랑,레저"},
    "900140.KS": {"name": "엘브이엠씨홀딩스", "sector": "경기소비재", "industry": "자동차"},
    "950210.KS": {"name": "프레스티지바이오파마", "sector": "헬스케어", "industry": "제약"},
}



def get_kospi():
    """KOSPI 전 종목 리스트 (보통주 ~807개). yfinance 형식의 .KS 티커.
    Naver Finance 시가총액 페이지 + WICS 업종 분류를 기반으로 하드코딩.
    필요 시 KOSPI_STOCKS 에 추가/수정."""
    _log("  Loading KOSPI list (full hardcoded) ...")
    info = {
        tk: {
            "name": v.get("name", tk),
            "sector": v.get("sector", "기타"),
            "industry": v.get("industry") or v.get("sector", "기타"),
        }
        for tk, v in KOSPI_STOCKS.items()
    }
    _log(f"    {len(info)} KOSPI tickers")
    return list(info.keys()), info


# ═══════════════════════════════════════════════════════════════════
# S&P 500 constituent list
# ═══════════════════════════════════════════════════════════════════
def get_sp500():
    _log("  Fetching S&P 500 list from Wikipedia ...")
    try:
        import requests as _req
        resp = _req.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15, verify=False,
        )
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        df = tables[0]
        df["sym"] = df["Symbol"].str.replace(".", "-", regex=False)
        info = {
            row["sym"]: {
                "name": row.get("Security", ""),
                "sector": row.get("GICS Sector", ""),
                "industry": row.get("GICS Sub-Industry", "") or row.get("GICS Sector", ""),
            }
            for _, row in df.iterrows()
        }
        _log(f"    {len(info)} constituents")
        return list(info.keys()), info
    except Exception as exc:
        _log(f"[!] Failed to fetch S&P 500 list: {exc}")
        return [], {}


# ═══════════════════════════════════════════════════════════════════
# SQLite price cache
# ═══════════════════════════════════════════════════════════════════
def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS prices "
        "(ticker TEXT NOT NULL, date TEXT NOT NULL, close REAL, "
        "PRIMARY KEY (ticker, date))"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_tk ON prices(ticker)")
    conn.commit()
    return conn


# ─── Narrative cache (날짜별 누적, 같은 날짜는 최신본만 유지) ────────────────
def _init_narrative_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS narratives ("
        "ticker TEXT NOT NULL, "
        "market TEXT NOT NULL, "
        "date TEXT NOT NULL, "          # 생성일 (YYYY-MM-DD)
        "generated_at TEXT NOT NULL, "  # 생성 시각 (YYYY-MM-DD HH:MM)
        "name TEXT, sector TEXT, industry TEXT, kind TEXT, "
        "price REAL, "                  # 생성 시점 종가
        "rs_5 REAL, rs_20 REAL, rs_60 REAL, ir_20 REAL, "
        "rough_narrative TEXT, "
        "narrative_md TEXT, "
        "issues_json TEXT, "
        "stage1_error TEXT, "
        "PRIMARY KEY (ticker, market, date))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_narr_tk "
        "ON narratives(ticker, market, date DESC)"
    )
    conn.commit()
    return conn


def _save_narrative_db(narr: dict, market: str):
    """narrative dict 를 DB 에 INSERT OR REPLACE.
    PK 가 (ticker, market, date) 라서 같은 날짜에 다시 생성하면 최신본으로 덮어씀."""
    if not narr or not narr.get("ticker"):
        return
    try:
        gen_at = narr.get("generated_at") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        date_part = (gen_at.split(" ")[0] if " " in gen_at else gen_at).strip()
        rs = narr.get("rs_metrics") or {}
        conn = _init_narrative_db()
        conn.execute(
            "INSERT OR REPLACE INTO narratives "
            "(ticker, market, date, generated_at, name, sector, industry, kind, "
            " price, rs_5, rs_20, rs_60, ir_20, "
            " rough_narrative, narrative_md, issues_json, stage1_error) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                narr.get("ticker"),
                (market or "US").upper(),
                date_part,
                gen_at,
                narr.get("name", ""),
                narr.get("sector", ""),
                narr.get("industry", ""),
                narr.get("kind", ""),
                float(narr["price"]) if narr.get("price") is not None else None,
                rs.get("rs_5"),
                rs.get("rs_20"),
                rs.get("rs_60"),
                rs.get("ir_20"),
                narr.get("rough_narrative", "") or "",
                narr.get("narrative_md", "") or "",
                json.dumps(narr.get("issues", []) or [], ensure_ascii=False),
                narr.get("stage1_error") or "",
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        _log(f"[!] narrative DB save failed: {exc}")


def _row_to_narrative(row) -> dict:
    """sqlite Row → narrative dict (UI/JS 가 기대하는 모양)."""
    (ticker, market, date, generated_at, name, sector, industry, kind,
     price, rs_5, rs_20, rs_60, ir_20,
     rough_narrative, narrative_md, issues_json, stage1_error) = row
    try:
        issues = json.loads(issues_json) if issues_json else []
    except Exception:
        issues = []
    return {
        "ticker": ticker,
        "name": name or ticker,
        "sector": sector or "",
        "industry": industry or sector or "",
        "market": market,
        "kind": kind or "",
        "price": price,
        "date": date,
        "generated_at": generated_at,
        "rs_metrics": {
            "rs_5": rs_5, "rs_20": rs_20, "rs_60": rs_60, "ir_20": ir_20,
        },
        "rough_narrative": rough_narrative or "",
        "narrative_md": narrative_md or "",
        "issues": issues,
        "stage1_error": stage1_error or None,
    }


def _load_narrative_db(ticker: str, market: str, date: str = None):
    """DB 에서 narrative 1건 로드. date=None 이면 가장 최신 일자."""
    try:
        conn = _init_narrative_db()
        cols = ("ticker, market, date, generated_at, name, sector, industry, kind, "
                "price, rs_5, rs_20, rs_60, ir_20, "
                "rough_narrative, narrative_md, issues_json, stage1_error")
        mk = (market or "US").upper()
        if date:
            row = conn.execute(
                f"SELECT {cols} FROM narratives "
                "WHERE ticker=? AND market=? AND date=?",
                (ticker, mk, date),
            ).fetchone()
        else:
            row = conn.execute(
                f"SELECT {cols} FROM narratives "
                "WHERE ticker=? AND market=? "
                "ORDER BY date DESC, generated_at DESC LIMIT 1",
                (ticker, mk),
            ).fetchone()
        conn.close()
        if not row:
            return None
        return _row_to_narrative(row)
    except Exception as exc:
        _log(f"[!] narrative DB load failed: {exc}")
        return None


def _load_narrative_history(ticker: str, market: str):
    """해당 ticker 의 날짜 목록 (최신순) + 각 날짜의 generated_at 메타."""
    try:
        conn = _init_narrative_db()
        mk = (market or "US").upper()
        rows = conn.execute(
            "SELECT date, generated_at FROM narratives "
            "WHERE ticker=? AND market=? ORDER BY date DESC",
            (ticker, mk),
        ).fetchall()
        conn.close()
        return [{"date": r[0], "generated_at": r[1]} for r in rows]
    except Exception as exc:
        _log(f"[!] narrative DB history failed: {exc}")
        return []


def _save_close(conn, ticker, series):
    if series is None or series.empty:
        return
    rows = [
        (ticker, d.strftime("%Y-%m-%d"), float(v))
        for d, v in series.items()
        if pd.notna(v)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO prices(ticker,date,close) VALUES(?,?,?)", rows
    )
    conn.commit()


def _download_one(sym, period=None, start=None):
    try:
        tk = yf.Ticker(sym, session=_YF_SESSION)
        kw = {"auto_adjust": True}
        if start:
            kw["start"] = start
        else:
            kw["period"] = period or INITIAL_PERIOD
        hist = tk.history(**kw)
        if not hist.empty and "Close" in hist.columns:
            return hist["Close"]
    except Exception:
        pass
    return None


def _batch_download_save(conn, syms, period=None, start=None):
    """Download a batch of tickers via yf.download and save to DB. Returns (ok, fail)."""
    ok = fail = 0
    try:
        kw = {"auto_adjust": True, "session": _YF_SESSION, "threads": True, "progress": False}
        if start:
            kw["start"] = start
        else:
            kw["period"] = period or INITIAL_PERIOD
        bulk = yf.download(syms, **kw)
        if bulk.empty:
            return 0, len(syms)
        closes = bulk["Close"] if len(syms) > 1 else bulk[["Close"]].rename(columns={"Close": syms[0]})
        for sym in syms:
            if sym in closes.columns:
                s = closes[sym].dropna()
                if not s.empty:
                    _save_close(conn, sym, s)
                    ok += 1
                else:
                    fail += 1
            else:
                fail += 1
    except Exception:
        fail = len(syms) - ok
    return ok, fail


def sync_prices(tickers, bench: str = BENCH):
    conn = _init_db()
    syms = sorted(set(tickers + [bench]))

    rows = conn.execute(
        "SELECT ticker, MAX(date) FROM prices GROUP BY ticker"
    ).fetchall()
    db_last = {r[0]: r[1] for r in rows}

    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    need_full, need_gap, up_to_date = [], [], []
    for sym in syms:
        if sym not in db_last:
            need_full.append(sym)
        else:
            last = datetime.datetime.strptime(db_last[sym], "%Y-%m-%d").date()
            if last >= yesterday:
                up_to_date.append(sym)
            else:
                need_gap.append((sym, last))

    _log(
        f"  Price sync: {len(up_to_date)} current, "
        f"{len(need_gap)} update, {len(need_full)} new"
    )

    total_work = len(need_full) + len(need_gap)

    if need_full:
        _log(f"    Downloading {len(need_full)} new tickers in batches of {BATCH_SIZE} ...")
        ok_all = fail_all = 0
        n_batches = (len(need_full) - 1) // BATCH_SIZE + 1
        for bi in range(n_batches):
            batch = need_full[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
            G_STATUS["detail"] = f"Batch {bi+1}/{n_batches} ({len(batch)} tickers)"
            G_STATUS["progress"] = 5 + int(50 * (bi + 1) / n_batches)
            ok, fail = _batch_download_save(conn, batch, period=INITIAL_PERIOD)
            ok_all += ok
            fail_all += fail
            _log(f"      Batch {bi+1}/{n_batches}: {ok} ok, {fail} fail")
            if bi < n_batches - 1:
                time.sleep(1.5)
        _log(f"    Full download done: {ok_all} ok, {fail_all} fail")

    if need_gap:
        _log(f"    Updating {len(need_gap)} tickers ...")
        G_STATUS["detail"] = f"{len(need_gap)} tickers batch update"
        G_STATUS["progress"] = 55
        earliest = min(last for _, last in need_gap)
        gap_start = (earliest + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        gap_syms = [sym for sym, _ in need_gap]
        ok, fail = _batch_download_save(conn, gap_syms, start=gap_start)
        _log(f"      Batch update: {ok}/{len(gap_syms)} ok")

    if total_work == 0:
        G_STATUS["detail"] = "All prices up to date"
        G_STATUS["progress"] = 58

    _log("    Loading from DB ...")
    ph = ",".join(["?"] * len(syms))
    df = pd.read_sql(
        f"SELECT ticker, date, close FROM prices WHERE ticker IN ({ph})",
        conn, params=syms,
    )
    conn.close()

    if df.empty:
        raise RuntimeError("No price data in DB")

    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="ticker", values="close").sort_index()
    good = close.columns[close.count() >= MIN_BARS]
    close = close[good]
    _log(f"    {close.shape[0]} days x {close.shape[1]} symbols ready")
    return close


# ═══════════════════════════════════════════════════════════════════
# Core screening logic
# ═══════════════════════════════════════════════════════════════════
def screen(close: pd.DataFrame, beta_adj: bool = BETA_ADJ, label: str = "",
           bench: str = BENCH) -> pd.DataFrame:
    if bench not in close.columns:
        raise RuntimeError(f"Benchmark '{bench}' not found")

    pb = close[bench]
    cols = [c for c in close.columns if c != bench]
    p = close[cols]

    if LOG_RET:
        r = np.log(p / p.shift(1))
        rb = np.log(pb / pb.shift(1))
    else:
        r = p.pct_change()
        rb = pb.pct_change()

    mr = r.rolling(W_BETA).mean()
    mrb = rb.rolling(W_BETA).mean()
    cov = (r.mul(rb, axis=0)).rolling(W_BETA).mean() - mr.mul(mrb, axis=0)
    vb = (rb**2).rolling(W_BETA).mean() - mrb**2
    beta = cov.div(vb, axis=0).replace([np.inf, -np.inf], np.nan)

    e_raw = r.sub(rb, axis=0)
    if beta_adj:
        e = (r - beta.mul(rb, axis=0)).where(beta.notna(), e_raw)
    else:
        e = e_raw

    RS = {w: e.rolling(w).sum() for w in (W_FILTER, W_JUDGE, W_SLOW, W_FAST)}

    def _ir(series, w):
        return (series.rolling(w).mean() / series.rolling(w).std()) * np.sqrt(ANN)

    ir20 = _ir(e, W_JUDGE)
    ir60 = _ir(e, W_FILTER)

    mr60 = r.rolling(W_FILTER).mean()
    mrb60 = rb.rolling(W_FILTER).mean()
    alpha_d = mr60 - beta.mul(mrb60, axis=0)
    alpha_a = alpha_d * ANN

    eps = r - (alpha_d + beta.mul(rb, axis=0))
    alpha_ir = _ir(eps, W_FILTER)

    L = np.log(p).sub(np.log(pb), axis=0)
    M = L - L.shift(K_MOM)
    A = M - M.shift(K_MOM)

    _pct = (lambda x: (np.exp(x) - 1) * 100) if LOG_RET else (lambda x: x * 100)

    o = pd.DataFrame(index=cols)
    for tag, w in [("60", W_FILTER), ("20", W_JUDGE), ("10", W_SLOW), ("5", W_FAST)]:
        o[f"rs_{tag}_raw"] = RS[w].iloc[-1]
        o[f"rs_{tag}"] = _pct(RS[w].iloc[-1])

    o["ir_60"] = ir60.iloc[-1]
    o["ir_20"] = ir20.iloc[-1]
    o["beta"] = beta.iloc[-1]
    o["alpha_ann"] = alpha_a.iloc[-1] * 100
    o["alpha_ir"] = alpha_ir.iloc[-1]
    o["mom"] = M.iloc[-1]
    o["acc"] = A.iloc[-1]

    o["filter_pass"] = o["rs_60_raw"] > 0
    o["judge_pass"] = o["rs_20_raw"] > 0
    o["early_trend"] = (o["mom"] > 0) & (o["acc"] > 0)
    o["consistency_pass"] = o["ir_20"] > 0
    o["leader"] = (
        o["filter_pass"] & o["judge_pass"] & o["consistency_pass"]
    )

    o["status"] = np.select(
        [
            o["leader"],
            o["early_trend"] & o["judge_pass"],
        ],
        ["LEADER", "Early Trend"],
        "Not Leading",
    )

    o = o.dropna(subset=["rs_60", "rs_20"])

    rs20_pct = o["rs_20"].rank(pct=True, na_option="bottom")
    ir20_pct = o["ir_20"].rank(pct=True, na_option="bottom")
    o["composite"] = 0.5 * rs20_pct + 0.5 * ir20_pct

    rank = {"LEADER": 0, "Early Trend": 1, "Not Leading": 2}
    o["_r"] = o["status"].map(rank)
    o = o.sort_values(["_r", "composite"], ascending=[True, False]).drop(columns=["_r"])
    o.index.name = "ticker"

    cnt = o["status"].value_counts()
    tag = f" [{label}]" if label else ""
    _log(
        f"    Screen{tag}: {int(cnt.get('LEADER', 0))} leaders, "
        f"{int(cnt.get('Early Trend', 0))} early, "
        f"{int(cnt.get('Not Leading', 0))} not leading"
    )
    return o


# ═══════════════════════════════════════════════════════════════════
# 52-Week High / Low Detection
# ═══════════════════════════════════════════════════════════════════
def detect_52w(close: pd.DataFrame, label: str = "", bench: str = BENCH) -> dict:
    cols = [c for c in close.columns if c != bench]
    result = {}
    for tk in cols:
        series = close[tk].dropna()
        if len(series) < W_52WEEK:
            result[tk] = {"status": None, "pct_from_high": None,
                          "pct_from_low": None, "high": None,
                          "low": None, "current": None}
            continue
        window = series.iloc[-W_52WEEK:]
        current = float(series.iloc[-1])
        high_52w = float(window.max())
        low_52w = float(window.min())
        pct_from_high = (current / high_52w - 1) * 100
        pct_from_low = (current / low_52w - 1) * 100
        if current >= high_52w:
            status = "52W_HIGH"
        elif current <= low_52w:
            status = "52W_LOW"
        else:
            status = None
        result[tk] = {
            "status": status,
            "pct_from_high": round(pct_from_high, 2),
            "pct_from_low": round(pct_from_low, 2),
            "high": round(high_52w, 2),
            "low": round(low_52w, 2),
            "current": round(current, 2),
        }
    highs = [tk for tk, v in result.items() if v["status"] == "52W_HIGH"]
    lows = [tk for tk, v in result.items() if v["status"] == "52W_LOW"]
    tag = f" [{label}]" if label else ""
    _log(f"    52W{tag}: {len(highs)} highs, {len(lows)} lows")
    return result


# ═══════════════════════════════════════════════════════════════════
# Market Breadth
# ═══════════════════════════════════════════════════════════════════
def calc_breadth(close, sp500_results, sp500_info, sp500_w52, bench: str = BENCH):
    _log("    Calculating market breadth ...")
    tickers = [c for c in sp500_results.index if c in close.columns and c != bench]
    p = close[tickers]
    total = len(tickers)

    sma50 = p.rolling(50).mean()
    above_50 = int((p.iloc[-1] > sma50.iloc[-1]).sum())
    sma200 = p.rolling(200).mean()
    above_200 = int((p.iloc[-1] > sma200.iloc[-1]).sum())

    advance = int((sp500_results["rs_20_raw"] > 0).sum())
    decline = int((sp500_results["rs_20_raw"] < 0).sum())
    ad_ratio = round(advance / max(decline, 1), 2)

    cnt = sp500_results["status"].value_counts()
    n_ld = int(cnt.get("LEADER", 0))
    n_et = int(cnt.get("Early Trend", 0))
    n_nl = int(cnt.get("Not Leading", 0))

    n_52h = sum(1 for v in sp500_w52.values() if v.get("status") == "52W_HIGH")
    n_52l = sum(1 for v in sp500_w52.values() if v.get("status") == "52W_LOW")

    leaders = sp500_results[sp500_results["status"] == "LEADER"]
    leader_sectors = {}
    leader_industries = {}
    for tk in leaders.index:
        meta = sp500_info.get(tk, {})
        sec = meta.get("sector", "Unknown") or "Unknown"
        ind = meta.get("industry", "Unknown") or "Unknown"
        leader_sectors[sec] = leader_sectors.get(sec, 0) + 1
        leader_industries[ind] = leader_industries.get(ind, 0) + 1
    leader_sectors = dict(sorted(leader_sectors.items(), key=lambda x: -x[1]))
    leader_industries = dict(sorted(leader_industries.items(), key=lambda x: -x[1]))

    _log(f"    Breadth: %50MA={above_50}/{total}, %200MA={above_200}/{total}, A/D={ad_ratio}")
    return {
        "total": total,
        "pct_above_50ma": round(above_50 / max(total, 1) * 100, 1),
        "pct_above_200ma": round(above_200 / max(total, 1) * 100, 1),
        "advance": advance, "decline": decline, "ad_ratio": ad_ratio,
        "n_52h": n_52h, "n_52l": n_52l,
        "n_leader": n_ld, "n_early": n_et, "n_not_leading": n_nl,
        "leader_sectors": leader_sectors,
        "leader_industries": leader_industries,
    }


# ═══════════════════════════════════════════════════════════════════
# Gemini AI Analysis
# ═══════════════════════════════════════════════════════════════════
def _calc_return(series, days):
    if series is None or len(series) < days + 1:
        return None
    try:
        return (float(series.iloc[-1]) / float(series.iloc[-(days + 1)]) - 1) * 100
    except (ZeroDivisionError, IndexError):
        return None


def gemini_analyze(sector_results, sector_info, sector_w52, close=None,
                   sp500_results=None, sp500_info=None, breadth=None,
                   market: str = "US"):
    api_key = _API_KEYS.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return ("GEMINI_API_KEY 환경변수를 설정하면 AI 분석을 이용할 수 있습니다.\n\n"
                "Windows: `set GEMINI_API_KEY=your_key`  /  Linux: `export GEMINI_API_KEY=your_key`")
    if not HAS_GENAI:
        return "`pip install google-generativeai` 패키지를 설치해주세요."

    mkt = MARKETS.get(market, MARKETS["US"])
    bench = mkt["bench"]
    bench_name = mkt["bench_name"]
    constituents_label = mkt["constituents_label"]
    currency = mkt["currency"]

    _log(f"  Generating Gemini AI analysis ({market}) ...")

    # --- Benchmark block ---
    bench_block = ""
    if close is not None and bench in close.columns:
        b = close[bench].dropna()
        r5 = _calc_return(b, 5)
        r20 = _calc_return(b, 20)
        r60 = _calc_return(b, 60)
        r120 = _calc_return(b, 120)
        bench_block = (
            f"[{bench_name} Benchmark]\n"
            f"Price: {currency}{float(b.iloc[-1]):,.2f} | "
            f"5d: {r5:+.2f}% | 20d: {r20:+.2f}% | 60d: {r60:+.2f}% | 120d: {r120:+.2f}%\n"
        )

    # --- Breadth block ---
    breadth_block = ""
    if breadth:
        breadth_block = (
            f"\n[{constituents_label} Market Breadth]\n"
            f"Screened: {breadth['total']} | Leaders: {breadth['n_leader']} | "
            f"Early Trend: {breadth['n_early']} | "
            f"Not Leading: {breadth['n_not_leading']}\n"
            f"% Above 50-day MA: {breadth['pct_above_50ma']}% | "
            f"% Above 200-day MA: {breadth['pct_above_200ma']}%\n"
            f"Advance/Decline: {breadth['advance']}/{breadth['decline']} "
            f"(ratio: {breadth['ad_ratio']})\n"
            f"52W New Highs: {breadth['n_52h']} | 52W New Lows: {breadth['n_52l']}\n"
        )
        if breadth.get("leader_sectors"):
            breadth_block += "\nLeader Sector Distribution:\n"
            for sec, cnt in breadth["leader_sectors"].items():
                breadth_block += f"  {sec}: {cnt}\n"
        if breadth.get("leader_industries"):
            breadth_block += "\nLeader Industry (Sub-Industry) Distribution (Top 20):\n"
            for ind, cnt in list(breadth["leader_industries"].items())[:20]:
                breadth_block += f"  {ind}: {cnt}\n"

    # --- Constituent Leaders list ---
    leaders_block = ""
    if sp500_results is not None and sp500_info:
        ld = sp500_results[sp500_results["status"] == "LEADER"].head(50)
        if not ld.empty:
            leaders_block = f"\n[{constituents_label} Top Leaders (up to 50)]\n"
            leaders_block += "Ticker | Company | Sector | Industry | RS20d% | IR20d | Alpha%\n"
            for tk, row in ld.iterrows():
                inf = sp500_info.get(tk, {})
                leaders_block += (
                    f"{tk} | {inf.get('name','')} | {inf.get('sector','')} | "
                    f"{inf.get('industry','')} | "
                    f"{row.get('rs_20',0):.2f} | {row.get('ir_20',0):.2f} | "
                    f"{row.get('alpha_ann',0):.2f}\n"
                )

    # --- Sector ETF table ---
    sec_lines = ["Ticker | ETF | Category | RS60d% | RS20d% | RS5d% | IR20d | Alpha% | Status | 52W"]
    for tk, row in sector_results.iterrows():
        inf = sector_info.get(tk, {})
        w = sector_w52.get(tk, {}).get("status") or "-"
        sec_lines.append(
            f"{tk} | {inf.get('name','')} | {inf.get('sector','')} | "
            f"{row.get('rs_60',0):.2f} | {row.get('rs_20',0):.2f} | "
            f"{row.get('rs_5',0):.2f} | {row.get('ir_20',0):.2f} | "
            f"{row.get('alpha_ann',0):.2f} | {row['status']} | {w}"
        )
    sector_table = "\n".join(sec_lines)

    s_highs = [f"{tk} ({sector_info.get(tk,{}).get('name','')})"
               for tk, v in sector_w52.items() if v.get("status") == "52W_HIGH"]
    s_lows = [f"{tk} ({sector_info.get(tk,{}).get('name','')})"
              for tk, v in sector_w52.items() if v.get("status") == "52W_LOW"]

    market_intro = (
        "S&P 500 전 종목 Breadth 데이터, 40개 섹터/테마 ETF 분석, 주도주 목록"
        if market == "US" else
        "KOSPI 주요 종목 Breadth, 한국 시장 섹터/테마 ETF 분석, KOSPI 주도주 목록"
    )
    prompt = f"""당신은 전문 시장 전략가이자 섹터 로테이션 분석가입니다.
아래는 {market_intro}입니다.

**지표 설명:**
- RS(Relative Strength): {bench_name} 대비 초과수익률 누적(%), IR: 초과수익 일관성(연환산)
- Alpha%: 베타 조정 후 연환산 초과수익률
- Status: LEADER(모든 RS+IR 양수) / Early Trend(모멘텀+가속 양수) / Not Leading

{bench_block}{breadth_block}{leaders_block}
[섹터 ETF 데이터]
{sector_table}

섹터 52W 신고가: {', '.join(s_highs) if s_highs else '없음'}
섹터 52W 신저가: {', '.join(s_lows) if s_lows else '없음'}

아래를 체계적으로 분석해주세요:

## Part 1: Market Regime 진단
1. **시장 건강도**: Breadth 지표(%Above50MA, %Above200MA, A/D ratio)로 본 시장 참여 폭과 건강 상태
2. **Risk-On vs Risk-Off**: 공격적 섹터(반도체/기술/소비재) vs 방어적 섹터(유틸리티/부동산/금) 비교
3. **경기 사이클**: 섹터 로테이션 + Breadth 패턴으로 추정하는 현재 국면(초기회복/확장/후기확장/수축)
4. **시장 폭 평가**: 상승이 소수 종목에 집중되어 있는지, 광범위한지 (Leader 수, 섹터 분포 기반)

## Part 2: 주도주 분석
5. **주도주 특성**: {constituents_label} Leader들의 섹터 집중도, 공통 특성 (성장주/가치주/경기순환주)
6. **주도 섹터 ETF**: LEADER 상태 섹터들의 공통 테마와 의미
7. **부상 / 약세 섹터**: Early Trend, Not Leading 섹터 해석

## Part 3: 테마별 흐름 & 전략
8. **테마별 분석**: 기술/에너지/귀금속/방산/바이오/클린에너지/금융 큰 흐름
9. **자금 흐름**: 어디에서 어디로 자금이 이동하는지
10. **52주 신고가/신저가 코멘트**
11. **종합 Regime 판단 및 전략**: Bull/Bear/Transition 판단, 포지셔닝 시사점

한국어로 명확하고 간결하게 마크다운 형식으로 작성해주세요."""

    try:
        text, model_used, fellback, original = _pro_with_flash_fallback(prompt, api_key)
        if fellback:
            _log("    AI analysis complete (Flash fallback)")
            banner = (
                f"> ⚠️ **{GEMINI_MODEL} 일일 quota 초과로 {GEMINI_FLASH_MODEL} 모델로 폴백되었습니다.**  \n"
                f"> 분석 품질이 다소 낮을 수 있습니다. 24시간 후 quota 회복 시 'Refresh' 버튼으로 재생성 가능.\n\n"
            )
            return banner + text
        _log("    AI analysis complete")
        return text
    except Exception as exc:
        _log(f"[!] Gemini API error: {exc}")
        return f"AI 분석 중 오류 발생:\n\n```\n{exc}\n```"


# ═══════════════════════════════════════════════════════════════════
# Narrative Curator Pipeline
#   Stage 1: Gemini 3 Flash Preview + Google Search → 최근 N일 이슈 수집
#   Stage 2: Gemini 3.1 Pro → 도메인 비전공자용 해설 + Catalyst
# ═══════════════════════════════════════════════════════════════════
def _get_gemini_key():
    return _API_KEYS.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")


def _response_grounding_summary(resp) -> dict:
    """새 SDK 응답에서 grounding metadata 를 추출. 검색이 진짜 일어났는지 판단할 때 사용."""
    info = {"used": False, "queries": [], "sources": 0}
    try:
        cand = resp.candidates[0]
        gm = getattr(cand, "grounding_metadata", None)
        if gm is None:
            return info
        queries = getattr(gm, "web_search_queries", None) or []
        chunks = getattr(gm, "grounding_chunks", None) or []
        if queries or chunks or getattr(gm, "search_entry_point", None):
            info["used"] = True
            info["queries"] = list(queries)
            info["sources"] = len(chunks)
    except (IndexError, AttributeError):
        pass
    return info


def _flash_with_search(prompt: str, api_key: str) -> str:
    """Gemini 3 Flash Preview 호출 + Google Search grounding.
    1순위: 새 SDK (google-genai) 의 표준 GoogleSearch 도구 사용
    2순위: 새 SDK 가 없으면 레거시 SDK 로 plain 호출 (검색 없음, 환각 위험)
    grounding 이 실제로 작동했는지 응답 메타로 검증해서 로그로 남김."""

    if HAS_NEW_GENAI:
        try:
            client = new_genai.Client(api_key=api_key)
            cfg = new_genai_types.GenerateContentConfig(
                tools=[new_genai_types.Tool(
                    google_search=new_genai_types.GoogleSearch()
                )]
            )
            resp = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=prompt,
                config=cfg,
            )
            g = _response_grounding_summary(resp)
            if g["used"]:
                qprev = ", ".join(g["queries"][:3])
                _log(f"    [grounding OK] queries={len(g['queries'])} sources={g['sources']}"
                     + (f" — '{qprev}'" if qprev else ""))
            else:
                _log("    [grounding WARN] response has no grounding metadata "
                     "(model may have answered from training data)")
            return resp.text or ""
        except Exception as e:
            _log(f"    [grounding ERROR] new SDK with GoogleSearch failed: {e}")
            # 새 SDK 자체가 실패하면 레거시로 폴백 (검색 없이라도 답은 받음)

    # 폴백: 레거시 SDK plain 호출 (검색 없음)
    _log("    [grounding FALLBACK] using legacy SDK without search — output may rely on stale training data")
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(GEMINI_FLASH_MODEL)
    return m.generate_content(prompt).text or ""


def _is_quota_error(exc: Exception) -> bool:
    """429 / quota / rate-limit / exceeded 류 에러 감지."""
    s = str(exc).lower()
    return any(k in s for k in ("429", "quota", "rate limit", "rate-limit",
                                "exceeded", "resource exhausted"))


def _pro_with_flash_fallback(prompt: str, api_key: str):
    """Pro 모델 우선 호출. quota/429 시 Flash 모델로 자동 폴백.
    반환: (text, model_used, fellback_bool, original_error_str_or_None)."""
    genai.configure(api_key=api_key)
    try:
        m = genai.GenerativeModel(GEMINI_MODEL)
        out = m.generate_content(prompt).text or ""
        return out, GEMINI_MODEL, False, None
    except Exception as exc:
        if not _is_quota_error(exc):
            raise
        original = str(exc)
        _log(f"[!] Pro quota exceeded, falling back to Flash: {original[:200]}")
        try:
            m = genai.GenerativeModel(GEMINI_FLASH_MODEL)
            out = m.generate_content(prompt).text or ""
            return out, GEMINI_FLASH_MODEL, True, original
        except Exception as exc2:
            raise exc2


def _generate_with_search_and_fallback(prompt: str, api_key: str,
                                       label: str = "narrative"):
    """Pro 모델 + Google Search grounding 우선. quota 시 Flash + grounding 으로 폴백.
    Stage 2 큐레이터처럼 모델 본인이 직접 원문을 검색해서 보강해야 하는 호출용.
    반환: (text, model_used, fellback_bool, original_error_or_None, grounding_summary_dict).
    grounding_summary_dict = {used: bool, queries: [...], sources: int, source_urls: [...]}"""
    if not HAS_NEW_GENAI:
        # 새 SDK 없으면 grounding 불가 → 레거시 plain 호출로 폴백
        _log(f"    [{label}] new SDK unavailable, falling back to plain Pro/Flash without search")
        text, model_used, fellback, original = _pro_with_flash_fallback(prompt, api_key)
        return text, model_used, fellback, original, {"used": False, "queries": [], "sources": 0, "source_urls": []}

    client = new_genai.Client(api_key=api_key)
    cfg = new_genai_types.GenerateContentConfig(
        tools=[new_genai_types.Tool(google_search=new_genai_types.GoogleSearch())]
    )

    def _summarize_grounding(resp) -> dict:
        info = {"used": False, "queries": [], "sources": 0, "source_urls": []}
        try:
            cand = resp.candidates[0]
            gm = getattr(cand, "grounding_metadata", None)
            if gm is None:
                return info
            queries = list(getattr(gm, "web_search_queries", None) or [])
            chunks = list(getattr(gm, "grounding_chunks", None) or [])
            if queries or chunks or getattr(gm, "search_entry_point", None):
                info["used"] = True
                info["queries"] = queries
                info["sources"] = len(chunks)
                for c in chunks:
                    web = getattr(c, "web", None)
                    if web is None:
                        continue
                    title = getattr(web, "title", "") or ""
                    uri = getattr(web, "uri", "") or ""
                    if uri:
                        info["source_urls"].append({"title": title, "url": uri})
        except (IndexError, AttributeError):
            pass
        return info

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt, config=cfg,
        )
        g = _summarize_grounding(resp)
        _log(f"    [{label}] {GEMINI_MODEL} grounding "
             f"{'OK' if g['used'] else 'WARN(no metadata)'}: "
             f"queries={len(g['queries'])} sources={g['sources']}")
        return resp.text or "", GEMINI_MODEL, False, None, g
    except Exception as exc:
        if not _is_quota_error(exc):
            raise
        original = str(exc)
        _log(f"    [{label}] Pro quota exceeded, falling back to Flash + search: {original[:160]}")
        try:
            resp = client.models.generate_content(
                model=GEMINI_FLASH_MODEL, contents=prompt, config=cfg,
            )
            g = _summarize_grounding(resp)
            _log(f"    [{label}] {GEMINI_FLASH_MODEL} grounding "
                 f"{'OK' if g['used'] else 'WARN(no metadata)'}: "
                 f"queries={len(g['queries'])} sources={g['sources']}")
            return resp.text or "", GEMINI_FLASH_MODEL, True, original, g
        except Exception as exc2:
            raise exc2


def _extract_json(text: str):
    """LLM 응답에서 JSON 블록 추출. 코드펜스 제거 후 파싱."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


# ═══════════════════════════════════════════════════════════════════
# Naver News API — KR 종목 뉴스 수집
# ═══════════════════════════════════════════════════════════════════
_NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"


def _naver_search_news(query: str, display: int = 30, sort: str = "date") -> list:
    """네이버 뉴스 검색 API. 키 미설정 시 빈 리스트.
    sort: 'date'(최신순) | 'sim'(정확도)."""
    if not (NAVER_CLIENT_ID and NAVER_CLIENT_SECRET):
        _log("    [Naver] CLIENT_ID/SECRET 미설정 — 검색 건너뜀")
        return []
    try:
        import requests as _rq
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        }
        params = {"query": query, "display": min(int(display), 100),
                  "start": 1, "sort": sort}
        resp = _rq.get(_NAVER_API_URL, headers=headers, params=params,
                       timeout=10, verify=False)
        if resp.status_code != 200:
            _log(f"    [Naver] HTTP {resp.status_code} for '{query}': {resp.text[:160]}")
            return []
        items = (resp.json() or {}).get("items", []) or []
        out = []
        for it in items:
            title = re.sub(r"<[^>]+>", "", it.get("title", "") or "").strip()
            desc = re.sub(r"<[^>]+>", "", it.get("description", "") or "").strip()
            url = it.get("originallink") or it.get("link", "")
            pub = it.get("pubDate", "")
            if not title:
                continue
            out.append({
                "title": title,
                "summary": desc[:500],
                "url": url,
                "published": pub,
            })
        return out
    except Exception as e:
        _log(f"    [Naver] error '{query}': {e}")
        return []


def _parse_naver_pubdate(s: str):
    """네이버 pubDate (RFC 2822 형식) → date 문자열 'YYYY-MM-DD'."""
    if not s:
        return ""
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(s)
        if dt is None:
            return ""
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def find_ticker_issues_naver(ticker: str, name: str, sector: str,
                             lookback_days: int = NARRATIVE_LOOKBACK_DAYS,
                             industry: str = "") -> dict:
    """KR Stage 1: 네이버 뉴스 검색 → 최근 lookback_days일 기사 수집 →
    Gemini Flash 가 이를 구조화된 issue JSON 으로 정리.
    Naver 가 1차 출처이고 Gemini 는 정리만 수행 (검색 도구 사용 X)."""
    api_key = _get_gemini_key()
    if not api_key:
        return {"error": "no_api_key", "issues": [], "rough_narrative": ""}
    if not (NAVER_CLIENT_ID and NAVER_CLIENT_SECRET):
        return {"error": "no_naver_key", "issues": [], "rough_narrative": ""}

    # 종목 코드만 따로 (예: '005930.KS' -> '005930'). 검색에는 종목명을 우선.
    code = ticker.split(".")[0] if "." in ticker else ticker

    # 네이버 검색은 한국어 종목명·산업명·핵심 키워드 조합이 가장 유효
    base_queries = [name, f"{name} 실적", f"{name} 공시", f"{name} 호재",
                    f"{name} 악재"]
    if industry and industry not in name:
        base_queries.append(f"{name} {industry}")
    if sector and sector not in name and sector not in (industry or ""):
        base_queries.append(f"{sector} 업황")

    today = datetime.date.today()
    cutoff = today - datetime.timedelta(days=lookback_days)

    seen = set()
    pool = []
    for q in base_queries:
        for it in _naver_search_news(q, display=30, sort="date"):
            url = it.get("url", "")
            key = url or it.get("title", "")
            if key in seen:
                continue
            seen.add(key)
            pub_str = _parse_naver_pubdate(it.get("published", ""))
            if pub_str:
                try:
                    pub_d = datetime.datetime.strptime(pub_str, "%Y-%m-%d").date()
                    if pub_d < cutoff:
                        continue
                    it["pub_date"] = pub_str
                except ValueError:
                    it["pub_date"] = pub_str
            else:
                it["pub_date"] = ""
            pool.append(it)
        time.sleep(0.1)
    pool.sort(key=lambda x: x.get("pub_date", ""), reverse=True)
    pool = pool[:60]
    _log(f"    [Naver] {ticker} ({name}): {len(pool)}건 수집")

    if not pool:
        return {
            "issues": [],
            "rough_narrative": f"최근 {lookback_days}일 네이버 뉴스 검색에서 "
                               f"{name} 관련 의미 있는 보도가 발견되지 않았습니다.",
            "raw_articles": [],
        }

    articles_block = "\n".join(
        f"- [{a.get('pub_date','?')}] {a.get('title','')} — "
        f"{(a.get('summary','') or '').strip()[:280]}"
        + (f" [src: {a.get('url')}]" if a.get('url') else "")
        for a in pool
    )

    prompt = f"""당신은 한국 주식시장 뉴스 큐레이션 전문가입니다.
아래는 네이버 뉴스에서 검색한 {name}({code}, KOSPI) 관련 최근 {lookback_days}일 기사 헤드라인+요약입니다.
이를 분석해서 **종목의 가격·내러티브에 영향을 미칠 만한 material event** 만 추려 JSON으로 정리하세요.

[종목]
- Ticker: {ticker}  (종목코드: {code})
- 회사명: {name}
- 섹터: {sector}
- 산업: {industry or sector}
- 오늘: {today.strftime('%Y-%m-%d')}
- Lookback: 최근 {lookback_days}일

[수집된 기사 ({len(pool)}건)]
{articles_block}

[작업]
1. 위 기사 중 종목 가격에 실제 영향을 줄 만한 사건만 골라낸다 (실적/가이던스, 신제품·수주, 파트너십, M&A,
   규제·소송·공시, 애널리스트 의견, 업황·매크로, 경영진 변화, 공급망, 지정학 등).
2. 단순 시세 보도, 매수 추천 광고, 무관한 기업 동음이의어 기사는 제외한다.
3. 동일 사건이 여러 매체에서 보도된 경우 1건으로 묶고 가장 유력한 출처를 기재.
4. 사실로 확인 가능한 것만 포함. 추측·뇌피셜 금지.

[출력 — STRICT JSON만, 코드펜스 금지]
{{
  "issues": [
    {{
      "date": "YYYY-MM-DD",
      "category": "earnings|guidance|product|partnership|m&a|regulatory|analyst|macro|management|supply|geopolitics|other",
      "headline": "<짧은 한 줄 제목, 한국어>",
      "summary": "<2-3문장 사실 요약, 한국어>",
      "impact": "positive|negative|mixed|neutral",
      "source": "<언론사 또는 도메인>"
    }}
  ],
  "rough_narrative": "<3-4줄 한국어로 지금 이 종목을 둘러싼 핵심 스토리/내러티브>"
}}

규칙:
- 유효한 issue 가 0건이면 issues=[] 와 rough_narrative 에 그 사실 명시.
- date 미상은 'YYYY-MM-??' 형식 허용. 가능한 한 실제 일자 기재.
- 최신 일자 → 과거 순으로 정렬.
- JSON 외 텍스트 금지.
"""
    try:
        # Naver 가 출처라서 Google Search grounding 불필요. 단순 호출로 충분.
        if HAS_NEW_GENAI:
            client = new_genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=GEMINI_FLASH_MODEL, contents=prompt,
            )
            raw = resp.text or ""
        else:
            genai.configure(api_key=api_key)
            m = genai.GenerativeModel(GEMINI_FLASH_MODEL)
            raw = m.generate_content(prompt).text or ""
        data = _extract_json(raw) or {}
        return {
            "issues": data.get("issues", []) or [],
            "rough_narrative": data.get("rough_narrative", "") or "",
            "raw": raw,
            "raw_articles": pool[:30],
        }
    except Exception as exc:
        _log(f"[!] Naver Stage 1 (Flash 정리) 실패 ({ticker}): {exc}")
        err = str(exc)
        if _is_quota_error(exc):
            err = f"[QUOTA EXCEEDED] {err}"
        return {"error": err, "issues": [], "rough_narrative": "",
                "raw_articles": pool[:30]}


def find_ticker_issues_flash(ticker: str, name: str, sector: str,
                             lookback_days: int = NARRATIVE_LOOKBACK_DAYS,
                             industry: str = "") -> dict:
    """Stage 1: 단일 종목의 최근 lookback_days일 이슈를 Flash + Search로 수집."""
    api_key = _get_gemini_key()
    if not api_key:
        return {"error": "no_api_key", "issues": [], "summary": ""}

    today = datetime.date.today().strftime("%Y-%m-%d")
    prompt = f"""You are a financial news researcher. Use Google Search to find recent material events for the following stock.
Strictly prefer authoritative sources (gov agencies, SEC filings/IR, IMF/OECD/World Bank, Reuters, Bloomberg, FT, Fortune, Forbes, major investment bank reports). Avoid blogs, opinion sites, and unverified sources.

[Target]
Ticker: {ticker}
Company: {name}
Sector: {sector}
Industry: {industry or sector}
Today: {today}
Lookback window: last {lookback_days} calendar days

[Task]
Find every newsworthy event in the lookback window that could affect the stock's price action or narrative.
Cover: earnings/guidance, product launches, partnerships, M&A, regulatory/legal, analyst rating changes,
macro/sector news, management changes, supply chain, geopolitics — anything material.

[Output]
Return STRICT JSON only (no prose, no code fences). Schema:
{{
  "issues": [
    {{
      "date": "YYYY-MM-DD",
      "category": "earnings|guidance|product|partnership|m&a|regulatory|analyst|macro|management|supply|geopolitics|other",
      "headline": "<short 1-line title>",
      "summary": "<2-3 sentence factual summary>",
      "impact": "positive|negative|mixed|neutral",
      "source": "<publisher or domain if known>"
    }}
  ],
  "rough_narrative": "<3-4 line plain-language narrative explaining what story is driving this stock right now>"
}}

Rules:
- Only include events you can actually source. If unsure, omit.
- Order issues by date (most recent first).
- If nothing material is found, return {{"issues": [], "rough_narrative": "No material events found in lookback window."}}
- Output JSON only.
"""
    try:
        raw = _flash_with_search(prompt, api_key)
        data = _extract_json(raw) or {}
        return {
            "issues": data.get("issues", []) or [],
            "rough_narrative": data.get("rough_narrative", "") or "",
            "raw": raw,
        }
    except Exception as exc:
        _log(f"[!] Flash issue-finding failed for {ticker}: {exc}")
        err = str(exc)
        if _is_quota_error(exc):
            err = f"[QUOTA EXCEEDED] {err}"
        return {"error": err, "issues": [], "rough_narrative": ""}


def curate_ticker_narrative_pro(ticker: str, name: str, sector: str,
                                stage1: dict, rs_metrics: dict,
                                industry: str = "",
                                market: str = "US") -> str:
    """Stage 2: Stage 1 결과 + 가격 메트릭을 받아 비전공자용 큐레이션 마크다운 생성.
    market 에 따라 검색 절차/우선 출처/벤치마크 명칭을 한국 시장에 맞게 조정."""
    api_key = _get_gemini_key()
    if not api_key:
        return "GEMINI_API_KEY 미설정."

    mkt = MARKETS.get(market, MARKETS["US"])
    bench_label = mkt["bench_name"]
    is_kr = (market == "KR")

    issues = stage1.get("issues", [])
    rough = stage1.get("rough_narrative", "")

    issues_block = "(no issues found)"
    if issues:
        lines = []
        for it in issues[:30]:
            lines.append(
                f"- [{it.get('date','?')}] ({it.get('category','?')}, "
                f"{it.get('impact','?')}) {it.get('headline','')} — "
                f"{it.get('summary','')}"
                + (f" [src: {it.get('source')}]" if it.get('source') else "")
            )
        issues_block = "\n".join(lines)

    rs_5 = rs_metrics.get("rs_5")
    rs_20 = rs_metrics.get("rs_20")
    rs_60 = rs_metrics.get("rs_60")
    ir_20 = rs_metrics.get("ir_20")

    def _fmt(v, suffix=""):
        if v is None:
            return "—"
        try:
            return f"{float(v):+.2f}{suffix}"
        except Exception:
            return "—"

    today = datetime.date.today().strftime("%Y-%m-%d")

    if is_kr:
        code_only = ticker.split(".")[0]
        search_procedure = f"""[강제 검색 절차 — 보고서 작성 전 반드시 실행]
오늘 날짜: {today}. {name} ({code_only}, KOSPI) 에 대해 아래 검색을 **순서대로, 최소 4회 이상** 실행하시오:

  검색 1) "{name} {today[:7]}" 또는 "{name} 최근 뉴스" — 최근 한 달 한국어 뉴스
  검색 2) [Stage 1 이슈] 중 가장 임팩트 있어 보이는 헤드라인 1건을 그대로 검색해서 **원문 기사 본문** 확인
  검색 3) "{name} 실적" 또는 "{name} 공시 DART" — 분기 실적·전자공시(DART) 1차 자료
  검색 4) "{name} 목표주가" 또는 "{sector} 업황" — 증권사 리포트·섹터 매크로
  (선택) {name} 의 주요 경쟁사·고객사·공급사·완성차/팹리스 동향, 환율·금리·정부정책 영향

각 검색에서 확인한 **구체적 수치, 인용문, 발표 일자, 출처 URL** 을 추출해서 보고서 작성에 사용."""
    else:
        search_procedure = f"""[강제 검색 절차 — 보고서 작성 전 반드시 실행]
오늘 날짜: {today}. {ticker} ({name}) 에 대해 아래 검색을 **순서대로, 최소 4회 이상** 실행하시오:

  검색 1) "{ticker} {name} latest news {today[:7]}" — 최근 한 달 뉴스 일반 검색
  검색 2) [Stage 1 이슈] 중 가장 임팩트 있어 보이는 헤드라인 1건을 그대로 검색해서 **원문 기사 본문** 확인
  검색 3) "{ticker} earnings guidance" 또는 "{ticker} SEC 8-K filing {today[:4]}" — 실적·공시 1차 자료
  검색 4) "{ticker} analyst rating price target" 또는 "{sector} sector outlook {today[:7]}" — 애널리스트 행동·섹터 거시
  (선택) {ticker} 와 산업 내 주요 경쟁사·고객사·공급사 동향 보강

각 검색에서 확인한 **구체적 수치, 인용문, 발표 일자, 출처 URL** 을 추출해서 보고서 작성에 사용."""

    prompt = f"""=================================================================
  ⚠️ STRICT PROCEDURAL REQUIREMENT — READ THIS FIRST, BEFORE ANYTHING ELSE
=================================================================

이 작업은 Google Search 도구를 사용한 1차 자료 검증 없이 완료할 수 없습니다.
검색을 한 번도 호출하지 않고 답변하면 응답은 무효이며, 사용자는 응답을 거부합니다.

{search_procedure}

[인용 의무]
- 본문의 모든 사실 단정문 끝에 `[출처: 언론사명/기관명, YYYY-MM-DD]` 인라인 표기 의무.
- 표기가 없는 단정문은 무효 처리. 검색에서 확인 못 한 내용은 단정문 금지.
- "복수의 보도에 따르면", "시장에서는 ~로 보고 있다" 같은 출처 모호 표현은 금지.

[검색이 0건 일치할 경우]
- 솔직하게 "검색에서 확인 가능한 보강 자료가 없음" 명시. Stage 1 이슈만으로 가능한 분석만 수행.
- 절대 자체 학습 데이터에서 사실을 끌어와 보고서를 채우지 말 것.

=================================================================

당신은 도메인 지식이 전무한 일반 독자를 위해 글을 쓰는 시장 큐레이터이자, 동시에 무비판적 내러티브 동조를 거부하는 비판적 분석가입니다.

아래 종목은 가격 데이터상 최근 모멘텀이 막 발현되기 시작한 'Early Trend' 상태입니다. 그러나 **가격이 올랐다는 사실 자체가 어떤 내러티브의 정당성을 입증하지는 않습니다.** 따라서 이 보고서의 목적은 "주가 상승의 이유를 어떻게든 만들어내는 것"이 아니라, **"실제로 어떤 내러티브가 형성되고 있는지를 사실 기반으로 식별하고, 그 합리성을 비판적으로 검증하는 것"** 입니다.

원시 이슈 리스트(다른 모델이 웹 검색으로 수집)를 받아 큐레이션하되:
- 이슈가 빈약하거나 가격 움직임을 설명하지 못하면 "설명 가능한 명확한 내러티브가 없음"이라고 정직하게 명시.
- 시장의 narrative가 논리적으로 약하면 약점을 직접 지적.
- 절대 끼워맞추기·포장·아부성 결론 금지.

[종목]
- Ticker: {ticker}
- Company: {name}
- Sector: {sector}
- Industry: {industry or sector}
{('- 시장: KOSPI (한국 유가증권시장)' if is_kr else '- 시장: US (NYSE/NASDAQ — S&P 500)')}

[가격 모멘텀 ({bench_label} 대비 초과수익)]
- RS 5d: {_fmt(rs_5, '%')}, RS 20d: {_fmt(rs_20, '%')}, RS 60d: {_fmt(rs_60, '%')}
- IR 20d (초과수익 일관성, 연환산): {_fmt(ir_20)}

[Stage 1 — {('네이버 뉴스에서 수집된' if is_kr else '웹 검색으로 수집된')} 최근 {NARRATIVE_LOOKBACK_DAYS}일 이슈]
{issues_block}

[Stage 1 의 거친 내러티브 초안]
{rough}

[연구 지침 보강 — 위 STRICT PROCEDURAL REQUIREMENT 참조]
- Stage 1 이슈는 **출발점**. 보고서 본문은 본인 검색에서 보강된 디테일로 채워야 함.
- 우선 출처: {('금감원 전자공시(DART)·한국거래소·한국은행·통계청·기업 IR 자료, 그리고 한국경제·매일경제·연합뉴스·이데일리·조선비즈·아시아경제·이투데이·블룸버그/로이터/FT 한국 기사·국내 메이저 증권사(미래에셋·한투·삼성·NH·KB·신한·하이투자·키움 등) 리포트' if is_kr else '정부기관·SEC·IR·로이터/블룸버그/FT/포춘/포브스·메이저 IB 보고서')}.
- {('블로그·종목토론·미확인 단톡방·주식카페 인용 금지. 일베/디시/리딩방 출처 절대 금지.' if is_kr else '블로그·출처 불명 사이트·SEO 농장 사이트 인용 금지.')}

[작성 지침 — 반드시 준수]
- 한국어. 마크다운. **총 분량 2500~3500자** (이전보다 2~3배 깊이 있게).
- 각 섹션마다 **단순 사실 나열이 아닌 인과관계·메커니즘·시장 반응 논리**를 풀어 설명.
- 객관적 사실은 출처와 함께 단정문, 해석·추론은 "...로 해석된다" / "...일 가능성이 있다" / "근거는 약하나..." 등으로 분리.
- 전문용어가 나오면 괄호로 풀어쓰기. 예: "GLP-1(비만·당뇨 치료제 계열)", "PMI(제조업 구매관리자지수)".
- 영어 사용은 고유명사·전문용어 등 꼭 필요한 경우로 엄격히 제한.
- 이슈가 비어 있거나 검색에서도 추가 사실을 못 찾으면: "현재 시점에서는 가격을 설명할 만한 명확한 내러티브가 발견되지 않습니다"를 명시하고, 그 상태에서 가능한 가설(예: 섹터 로테이션 동조, 무관한 자금 흐름, 단기 기술적 반등)을 짧게 제시.

[필수 섹션 — 정확히 다음 헤더와 순서를 사용]

## 1. 이 회사는 무엇을 하는가 (3~5문단)
- 사업 핵심을 친숙한 비유로 시작 (예: "OO은 클라우드 보안의 '경비원'")
- 주요 매출 구성과 비중 (가능한 경우 % 또는 segment 명)
- 산업 내 포지션, 주요 경쟁사
- 회사 규모 (시총, 매출 규모, 직원수 등 직관적 수치)
- 일반 독자가 "아, 이런 회사구나" 그림이 그려지도록 작성

## 2. 지금 형성되고 있는 내러티브 (4~6문단)
- 시장이 이 종목·산업에 어떤 스토리를 부여하고 있는지
- 그 스토리의 시발점 (catalyst의 시간적 출발점)
- 어떤 투자자 그룹이 이 내러티브에 동조하고 있는가 (기관/리테일/특정 펀드 등)
- 가격 모멘텀(RS 데이터)과 내러티브의 정합성 평가
- **이 내러티브가 합리적인가에 대한 1차 비판적 코멘트**

## 3. 발견된 이슈와 그 의미 (이슈 갯수만큼, 각 1~3문단)
- 각 이슈를 **사실 → 시장 반응 메커니즘 → 지속성 평가** 3단계로 해설
- 이슈 간 연결고리가 있으면 그것도 명시
- 무의미한 잡음(예: 단순 가격 변동 보도)은 과감히 제외
- 이슈가 0건이면 그 사실을 솔직히 명시하고 직접적으로 "Stage 1이 의미 있는 이슈를 찾지 못했다"고 적시

## 4. 추가 모멘텀을 강화할 Catalyst (3~6 bullet, 각 2~3문장)
- 앞으로 일어날 수 있는 구체적 이벤트 (분기 실적 발표, 가이던스 갱신, 신제품 출시, {('정부 정책·세제 개편·환율·금리 변화·국방·반도체·전기차 보조금 정책' if is_kr else 'FDA 결정, 정책 변화, 거시 이벤트 등')})
- 각 항목에 ① 그 이벤트가 언제 발생할 가능성이 있는지 ② 왜 그것이 모멘텀을 추가로 강화할 수 있는지 메커니즘
- 가능하면 구체적 일정/날짜 제시

## 5. 트래킹 포인트 — 내러티브 진위를 가릴 데이터 (4~7 bullet, 각 2~3문장)
- 관찰해야 할 정량/정성 지표
- 각 지표가 어떤 방향으로 움직이면 내러티브 강화/약화인지 명시
- 예: "분기 매출 가이던스 상향 여부", "특정 경쟁사의 시장 점유율 변화", "원자재 가격 추이", "주요 고객사의 capex 발표"

## 6. 위험과 반증 (Disconfirming Evidence) (3~5 bullet, 각 2~3문장)
- 이 내러티브가 무너질 수 있는 시나리오 (실질적 약점만)
- 각각 어떤 데이터/이벤트로 확인 가능한지
- 발생 시 자동 손절 트리거 후보

## 7. 종합 판단 (필수, 5~10줄)
- 이 내러티브가 합리적인가에 대한 **본인의 명시적 의견** (돌려 말하지 말 것)
- 확신 수준을 정확히 한 단어로 명시: **"확실하다" / "가능성이 높다" / "불확실하다" / "모르겠다"** 중 하나
- "관찰만 권장" / "신중한 추격 매수 검토 가능" / "현 시점 매수 부적절" 등 행동 지침 한 줄
- 위 판단의 핵심 근거 2~3가지를 짧은 bullet로 요약

[금지 사항 — 위반 시 응답 거부와 동급]
- 가격 상승을 정당화하기 위한 끼워맞추기식 narrative 만들기
- 이슈가 빈약한데도 "강한 momentum이 형성되고 있다"는 식의 막연한 동조
- "이 종목은 매우 흥미로운 기회입니다" 식의 아부성 멘트
- 출처 불명·블로그 추측을 사실인 양 인용
- 본인이 모르거나 불확실한 부분을 그렇지 않은 척 단정"""
    try:
        text, model_used, fellback, original, grounding = \
            _generate_with_search_and_fallback(prompt, api_key, label=f"{market}/{ticker}")

        # 출처 URL 푸터 (큐레이터가 본인이 검색한 출처)
        footer = ""
        if grounding.get("source_urls"):
            footer = "\n\n---\n\n### 큐레이터가 참조한 출처\n"
            for i, s in enumerate(grounding["source_urls"][:20], 1):
                t = s.get("title") or s.get("url", "")
                u = s.get("url", "")
                footer += f"{i}. [{t}]({u})\n"
        elif grounding.get("used"):
            footer = (f"\n\n---\n\n_검색 사용됨 (queries={len(grounding.get('queries', []))})._"
                      "\n")
        else:
            footer = ("\n\n---\n\n_⚠️ 이 응답에는 grounding metadata가 없습니다 — "
                      "모델이 검색 없이 답했을 수 있습니다._\n")

        banner = ""
        if fellback:
            _log(f"    [{ticker}] Pro quota → Flash fallback (grounding 유지)")
            banner = (
                f"> ⚠️ **{GEMINI_MODEL} 일일 quota 초과로 {GEMINI_FLASH_MODEL} 로 폴백되었습니다 "
                f"(Google Search grounding 은 유지).**\n\n"
            )

        return banner + text + footer
    except Exception as exc:
        _log(f"[!] Pro narrative curation failed for {ticker}: {exc}")
        if _is_quota_error(exc):
            return ("## 큐레이션 실패 — Quota 초과\n\n"
                    f"Pro 모델과 Flash 모델 모두 quota 한도를 초과했습니다.\n\n"
                    f"```\n{exc}\n```\n\n"
                    "24시간 후 다시 시도하거나, 다른 API 키로 교체해주세요.")
        return f"## 큐레이션 실패\n\n```\n{exc}\n```"


def _process_one_narrative(ticker: str, name: str, sector: str,
                           rs_metrics: dict, industry: str = "",
                           market: str = "US") -> dict:
    """단일 종목 파이프라인: Stage 1 → Stage 2.
    market='KR' 이면 Stage 1 은 네이버 뉴스 API 사용,
    market='US' 이면 기존대로 Gemini Flash + Google Search 사용."""
    if market == "KR":
        s1 = find_ticker_issues_naver(ticker, name, sector, industry=industry)
    else:
        s1 = find_ticker_issues_flash(ticker, name, sector, industry=industry)
    s2 = curate_ticker_narrative_pro(ticker, name, sector, s1, rs_metrics,
                                     industry=industry, market=market)
    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "industry": industry or sector,
        "market": market,
        "rs_metrics": rs_metrics,
        "issues": s1.get("issues", []),
        "rough_narrative": s1.get("rough_narrative", ""),
        "narrative_md": s2,
        "stage1_error": s1.get("error"),
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ═══════════════════════════════════════════════════════════════════
# Flask app + state
#
# G 구조:
#   G["US"] = {close, sector_results, sector_info, sector_w52,
#              sp500_results, sp500_info, sp500_w52,  # ← constituents (KR 의 경우 KOSPI)
#              breadth, beta_adj, updated, gemini_analysis,
#              gemini_updated, narratives:{ticker: {...}}}
#   G["KR"] = {... 동일 구조 ...}
#   G["KR_ERROR"] (선택) = KR 분석 실패시 에러 메시지
# ═══════════════════════════════════════════════════════════════════
app = Flask(__name__)
G: dict = {}
G_STATUS: dict = {"state": "IDLE", "step": "", "detail": "", "progress": 0}


def _market_state(market: str) -> dict:
    """G[market] 안전 접근. 없으면 None."""
    return G.get(market) if market in MARKETS else None


def _run_analysis_for_market(market: str, base_progress: int, span: int) -> dict:
    """단일 시장 분석 파이프라인. progress 는 base_progress ~ base_progress+span 매핑.
    G 에 직접 쓰지 않고 결과 dict 만 반환."""
    if market not in MARKETS:
        raise ValueError(f"unknown market: {market}")
    mkt = MARKETS[market]
    bench = mkt["bench"]
    cons_label = mkt["constituents_label"]

    if market == "US":
        constituents_tickers, constituents_info = get_sp500()
        etf_tickers, etf_info = get_sector_etfs()
    else:  # KR
        constituents_tickers, constituents_info = get_kospi()
        etf_tickers, etf_info = get_kr_sector_etfs()

    if not constituents_tickers:
        raise RuntimeError(f"{market}: {cons_label} 종목 목록을 가져올 수 없습니다")

    all_tickers = sorted(set(constituents_tickers + etf_tickers + [bench]))
    G_STATUS.update(step=f"[{market}] 가격 데이터 동기화",
                    detail=f"{len(all_tickers)} tickers",
                    progress=base_progress + max(1, int(span * 0.05)))
    close = sync_prices(all_tickers, bench=bench)

    if bench not in close.columns:
        raise RuntimeError(f"{market} 벤치마크({bench}) 가격을 가져오지 못했습니다")

    cons_cols = [bench] + [t for t in constituents_tickers if t in close.columns]
    etf_cols = [bench] + [t for t in etf_tickers if t in close.columns]

    G_STATUS.update(step=f"[{market}] {cons_label} 분석",
                    detail="RS / IR 계산 중...",
                    progress=base_progress + int(span * 0.55))
    cons_results = screen(close[cons_cols], beta_adj=BETA_ADJ,
                          label=f"{market} {cons_label}", bench=bench)

    G_STATUS.update(step=f"[{market}] ETF 분석",
                    detail="RS / IR 계산 중...",
                    progress=base_progress + int(span * 0.70))
    sector_results = screen(close[etf_cols], beta_adj=BETA_ADJ,
                            label=f"{market} ETF", bench=bench)

    G_STATUS.update(step=f"[{market}] 52주 신고가/신저가", detail="",
                    progress=base_progress + int(span * 0.80))
    cons_w52 = detect_52w(close[cons_cols], label=f"{market} {cons_label}",
                          bench=bench)
    sector_w52 = detect_52w(close[etf_cols], label=f"{market} ETF", bench=bench)

    G_STATUS.update(step=f"[{market}] Market Breadth", detail="",
                    progress=base_progress + int(span * 0.95))
    breadth = calc_breadth(close, cons_results, constituents_info, cons_w52,
                           bench=bench)

    return {
        "market": market,
        "close": close,
        "sector_results": sector_results,
        "sector_info": etf_info,
        "sector_w52": sector_w52,
        "sp500_results": cons_results,
        "sp500_info": constituents_info,
        "sp500_w52": cons_w52,
        "breadth": breadth,
        "beta_adj": BETA_ADJ,
        "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "gemini_analysis": "",
        "gemini_updated": "",
        "narratives": {},
    }


def _run_analysis():
    G_STATUS.update(state="LOADING", step="시작", detail="US + KR 시장 분석", progress=1)
    G.pop("KR_ERROR", None)
    try:
        # US: 1 ~ 50
        G_STATUS.update(step="[1/2] US 분석 시작", detail="S&P 500 + Sector ETF",
                        progress=2)
        G["US"] = _run_analysis_for_market("US", base_progress=2, span=48)
        _log("\n* US Analysis complete\n")

        # KR: 50 ~ 99 (KR 실패해도 US 결과는 유지)
        G_STATUS.update(step="[2/2] KR 분석 시작",
                        detail="KOSPI + 한국 테마 ETF", progress=52)
        try:
            G["KR"] = _run_analysis_for_market("KR", base_progress=52, span=46)
            _log("\n* KR Analysis complete\n")
        except Exception as kr_exc:
            _log(f"[!] KR analysis failed (US 결과는 유지): {kr_exc}")
            import traceback; traceback.print_exc()
            G["KR_ERROR"] = str(kr_exc)

        G_STATUS.update(state="READY", step="완료", detail="", progress=100)
        _log("\n* Analysis complete\n")
    except Exception as exc:
        _log(f"[!] Analysis failed: {exc}")
        import traceback; traceback.print_exc()
        G_STATUS.update(state="ERROR", step="오류", detail=str(exc), progress=0)


# ═══════════════════════════════════════════════════════════════════
# Landing page
# ═══════════════════════════════════════════════════════════════════
TEMPLATE_LANDING = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Regime Analyzer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{display:flex;justify-content:center;align-items:center;min-height:100vh;
  background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0f172a 100%);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
.card{background:#fff;border-radius:20px;padding:3rem 2.5rem;max-width:520px;width:92%;
  box-shadow:0 25px 60px rgba(0,0,0,.35);text-align:center}
.logo{width:64px;height:64px;border-radius:16px;background:linear-gradient(135deg,#059669,#0d9488);
  display:flex;align-items:center;justify-content:center;margin:0 auto 1.5rem;font-size:1.6rem;font-weight:800;color:#fff}
h1{font-size:1.5rem;font-weight:700;color:#1e293b;margin-bottom:.4rem}
.sub{color:#64748b;font-size:.88rem;margin-bottom:2rem}
.feats{text-align:left;margin-bottom:2rem}
.feat{display:flex;align-items:center;gap:.7rem;padding:.55rem 0;border-bottom:1px solid #f1f5f9;font-size:.85rem;color:#334155}
.feat:last-child{border:none}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.btn{background:linear-gradient(135deg,#1e293b,#334155);color:#fff;border:none;
  padding:.95rem 2.5rem;border-radius:12px;font-size:1rem;font-weight:600;cursor:pointer;transition:all .2s}
.btn:hover{transform:translateY(-1px);box-shadow:0 8px 24px rgba(30,41,59,.3)}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none;box-shadow:none}
.progress-area{display:none;margin-top:1.5rem}
.pbar-wrap{background:#e2e8f0;border-radius:10px;height:10px;overflow:hidden;margin-bottom:1rem}
.pbar{background:linear-gradient(90deg,#059669,#0d9488);height:100%;border-radius:10px;transition:width .5s ease;width:0%}
.step{font-size:.88rem;font-weight:600;color:#1e293b;margin-bottom:.3rem}
.detail{font-size:.8rem;color:#94a3b8}
.err{display:none;margin-top:1.5rem;padding:1rem;background:#fef2f2;border-radius:10px;color:#991b1b;font-size:.85rem;text-align:left}
.err-retry{margin-top:.8rem;text-align:center}
</style></head><body>
<div class="card">
  <div class="logo">M</div>
  <h1>Market Regime Analyzer</h1>
  <p class="sub">US (S&amp;P 500) + KR (KOSPI) &mdash; RS &middot; Breadth &middot; 52W &middot; On-demand AI</p>
  <div class="feats">
    <div class="feat"><span class="dot" style="background:#059669"></span><strong>US</strong>: S&amp;P 500 + 40개 Sector ETF &mdash; SPY 대비 상대강도/Breadth</div>
    <div class="feat"><span class="dot" style="background:#0d9488"></span><strong>KR</strong>: KOSPI 주요 종목 + 한국 테마 ETF &mdash; KODEX 200 대비 분석</div>
    <div class="feat"><span class="dot" style="background:#6366f1"></span>Early Trend 종목 AI 내러티브 (US: Google Search &middot; KR: 네이버 뉴스 API)</div>
  </div>
  <div id="idle"><button class="btn" id="startBtn" onclick="startAnalysis()">분석 시작</button></div>
  <div class="progress-area" id="progressArea">
    <div class="pbar-wrap"><div class="pbar" id="pbar"></div></div>
    <div class="step" id="stepText">준비 중...</div>
    <div class="detail" id="detailText"></div>
  </div>
  <div class="err" id="errArea"><strong>오류:</strong><p id="errText"></p>
    <div class="err-retry"><button class="btn" onclick="startAnalysis()">재시도</button></div></div>
</div>
<script>
function startAnalysis(){
  document.getElementById('idle').style.display='none';
  document.getElementById('errArea').style.display='none';
  document.getElementById('progressArea').style.display='block';
  document.getElementById('startBtn').disabled=true;
  fetch('/api/start',{method:'POST'}).then(()=>pollStatus());
}
function pollStatus(){
  fetch('/api/status').then(r=>r.json()).then(d=>{
    document.getElementById('pbar').style.width=d.progress+'%';
    document.getElementById('stepText').textContent=d.step;
    document.getElementById('detailText').textContent=d.detail;
    if(d.state==='READY'){window.location.href='/dashboard';}
    else if(d.state==='ERROR'){
      document.getElementById('progressArea').style.display='none';
      document.getElementById('errArea').style.display='block';
      document.getElementById('errText').textContent=d.detail;
      document.getElementById('idle').style.display='block';
      document.getElementById('startBtn').disabled=false;
    } else{setTimeout(pollStatus,1200);}
  }).catch(()=>setTimeout(pollStatus,2000));
}
</script></body></html>"""

# ═══════════════════════════════════════════════════════════════════
# Dashboard template (tabbed)
# ═══════════════════════════════════════════════════════════════════
TEMPLATE_DASHBOARD = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Regime Analyzer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:#f0f2f5;color:#1a1a2e}
.hdr{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#1e293b 100%);color:#fff;padding:1.4rem 2rem 0}
.hdr-inner{max-width:1800px;margin:0 auto;display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap}
.hdr h1{font-size:1.6rem;font-weight:700}.hdr p{opacity:.7;margin-top:.2rem;font-size:.88rem}
.hdr-btns{display:flex;gap:.5rem;align-items:center}
.hdr-btn{padding:.45rem 1rem;border:1px solid rgba(255,255,255,.25);border-radius:8px;background:rgba(255,255,255,.08);color:#fff;cursor:pointer;font-size:.8rem;transition:.15s;text-decoration:none}
.hdr-btn:hover{background:rgba(255,255,255,.18)}
.mkt-tabs{max-width:1800px;margin:1.1rem auto 0;display:flex;gap:.4rem;padding:0 0;border-bottom:0;align-items:flex-end}
.mkt-tab{display:inline-flex;align-items:center;gap:.5rem;padding:.65rem 1.4rem;border-radius:10px 10px 0 0;background:rgba(255,255,255,.08);color:#cbd5e1;text-decoration:none;font-size:.92rem;font-weight:700;letter-spacing:.02em;border:1px solid rgba(255,255,255,.12);border-bottom:none;transition:.15s}
.mkt-tab:hover{background:rgba(255,255,255,.16);color:#fff}
.mkt-tab.on{background:#f0f2f5;color:#0f172a;border-color:#f0f2f5}
.mkt-tab .ind{font-size:.62rem;font-weight:600;padding:.1rem .42rem;border-radius:5px;background:rgba(15,23,42,.16);color:#475569;border:1px solid rgba(15,23,42,.1)}
.mkt-tab.on .ind{background:#dcfce7;color:#166534;border-color:transparent}
.mkt-tab .ind.bad{background:#fee2e2;color:#991b1b;border-color:transparent}
.mkt-tab .ind.na{background:rgba(15,23,42,.08);color:#94a3b8;border-color:transparent}
.wrap{max-width:1800px;margin:0 auto;padding:1.2rem}
.kr-banner{background:#fef3c7;color:#92400e;border-left:4px solid #f59e0b;padding:.75rem 1rem;border-radius:8px;margin-bottom:1rem;font-size:.82rem}

.tabs{display:flex;gap:0;margin-bottom:1.2rem;border-bottom:2px solid #e2e8f0}
.tab{padding:.65rem 1.4rem;border:none;background:none;cursor:pointer;font-size:.88rem;font-weight:600;color:#64748b;border-bottom:2px solid transparent;margin-bottom:-2px;transition:.15s}
.tab:hover{color:#1e293b}.tab.on{color:#1e293b;border-bottom-color:#1e293b}
.tab-content{display:none}.tab-content.active{display:block}

.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.8rem;margin-bottom:1.2rem}
.card{background:#fff;border-radius:12px;padding:.9rem 1rem;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.card .lb{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em}
.card .vl{font-size:1.5rem;font-weight:700;margin-top:.15rem}
.card.c-ld .vl{color:#059669}.card.c-et .vl{color:#0d9488}
.card.c-52h .vl{color:#16a34a}.card.c-52l .vl{color:#dc2626}

.section-52w{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.2rem}
.w52-card{background:#fff;border-radius:12px;padding:1.1rem 1.3rem;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.w52-card h3{font-size:.82rem;font-weight:700;margin-bottom:.6rem}
.w52-high{border-left:4px solid #16a34a}.w52-high h3{color:#16a34a}
.w52-low{border-left:4px solid #dc2626}.w52-low h3{color:#dc2626}
.w52-items{display:flex;flex-wrap:wrap;gap:.4rem}
.w52-item{background:#f8fafc;border-radius:7px;padding:.3rem .6rem;font-size:.75rem;display:flex;gap:.35rem;align-items:center}
.w52-item .tk{font-weight:700;color:#1e293b}.w52-item .nm{color:#64748b;max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.w52-empty{color:#94a3b8;font-size:.8rem;font-style:italic}

.ldist{background:#fff;border-radius:12px;padding:1.1rem 1.3rem;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:1.2rem}
.ldist-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:.7rem;flex-wrap:wrap;gap:.5rem}
.ldist h3{font-size:.85rem;font-weight:700;color:#1e293b}
.ldist-toggle{display:flex;gap:.3rem}
.ldist-tbtn{padding:.3rem .8rem;border:1px solid #e2e8f0;border-radius:6px;background:#fff;cursor:pointer;font-size:.72rem;font-weight:600;color:#64748b;transition:.15s}
.ldist-tbtn:hover{background:#f1f5f9;color:#1e293b}
.ldist-tbtn.on{background:#1e293b;color:#fff;border-color:#1e293b}
.ldist-list{max-height:520px;overflow-y:auto;padding-right:.3rem}
.ldist-list::-webkit-scrollbar{width:6px}
.ldist-list::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}
.ldist-row{display:flex;align-items:center;gap:.5rem;margin-bottom:.35rem;font-size:.78rem}
.ldist-label{width:240px;color:#475569;flex-shrink:0;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ldist-bar-wrap{flex:1;background:#e2e8f0;border-radius:4px;height:14px;overflow:hidden}
.ldist-bar{background:linear-gradient(90deg,#059669,#0d9488);height:100%;border-radius:4px;transition:width .3s}
.ldist-cnt{width:30px;text-align:right;font-weight:600;color:#1e293b}

.ai-section{background:#fff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:1.2rem;overflow:hidden}
.ai-header{display:flex;justify-content:space-between;align-items:center;padding:.9rem 1.3rem;border-bottom:1px solid #e2e8f0;background:linear-gradient(135deg,#eff6ff 0%,#f0fdf4 100%)}
.ai-header h3{font-size:.9rem;font-weight:700;color:#1e293b}
.ai-meta{display:flex;align-items:center;gap:.7rem}
.ai-updated{font-size:.7rem;color:#94a3b8}
.ai-refresh{padding:.3rem .7rem;border:1px solid #e2e8f0;border-radius:6px;background:#fff;cursor:pointer;font-size:.75rem;transition:.15s}
.ai-refresh:hover{background:#f1f5f9}
.ai-content{padding:1.3rem;font-size:.86rem;line-height:1.7;max-height:650px;overflow-y:auto}
.ai-content h1{font-size:1.1rem;font-weight:700;margin:1rem 0 .4rem;color:#1e293b}
.ai-content h2{font-size:1rem;font-weight:700;margin:.9rem 0 .4rem;color:#334155}
.ai-content h3{font-size:.92rem;font-weight:600;margin:.7rem 0 .3rem;color:#475569}
.ai-content p{margin-bottom:.5rem}.ai-content ul,.ai-content ol{margin:.3rem 0 .5rem 1.4rem}.ai-content li{margin-bottom:.2rem}
.ai-content strong{color:#1e293b}

.narr-bar{background:#fff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.08);padding:1rem 1.2rem;margin-bottom:1rem;display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap}
.narr-bar h3{font-size:.95rem;font-weight:700;color:#1e293b}
.narr-bar p{font-size:.78rem;color:#64748b;margin-top:.2rem}
.narr-batch{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;justify-content:flex-end}
.narr-batch-opt{font-size:.75rem;color:#475569;display:flex;align-items:center;gap:.3rem;cursor:pointer;user-select:none}
.narr-batch-opt input{cursor:pointer}
.narr-batch-btn{background:linear-gradient(135deg,#0d9488,#059669);color:#fff;border:none;
  padding:.55rem 1rem;border-radius:8px;font-size:.82rem;font-weight:600;cursor:pointer;transition:.15s;white-space:nowrap}
.narr-batch-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(13,148,136,.35)}
.narr-batch-btn:disabled{opacity:.55;cursor:not-allowed;transform:none;box-shadow:none}
.narr-batch-stop{background:#dc2626;color:#fff;border:none;padding:.55rem 1rem;border-radius:8px;
  font-size:.82rem;font-weight:600;cursor:pointer;transition:.15s;white-space:nowrap}
.narr-batch-stop:hover{background:#b91c1c}
.narr-batch-prog{display:flex;flex-direction:column;gap:.25rem;min-width:220px;width:100%;
  margin-top:.4rem;flex-basis:100%}
.narr-batch-prog-bar{height:6px;background:#e2e8f0;border-radius:3px;overflow:hidden}
.narr-batch-prog-fill{height:100%;background:linear-gradient(90deg,#0d9488,#059669);width:0%;transition:width .3s}
.narr-batch-prog-txt{font-size:.72rem;color:#475569;font-family:monospace}
.narr-actions{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
.narr-btn{padding:.55rem 1.1rem;border:none;border-radius:8px;background:linear-gradient(135deg,#0d9488,#059669);color:#fff;cursor:pointer;font-size:.82rem;font-weight:600;transition:.15s}
.narr-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(13,148,136,.3)}
.narr-btn:disabled{opacity:.5;cursor:not-allowed;transform:none;box-shadow:none}
.narr-btn-sec{background:#fff;color:#475569;border:1px solid #e2e8f0;padding:.55rem 1.1rem;border-radius:8px;cursor:pointer;font-size:.82rem;font-weight:600;transition:.15s}
.narr-btn-sec:hover{background:#f1f5f9}
.narr-progress{flex:1;min-width:240px}
.narr-pbar-wrap{background:#e2e8f0;border-radius:8px;height:8px;overflow:hidden;margin-bottom:.3rem}
.narr-pbar{background:linear-gradient(90deg,#0d9488,#059669);height:100%;transition:width .3s}
.narr-detail{font-size:.75rem;color:#64748b}
.narr-empty{background:#fff;border-radius:12px;padding:3rem 1.5rem;text-align:center;color:#94a3b8;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.narr-empty h3{color:#475569;font-weight:600;margin-bottom:.5rem;font-size:1rem}
.narr-empty p{font-size:.85rem;line-height:1.6;max-width:520px;margin:0 auto}

.narr-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:1rem}
.narr-card{background:#fff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.08);overflow:hidden;display:flex;flex-direction:column;transition:.15s}
.narr-card:hover{box-shadow:0 4px 16px rgba(0,0,0,.1)}
.narr-card-hdr{padding:.9rem 1.1rem;border-bottom:1px solid #f1f5f9;display:flex;justify-content:space-between;align-items:flex-start;gap:.5rem}
.narr-card-hdr .l{flex:1;min-width:0}
.narr-card-name{font-size:1rem;font-weight:700;color:#1e293b;cursor:pointer;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.narr-card-name:hover{color:#0d9488}
.narr-card-tk{font-size:.72rem;color:#64748b;margin-top:.2rem;font-family:monospace;letter-spacing:.02em}
.narr-card-tk .kind{font-size:.6rem;font-weight:600;padding:.1rem .4rem;border-radius:4px;margin-left:.3rem;vertical-align:middle;font-family:inherit}
.narr-card-tk .kind.s{background:#e0e7ff;color:#3730a3}
.narr-card-tk .kind.e{background:#fef3c7;color:#92400e}
.narr-card-sec{font-size:.7rem;color:#94a3b8;margin-top:.15rem}
.narr-card-rs{display:flex;gap:.4rem;font-size:.7rem;flex-wrap:wrap}
.narr-card-rs span{padding:.15rem .45rem;border-radius:5px;background:#f1f5f9;color:#475569;font-weight:600}
.narr-card-rs span.p{background:#dcfce7;color:#166534}
.narr-card-rs span.n{background:#fee2e2;color:#991b1b}
.narr-card-body{padding:.9rem 1.1rem;flex:1}
.narr-card-rough{font-size:.82rem;color:#475569;line-height:1.55;margin-bottom:.6rem;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}
.narr-card-meta{display:flex;justify-content:space-between;align-items:center;font-size:.7rem;color:#94a3b8;padding:.5rem 1.1rem;border-top:1px solid #f1f5f9;background:#fafbfc}
.narr-card-meta .iss{font-weight:600;color:#0d9488}
.narr-open{cursor:pointer;color:#0d9488;font-weight:600}
.narr-open:hover{color:#0f766e}
.narr-err{color:#dc2626;font-size:.75rem;font-style:italic}

.narr-modal{display:none;position:fixed;inset:0;z-index:1100;background:rgba(0,0,0,.55);justify-content:center;align-items:center}
.narr-modal.open{display:flex}
.narr-modal-box{background:#fff;border-radius:14px;width:96%;max-width:1000px;max-height:92vh;overflow-y:auto;padding:1.5rem 1.8rem;box-shadow:0 25px 60px rgba(0,0,0,.35);position:relative}
.narr-modal-hdr{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem;padding-bottom:.8rem;border-bottom:1px solid #e2e8f0;flex-wrap:wrap;gap:.5rem}
.narr-modal-hdr h2{font-size:1.3rem;font-weight:700;color:#1e293b}
.narr-modal-hdr .sub{font-size:.85rem;color:#64748b;margin-top:.2rem}
.narr-modal-state{padding:2.5rem 1rem;text-align:center;font-size:.88rem;color:#475569}
.narr-modal-state .spinner{display:inline-block;width:32px;height:32px;border:3px solid #e2e8f0;border-top-color:#0d9488;border-radius:50%;animation:narrSpin .8s linear infinite;margin-bottom:1rem}
.narr-modal-state .hint{color:#94a3b8;font-size:.78rem;margin-top:.6rem}
.narr-modal-state.error{color:#dc2626}
@keyframes narrSpin{to{transform:rotate(360deg)}}
.narr-card-status[data-status="idle"]{color:#94a3b8}
.narr-card-status[data-status="loading"]{color:#0d9488;font-weight:600}
.narr-card-status[data-status="ready"]{color:#059669;font-weight:600}
.narr-card-status[data-status="error"]{color:#dc2626;font-weight:600}
.narr-modal-md{font-size:.88rem;line-height:1.7;color:#1e293b;margin-bottom:1.5rem}
.narr-modal-md h1{font-size:1.15rem;font-weight:700;margin:1.2rem 0 .5rem;color:#1e293b;border-bottom:1px solid #e2e8f0;padding-bottom:.3rem}
.narr-modal-md h2{font-size:1.02rem;font-weight:700;margin:1.1rem 0 .5rem;color:#0d9488}
.narr-modal-md h3{font-size:.92rem;font-weight:600;margin:.8rem 0 .3rem;color:#475569}
.narr-modal-md p{margin-bottom:.6rem}
.narr-modal-md ul,.narr-modal-md ol{margin:.3rem 0 .7rem 1.4rem}
.narr-modal-md li{margin-bottom:.3rem}
.narr-modal-md strong{color:#1e293b;font-weight:700}
.narr-modal-md code{background:#f1f5f9;padding:.1rem .35rem;border-radius:4px;font-size:.85em}
.narr-issues{background:#f8fafc;border-radius:10px;padding:1rem 1.2rem;margin-top:1rem}
.narr-issues h4{font-size:.85rem;font-weight:700;color:#1e293b;margin-bottom:.6rem}
.narr-issue{padding:.55rem 0;border-bottom:1px dashed #e2e8f0;font-size:.8rem}
.narr-issue:last-child{border-bottom:none}
.narr-issue-meta{display:flex;gap:.5rem;align-items:center;margin-bottom:.2rem;flex-wrap:wrap}
.narr-issue-date{font-size:.7rem;color:#64748b;font-family:monospace}
.narr-issue-cat{font-size:.65rem;padding:.1rem .4rem;border-radius:4px;background:#e2e8f0;color:#475569;font-weight:600}
.narr-issue-imp{font-size:.65rem;padding:.1rem .4rem;border-radius:4px;font-weight:600}
.narr-issue-imp.positive{background:#dcfce7;color:#166534}
.narr-issue-imp.negative{background:#fee2e2;color:#991b1b}
.narr-issue-imp.mixed{background:#fef3c7;color:#92400e}
.narr-issue-imp.neutral{background:#f1f5f9;color:#64748b}
.narr-issue-headline{font-weight:600;color:#1e293b;font-size:.83rem;margin-bottom:.2rem}
.narr-issue-summary{color:#475569;line-height:1.5;font-size:.78rem}
.narr-issue-src{font-size:.7rem;color:#94a3b8;margin-top:.15rem}

/* ── 모달 상단 메트릭 스트립 (주가 + RS20/60) + 일자 선택기 ── */
.narr-metric-strip{display:flex;flex-wrap:wrap;align-items:stretch;gap:.6rem;
  background:linear-gradient(135deg,#0f172a,#1e293b);color:#f1f5f9;
  border-radius:12px;padding:.85rem 1.1rem;margin-bottom:1.1rem}
.narr-metric{flex:1;min-width:110px;display:flex;flex-direction:column;justify-content:center;
  padding:.35rem .55rem;border-right:1px solid rgba(255,255,255,.08)}
.narr-metric:last-child{border-right:none}
.narr-metric .lb{font-size:.66rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;
  margin-bottom:.25rem;font-weight:600}
.narr-metric .vl{font-size:1.15rem;font-weight:700;color:#f8fafc;font-feature-settings:"tnum" 1;line-height:1.1}
.narr-metric .vl.p{color:#34d399}
.narr-metric .vl.n{color:#f87171}
.narr-metric .vl.muted{color:#64748b;font-weight:500;font-size:.95rem}
.narr-metric .sub{font-size:.65rem;color:#cbd5e1;margin-top:.15rem}

.narr-hist{display:flex;align-items:center;gap:.5rem;margin:0 0 .9rem;font-size:.78rem;color:#475569}
.narr-hist label{font-weight:600;color:#1e293b}
.narr-hist select{padding:.32rem .55rem;border:1px solid #e2e8f0;border-radius:6px;
  font-size:.78rem;background:#fff;color:#1e293b;outline:none;min-width:200px}
.narr-hist select:focus{border-color:#0d9488;box-shadow:0 0 0 2px rgba(13,148,136,.18)}
.narr-hist .badge-old{background:#fef3c7;color:#92400e;font-size:.65rem;font-weight:700;
  padding:.15rem .45rem;border-radius:5px;letter-spacing:.02em}

/* ── 모달 내장 차트 (주가 + RS) ── */
.narr-chart-box{background:#fff;border:1px solid #e2e8f0;border-radius:12px;
  padding:.9rem 1rem .7rem;margin-bottom:1.1rem}
.narr-chart-loading{padding:1.5rem 0;text-align:center;font-size:.82rem;color:#64748b}
.narr-chart-wrap{position:relative;height:300px;margin-bottom:.5rem}
.narr-rs-wrap{position:relative;height:160px}

.filters{display:flex;gap:.4rem;margin-bottom:.9rem;flex-wrap:wrap}
.fbtn{padding:.4rem .85rem;border:1px solid #e2e8f0;border-radius:8px;background:#fff;cursor:pointer;font-size:.8rem;transition:.15s}
.fbtn:hover{background:#f1f5f9}.fbtn.on{background:#1e293b;color:#fff;border-color:#1e293b}
.srch{margin-bottom:.9rem}.srch input{padding:.45rem .8rem;border:1px solid #e2e8f0;border-radius:8px;font-size:.83rem;width:260px;outline:none}
.srch input:focus{border-color:#94a3b8;box-shadow:0 0 0 2px rgba(148,163,184,.2)}

.tw{background:#fff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.08);overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.8rem}
thead{background:#f8fafc;position:sticky;top:0;z-index:2}
th{padding:.6rem .45rem;text-align:left;font-weight:600;color:#475569;cursor:pointer;user-select:none;white-space:nowrap;border-bottom:2px solid #e2e8f0}
th:hover{color:#1e293b}th .si{margin-left:2px;opacity:.3;font-size:.65rem}
td{padding:.45rem;border-bottom:1px solid #f1f5f9;white-space:nowrap}
tr:hover td{background:#f8fafc}
.cat-cell{display:flex;flex-direction:column;gap:.1rem;line-height:1.25}
.cat-sec{font-size:.78rem;color:#475569;font-weight:500}
.cat-ind{font-size:.7rem;color:#94a3b8}
.pos{color:#059669}.neg{color:#dc2626}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:6px;font-size:.7rem;font-weight:600}
.b-ld{background:#dcfce7;color:#166534}.b-et{background:#ccfbf1;color:#115e59}
.b-no{background:#f3f4f6;color:#6b7280}
.b-52h{background:#dcfce7;color:#166534;font-size:.66rem}.b-52l{background:#fee2e2;color:#991b1b;font-size:.66rem}
.tklink{cursor:pointer;color:#2563eb;transition:.15s}.tklink:hover{color:#1d4ed8;text-decoration:underline}
.chart-modal{display:none;position:fixed;inset:0;z-index:1000;background:rgba(0,0,0,.55);justify-content:center;align-items:center}
.chart-modal.open{display:flex}
.chart-box{background:#fff;border-radius:16px;width:94%;max-width:1100px;max-height:92vh;overflow-y:auto;padding:1.5rem 1.8rem;box-shadow:0 25px 60px rgba(0,0,0,.35);position:relative;transition:all .25s ease}
.chart-box.fs{width:100%;max-width:100%;height:100vh;max-height:100vh;border-radius:0;overflow-y:hidden;display:flex;flex-direction:column}
.chart-box.fs .chart-wrap{flex:1;height:0;margin-bottom:4px}
.chart-box.fs .rs-wrap{height:0;flex:0 0 30%}
.chart-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:.8rem;flex-shrink:0}
.chart-hdr h2{font-size:1.15rem;font-weight:700;color:#1e293b}
.chart-hdr .sub{font-size:.82rem;color:#64748b;margin-left:.5rem;font-weight:400}
.chart-btns{display:flex;gap:.35rem}
.cbtn{border:none;background:#f1f5f9;border-radius:8px;padding:.4rem .8rem;cursor:pointer;font-size:.8rem;font-weight:600;color:#475569;transition:.15s;white-space:nowrap}
.cbtn:hover{background:#e2e8f0}
.cbtn.acc{background:#1e293b;color:#fff}.cbtn.acc:hover{background:#334155}
.chart-loading{text-align:center;padding:3rem;color:#94a3b8;font-size:.9rem;flex-shrink:0}
.chart-wrap{position:relative;height:370px;margin-bottom:.8rem}
.rs-wrap{position:relative;height:200px}
.zoom-hint{text-align:center;font-size:.7rem;color:#94a3b8;margin-top:.4rem;flex-shrink:0}
.ft{text-align:center;padding:1.4rem;color:#94a3b8;font-size:.75rem;line-height:1.6}
@media(max-width:768px){.section-52w{grid-template-columns:1fr}.hdr-inner{flex-direction:column;gap:.6rem;align-items:flex-start}}
</style></head><body>

<div class="hdr">
  <div class="hdr-inner">
    <div><h1>Market Regime Analyzer &mdash; {{ market_label }}</h1>
      <p>{{ market_title }} &mdash; Breadth + RS + 52W + Gemini AI &middot; Bench: {{ bench_name }}</p></div>
    <div class="hdr-btns">
      <a class="hdr-btn" href="/toggle_beta?market={{ market | lower }}">Beta: {{ 'ON' if beta_adj else 'OFF' }}</a>
      <a class="hdr-btn" href="/reset">재분석</a>
    </div>
  </div>
  <div class="mkt-tabs">
    {% for mt in market_tabs %}
    <a class="mkt-tab{% if mt.active %} on{% endif %}" href="{{ mt.url }}">
      {{ mt.label }}
      {% if mt.ready %}<span class="ind">READY</span>
      {% elif mt.code == 'KR' and kr_error %}<span class="ind bad">FAILED</span>
      {% else %}<span class="ind na">N/A</span>{% endif %}
    </a>
    {% endfor %}
  </div>
</div>

<div class="wrap">
  {% if kr_error %}
  <div class="kr-banner">
    <strong>KR 시장 분석 실패</strong> &mdash; {{ kr_error }}
    <span style="color:#92400e;opacity:.8"> (US 결과는 정상적으로 표시됩니다.)</span>
  </div>
  {% endif %}
  <div class="tabs">
    <button class="tab on" data-tab="market" onclick="switchTab('market',this)">시장 분석</button>
    <button class="tab" data-tab="narratives" onclick="switchTab('narratives',this)">Narratives</button>
    <button class="tab" data-tab="sector" onclick="switchTab('sector',this)">{{ etf_label }} ({{ s_total }})</button>
    <button class="tab" data-tab="stocks" onclick="switchTab('stocks',this)">{{ constituents_label }} ({{ breadth.total }})</button>
  </div>

  <!-- ═══ TAB 1: MARKET ═══ -->
  <div id="tab-market" class="tab-content active">
    <div class="cards">
      <div class="card"><div class="lb">{{ constituents_label }} Screened</div><div class="vl">{{ breadth.total }}</div></div>
      <div class="card c-ld"><div class="lb">{{ constituents_label }} Leaders</div><div class="vl">{{ breadth.n_leader }}</div></div>
      <div class="card"><div class="lb">% Above 50 MA</div><div class="vl">{{ breadth.pct_above_50ma }}%</div></div>
      <div class="card"><div class="lb">% Above 200 MA</div><div class="vl">{{ breadth.pct_above_200ma }}%</div></div>
      <div class="card"><div class="lb">A/D Ratio</div><div class="vl" style="color:{{ '#059669' if breadth.ad_ratio >= 1 else '#dc2626' }}">{{ breadth.ad_ratio }}</div></div>
      <div class="card c-52h"><div class="lb">52W Highs</div><div class="vl">{{ breadth.n_52h }}</div></div>
      <div class="card c-52l"><div class="lb">52W Lows</div><div class="vl">{{ breadth.n_52l }}</div></div>
      <div class="card"><div class="lb">Updated</div><div class="vl" style="font-size:.85rem">{{ updated }}</div></div>
    </div>

    {% if breadth.leader_sectors %}
    <div class="ldist">
      <div class="ldist-hdr">
        <h3 id="ldist-title">Leader Distribution by Industry ({{ constituents_label }})</h3>
        <div class="ldist-toggle">
          <button class="ldist-tbtn on" data-mode="industry" onclick="setLdistMode('industry',this)">Industry ({{ breadth.leader_industries|length }})</button>
          <button class="ldist-tbtn" data-mode="sector" onclick="setLdistMode('sector',this)">Sector ({{ breadth.leader_sectors|length }})</button>
        </div>
      </div>

      <div id="ldist-industry" class="ldist-list">
        {% for ind, cnt in breadth.leader_industries.items() %}
        <div class="ldist-row">
          <span class="ldist-label" title="{{ ind }}">{{ ind }}</span>
          <div class="ldist-bar-wrap"><div class="ldist-bar" style="width:{{ (cnt / breadth.n_leader * 100) | round(1) }}%"></div></div>
          <span class="ldist-cnt">{{ cnt }}</span>
        </div>
        {% endfor %}
      </div>

      <div id="ldist-sector" class="ldist-list" style="display:none">
        {% for sec, cnt in breadth.leader_sectors.items() %}
        <div class="ldist-row">
          <span class="ldist-label" title="{{ sec }}">{{ sec }}</span>
          <div class="ldist-bar-wrap"><div class="ldist-bar" style="width:{{ (cnt / breadth.n_leader * 100) | round(1) }}%"></div></div>
          <span class="ldist-cnt">{{ cnt }}</span>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endif %}

    <div class="section-52w">
      <div class="w52-card w52-high"><h3>&#9650; 52-Week Highs ({{ etf_label }})</h3><div class="w52-items">
        {% if sector_highs %}{% for h in sector_highs %}<div class="w52-item"><span class="tk">{{ h.ticker }}</span><span class="nm">{{ h.name }}</span></div>{% endfor %}
        {% else %}<span class="w52-empty">None</span>{% endif %}</div></div>
      <div class="w52-card w52-low"><h3>&#9660; 52-Week Lows ({{ etf_label }})</h3><div class="w52-items">
        {% if sector_lows %}{% for l in sector_lows %}<div class="w52-item"><span class="tk">{{ l.ticker }}</span><span class="nm">{{ l.name }}</span></div>{% endfor %}
        {% else %}<span class="w52-empty">None</span>{% endif %}</div></div>
    </div>

    <div class="ai-section">
      <div class="ai-header"><h3>AI Market Regime Analysis &mdash; {{ market_label }} (Gemini 3.1 Pro &middot; on-demand)</h3>
        <div class="ai-meta">
          <span class="ai-updated" id="aiUpdated">{{ gemini_updated or '' }}</span>
          <button class="ai-refresh" id="aiBtn" onclick="generateMarketAI()">{% if gemini_md %}Refresh{% else %}생성하기{% endif %}</button>
        </div>
      </div>
      <div id="ai-content" class="ai-content"></div>
    </div>
  </div>

  <!-- ═══ TAB: NARRATIVES (Early Trend Curator) ═══ -->
  <div id="tab-narratives" class="tab-content">
    <div class="narr-bar">
      <div>
        <h3>Narrative Curator &mdash; {{ market_label }} Early Trend 종목 (총 {{ narr_candidates|length }}개)</h3>
        <p>아래 카드를 클릭하면 해당 종목의 최근 {{ narr_lookback }}일 이슈 + 해설을 생성합니다.
          {% if market == 'KR' %}(Stage 1: <strong>네이버 뉴스 API</strong> + Gemini Flash 정리 &middot; Stage 2: 3.1 Pro + Google Search)
          {% else %}(Stage 1: 3 Flash Preview + Google Search &middot; Stage 2: 3.1 Pro)
          {% endif %} 종목당 약 30~90초 소요.</p>
      </div>
      {% if narr_candidates|length > 0 %}
      <div class="narr-batch">
        <label class="narr-batch-opt"><input type="checkbox" id="narrBatchForce"> 캐시 무시 (전체 재생성)</label>
        <button class="narr-batch-btn" id="narrBatchBtn" onclick="runBatchNarratives()">▶ 순차 생성 (전체)</button>
        <button class="narr-batch-stop" id="narrBatchStop" onclick="stopBatchNarratives()" style="display:none">■ 중지</button>
        <div class="narr-batch-prog" id="narrBatchProg" style="display:none">
          <div class="narr-batch-prog-bar"><div class="narr-batch-prog-fill" id="narrBatchProgFill"></div></div>
          <div class="narr-batch-prog-txt" id="narrBatchProgTxt"></div>
        </div>
      </div>
      {% endif %}
    </div>

    {% if narr_candidates|length == 0 %}
    <div class="narr-empty">
      <h3>현재 Early Trend 상태인 종목이 없습니다</h3>
      <p>시장 상황에 따라 Early Trend 종목이 0개일 수 있습니다. 시장 분석을 다시 실행하거나 잠시 후에 다시 확인해주세요.</p>
    </div>
    {% else %}
    <div class="narr-grid">
      {% for c in narr_candidates %}
      <div class="narr-card" data-tk="{{ c.ticker }}" onclick="openNarrative('{{ c.ticker }}')">
        <div class="narr-card-hdr">
          <div class="l">
            <div class="narr-card-name" title="{{ c.name }}">{{ c.name }}</div>
            <div class="narr-card-tk">{{ c.ticker }}{% if c.kind == 'etf' %}<span class="kind e">ETF</span>{% else %}<span class="kind s">STOCK</span>{% endif %}</div>
            <div class="narr-card-sec">{% if c.industry and c.industry != c.sector %}{{ c.industry }} <span style="color:#cbd5e1">·</span> {% endif %}{{ c.sector }}</div>
          </div>
          <div class="narr-card-rs">
            <span class="{{ 'p' if c.rs_5 is not none and c.rs_5 > 0 else 'n' if c.rs_5 is not none else '' }}">5d {% if c.rs_5 is none %}—{% else %}{{ '%+.2f'|format(c.rs_5) }}%{% endif %}</span>
            <span class="{{ 'p' if c.rs_20 is not none and c.rs_20 > 0 else 'n' if c.rs_20 is not none else '' }}">20d {% if c.rs_20 is none %}—{% else %}{{ '%+.2f'|format(c.rs_20) }}%{% endif %}</span>
            <span class="{{ 'p' if c.rs_60 is not none and c.rs_60 > 0 else 'n' if c.rs_60 is not none else '' }}">60d {% if c.rs_60 is none %}—{% else %}{{ '%+.2f'|format(c.rs_60) }}%{% endif %}</span>
          </div>
        </div>
        <div class="narr-card-body">
          <div class="narr-card-rough" data-status="idle">클릭하여 내러티브 생성</div>
        </div>
        <div class="narr-card-meta">
          <span class="narr-card-status" data-status="idle">미생성</span>
          <span class="narr-open">열기 →</span>
        </div>
      </div>
      {% endfor %}
    </div>
    {% endif %}
  </div>

  <!-- ═══ TAB 2: SECTOR ETF ═══ -->
  <div id="tab-sector" class="tab-content">
    <div class="cards">
      <div class="card"><div class="lb">Sectors</div><div class="vl">{{ s_total }}</div></div>
      <div class="card c-ld"><div class="lb">Leaders</div><div class="vl">{{ s_n_leader }}</div></div>
      <div class="card c-et"><div class="lb">Early Trend</div><div class="vl">{{ s_n_early }}</div></div>
      <div class="card c-52h"><div class="lb">52W High</div><div class="vl">{{ s_n_52h }}</div></div>
      <div class="card c-52l"><div class="lb">52W Low</div><div class="vl">{{ s_n_52l }}</div></div>
    </div>
    <div class="filters" id="sf">
      <button class="fbtn on" onclick="filt('tblS','all',this)">All</button>
      <button class="fbtn" onclick="filt('tblS','LEADER',this)">Leaders</button>
      <button class="fbtn" onclick="filt('tblS','Early Trend',this)">Early Trend</button>
      <button class="fbtn" onclick="filt('tblS','Not Leading',this)">Not Leading</button>
    </div>
    <div class="srch"><input type="text" placeholder="Search sector ETF..." oninput="search('tblS',this.value)"></div>
    <div class="tw"><table id="tblS"><thead><tr>
      <th onclick="srt('tblS',0,1)">#<span class="si">⇅</span></th>
      <th onclick="srt('tblS',1,0)">Ticker<span class="si">⇅</span></th>
      <th onclick="srt('tblS',2,1)">Score<span class="si">⇅</span></th>
      <th onclick="srt('tblS',3,0)">ETF Name<span class="si">⇅</span></th>
      <th onclick="srt('tblS',4,0)">Category<span class="si">⇅</span></th>
      <th onclick="srt('tblS',5,1)">RS 5d<span class="si">⇅</span></th>
      <th onclick="srt('tblS',6,1)">RS 10d<span class="si">⇅</span></th>
      <th onclick="srt('tblS',7,1)">RS 20d<span class="si">⇅</span></th>
      <th onclick="srt('tblS',8,1)">RS 60d<span class="si">⇅</span></th>
      <th onclick="srt('tblS',9,1)">IR 20d<span class="si">⇅</span></th>
      <th onclick="srt('tblS',10,1)">Alpha%<span class="si">⇅</span></th>
      <th onclick="srt('tblS',11,1)">vs52H<span class="si">⇅</span></th>
      <th onclick="srt('tblS',12,0)">52W<span class="si">⇅</span></th>
      <th onclick="srt('tblS',13,0)">Status<span class="si">⇅</span></th>
    </tr></thead><tbody>
    {% for r in sector_rows %}
    <tr data-st="{{ r.status }}" data-sk="{{ r.ticker }}|{{ r.name }}|{{ r.category }}|{{ r.industry }}">
      <td>{{ loop.index }}</td><td><strong class="tklink" onclick="showChart('{{ r.ticker }}')">{{ r.ticker }}</strong></td>
      <td style="font-weight:600">{{ "%.2f"|format(r.composite) if r.composite is not none else '--' }}</td>
      <td class="tklink" onclick="showChart('{{ r.ticker }}')">{{ r.name }}</td><td>{{ r.category }}</td>
      <td class="{{ 'pos' if r.rs_5 and r.rs_5 > 0 else 'neg' }}">{{ "%.2f"|format(r.rs_5) if r.rs_5 is not none else '—' }}</td>
      <td class="{{ 'pos' if r.rs_10 and r.rs_10 > 0 else 'neg' }}">{{ "%.2f"|format(r.rs_10) if r.rs_10 is not none else '—' }}</td>
      <td class="{{ 'pos' if r.rs_20 and r.rs_20 > 0 else 'neg' }}">{{ "%.2f"|format(r.rs_20) if r.rs_20 is not none else '—' }}</td>
      <td class="{{ 'pos' if r.rs_60 and r.rs_60 > 0 else 'neg' }}">{{ "%.2f"|format(r.rs_60) if r.rs_60 is not none else '—' }}</td>
      <td class="{{ 'pos' if r.ir_20 and r.ir_20 > 0 else 'neg' }}">{{ "%.2f"|format(r.ir_20) if r.ir_20 is not none else '—' }}</td>
      <td class="{{ 'pos' if r.alpha_ann and r.alpha_ann > 0 else 'neg' }}">{{ "%.2f"|format(r.alpha_ann) if r.alpha_ann is not none else '—' }}</td>
      <td class="{{ 'pos' if r.pct_from_high is not none and r.pct_from_high >= 0 else 'neg' }}">{{ "%.1f"|format(r.pct_from_high) if r.pct_from_high is not none else '—' }}%</td>
      <td>{% if r.w52_status=='52W_HIGH' %}<span class="badge b-52h">HIGH</span>{% elif r.w52_status=='52W_LOW' %}<span class="badge b-52l">LOW</span>{% else %}—{% endif %}</td>
      <td>{% if r.status=='LEADER' %}<span class="badge b-ld">LEADER</span>{% elif r.status=='Early Trend' %}<span class="badge b-et">Early</span>{% else %}<span class="badge b-no">—</span>{% endif %}</td>
    </tr>{% endfor %}</tbody></table></div>
  </div>

  <!-- ═══ TAB 3: Constituents (S&P 500 / KOSPI) ═══ -->
  <div id="tab-stocks" class="tab-content">
    <div class="cards">
      <div class="card"><div class="lb">Screened</div><div class="vl">{{ breadth.total }}</div></div>
      <div class="card c-ld"><div class="lb">Leaders</div><div class="vl">{{ breadth.n_leader }}</div></div>
      <div class="card c-et"><div class="lb">Early Trend</div><div class="vl">{{ breadth.n_early }}</div></div>
      <div class="card"><div class="lb">Not Leading</div><div class="vl">{{ breadth.n_not_leading }}</div></div>
      <div class="card c-52h"><div class="lb">52W Highs</div><div class="vl">{{ breadth.n_52h }}</div></div>
      <div class="card c-52l"><div class="lb">52W Lows</div><div class="vl">{{ breadth.n_52l }}</div></div>
    </div>
    <div class="filters" id="stf">
      <button class="fbtn on" onclick="filt('tblK','all',this)">All</button>
      <button class="fbtn" onclick="filt('tblK','LEADER',this)">Leaders ({{ breadth.n_leader }})</button>
      <button class="fbtn" onclick="filt('tblK','Early Trend',this)">Early Trend</button>
      <button class="fbtn" onclick="filt('tblK','Not Leading',this)">Not Leading</button>
    </div>
    <div class="srch"><input type="text" placeholder="Search ticker / company / industry (e.g. semiconductor, biotech)..." oninput="search('tblK',this.value)"></div>
    <div class="tw"><table id="tblK"><thead><tr>
      <th onclick="srt('tblK',0,1)">#<span class="si">⇅</span></th>
      <th onclick="srt('tblK',1,0)">Ticker<span class="si">⇅</span></th>
      <th onclick="srt('tblK',2,1)">Score<span class="si">⇅</span></th>
      <th onclick="srt('tblK',3,0)">Company<span class="si">⇅</span></th>
      <th onclick="srt('tblK',4,0)">Sector / Industry<span class="si">⇅</span></th>
      <th onclick="srt('tblK',5,1)">RS 5d<span class="si">⇅</span></th>
      <th onclick="srt('tblK',6,1)">RS 10d<span class="si">⇅</span></th>
      <th onclick="srt('tblK',7,1)">RS 20d<span class="si">⇅</span></th>
      <th onclick="srt('tblK',8,1)">RS 60d<span class="si">⇅</span></th>
      <th onclick="srt('tblK',9,1)">IR 20d<span class="si">⇅</span></th>
      <th onclick="srt('tblK',10,1)">Alpha%<span class="si">⇅</span></th>
      <th onclick="srt('tblK',11,1)">vs52H<span class="si">⇅</span></th>
      <th onclick="srt('tblK',12,0)">52W<span class="si">⇅</span></th>
      <th onclick="srt('tblK',13,0)">Status<span class="si">⇅</span></th>
    </tr></thead><tbody>
    {% for r in stock_rows %}
    <tr data-st="{{ r.status }}" data-sk="{{ r.ticker }}|{{ r.name }}|{{ r.category }}|{{ r.industry }}">
      <td>{{ loop.index }}</td><td><strong class="tklink" onclick="showChart('{{ r.ticker }}')">{{ r.ticker }}</strong></td>
      <td style="font-weight:600">{{ "%.2f"|format(r.composite) if r.composite is not none else '--' }}</td>
      <td class="tklink" onclick="showChart('{{ r.ticker }}')">{{ r.name }}</td><td><div class="cat-cell"><div class="cat-sec">{{ r.category }}</div>{% if r.industry and r.industry != r.category %}<div class="cat-ind">{{ r.industry }}</div>{% endif %}</div></td>
      <td class="{{ 'pos' if r.rs_5 and r.rs_5 > 0 else 'neg' }}">{{ "%.2f"|format(r.rs_5) if r.rs_5 is not none else '—' }}</td>
      <td class="{{ 'pos' if r.rs_10 and r.rs_10 > 0 else 'neg' }}">{{ "%.2f"|format(r.rs_10) if r.rs_10 is not none else '—' }}</td>
      <td class="{{ 'pos' if r.rs_20 and r.rs_20 > 0 else 'neg' }}">{{ "%.2f"|format(r.rs_20) if r.rs_20 is not none else '—' }}</td>
      <td class="{{ 'pos' if r.rs_60 and r.rs_60 > 0 else 'neg' }}">{{ "%.2f"|format(r.rs_60) if r.rs_60 is not none else '—' }}</td>
      <td class="{{ 'pos' if r.ir_20 and r.ir_20 > 0 else 'neg' }}">{{ "%.2f"|format(r.ir_20) if r.ir_20 is not none else '—' }}</td>
      <td class="{{ 'pos' if r.alpha_ann and r.alpha_ann > 0 else 'neg' }}">{{ "%.2f"|format(r.alpha_ann) if r.alpha_ann is not none else '—' }}</td>
      <td class="{{ 'pos' if r.pct_from_high is not none and r.pct_from_high >= 0 else 'neg' }}">{{ "%.1f"|format(r.pct_from_high) if r.pct_from_high is not none else '—' }}%</td>
      <td>{% if r.w52_status=='52W_HIGH' %}<span class="badge b-52h">HIGH</span>{% elif r.w52_status=='52W_LOW' %}<span class="badge b-52l">LOW</span>{% else %}—{% endif %}</td>
      <td>{% if r.status=='LEADER' %}<span class="badge b-ld">LEADER</span>{% elif r.status=='Early Trend' %}<span class="badge b-et">Early</span>{% else %}<span class="badge b-no">—</span>{% endif %}</td>
    </tr>{% endfor %}</tbody></table></div>
  </div>

</div>

<div id="narrModal" class="narr-modal" onclick="if(event.target===this)closeNarrative()">
  <div class="narr-modal-box">
    <div class="narr-modal-hdr">
      <div>
        <h2 id="narrModalTk"></h2>
        <div class="sub" id="narrModalMeta"></div>
      </div>
      <div style="display:flex;gap:.4rem">
        <button class="narr-btn" id="narrModalGen" onclick="generateOne(false)" style="display:none">생성하기</button>
        <button class="narr-btn-sec" id="narrModalRegen" onclick="generateOne(true)" style="display:none">재생성</button>
        <button class="cbtn acc" onclick="closeNarrative()">Close</button>
      </div>
    </div>
    <div id="narrModalState" class="narr-modal-state"></div>
    <div id="narrModalMetric" class="narr-metric-strip" style="display:none"></div>
    <div id="narrModalChart" class="narr-chart-box" style="display:none">
      <div class="narr-chart-loading" id="narrChartLoading">차트 데이터 로드 중...</div>
      <div class="narr-chart-wrap"><canvas id="narrCvPrice"></canvas></div>
      <div class="narr-rs-wrap"><canvas id="narrCvRS"></canvas></div>
    </div>
    <div id="narrModalHist" class="narr-hist" style="display:none"></div>
    <div class="narr-modal-md" id="narrModalMd" style="display:none"></div>
    <div class="narr-issues" id="narrModalIssues" style="display:none"></div>
  </div>
</div>

<div id="chartModal" class="chart-modal" onclick="if(event.target===this)closeChart()">
  <div class="chart-box" id="chartBox">
    <div class="chart-hdr">
      <h2><span id="chartTicker"></span><span class="sub" id="chartName"></span></h2>
      <div class="chart-btns">
        <button class="cbtn" onclick="resetZoom()" title="Reset Zoom">Reset Zoom</button>
        <button class="cbtn" onclick="toggleFS()" id="fsBtn" title="Fullscreen">Fullscreen</button>
        <button class="cbtn acc" onclick="closeChart()">Close</button>
      </div>
    </div>
    <div class="chart-loading" id="chartLoading">Loading chart data...</div>
    <div class="chart-wrap"><canvas id="cvPrice"></canvas></div>
    <div class="rs-wrap"><canvas id="cvRS"></canvas></div>
    <div class="zoom-hint">Scroll to zoom · Drag to pan · Double-click to reset</div>
  </div>
</div>

<div class="ft">
  <p>RS Stack + Consistency (IR) + Momentum + 52W High/Low + Market Breadth + Gemini AI</p>
  <p>{{ market_label }} &middot; Benchmark: {{ bench_name }} &middot; RS Mode: {{ 'Beta-neutral' if beta_adj else 'Raw excess' }} &middot; {{ updated }}</p>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
const MARKET = {{ market | tojson }};
const BENCH_LABEL = {{ bench_name | tojson }};
const CURRENCY = {{ currency | tojson }};
const Q = '?market=' + MARKET.toLowerCase();
const aiMd={{ gemini_md | tojson }};
if(aiMd){
  try{document.getElementById('ai-content').innerHTML=marked.parse(aiMd)}
  catch(e){document.getElementById('ai-content').innerText=aiMd}
}else{
  document.getElementById('ai-content').innerHTML=
    '<div style="padding:1rem;color:#94a3b8;font-size:.85rem;line-height:1.6">'+
    '시장 Regime 종합 분석은 비용/시간이 큰 작업이라 자동 실행되지 않습니다.<br>'+
    '필요할 때 우측 상단 <strong>"생성하기"</strong> 버튼을 눌러주세요. (약 30~60초 소요)'+
    '</div>';
}

function generateMarketAI(){
  const btn=document.getElementById('aiBtn');
  const content=document.getElementById('ai-content');
  btn.disabled=true;btn.textContent='생성 중...';
  content.innerHTML='<div style="padding:1.2rem;text-align:center;color:#64748b;font-size:.85rem"><div style="display:inline-block;width:24px;height:24px;border:3px solid #e2e8f0;border-top-color:#0d9488;border-radius:50%;animation:narrSpin .8s linear infinite;margin-bottom:.6rem"></div><div>Gemini 3.1 Pro 시장 분석 생성 중 ('+MARKET+') ...</div></div>';
  fetch('/api/market_ai'+Q,{method:'POST'}).then(r=>r.json()).then(d=>{
    btn.disabled=false;btn.textContent='Refresh';
    if(d.ok){
      try{content.innerHTML=marked.parse(d.md)}
      catch(e){content.innerText=d.md}
      document.getElementById('aiUpdated').textContent=d.updated||'';
    }else{
      content.innerHTML='<div style="padding:1rem;color:#dc2626;font-size:.85rem">생성 실패: '+(d.msg||'알 수 없는 오류')+'</div>';
    }
  }).catch(e=>{
    btn.disabled=false;btn.textContent='생성하기';
    content.innerHTML='<div style="padding:1rem;color:#dc2626;font-size:.85rem">네트워크 오류: '+e+'</div>';
  });
}

let _narrCurrent=null;
let _narrHistory=[];   // [{date, generated_at}, ...] 최신순
let _narrCP=null,_narrCR=null,_narrSyncing=false;
let _narrChartLoadedFor=null;  // 마지막으로 차트를 로드한 ticker

function _destroyNarrCharts(){
  if(_narrCP){try{_narrCP.destroy();}catch(e){}_narrCP=null;}
  if(_narrCR){try{_narrCR.destroy();}catch(e){}_narrCR=null;}
}

function _resetModalSections(){
  document.getElementById('narrModalMd').style.display='none';
  document.getElementById('narrModalIssues').style.display='none';
  document.getElementById('narrModalMetric').style.display='none';
  document.getElementById('narrModalMetric').innerHTML='';
  document.getElementById('narrModalHist').style.display='none';
  document.getElementById('narrModalHist').innerHTML='';
  document.getElementById('narrModalGen').style.display='none';
  document.getElementById('narrModalRegen').style.display='none';
}

function openNarrative(tk){
  _narrCurrent=tk;
  _narrHistory=[];
  _destroyNarrCharts();
  _narrChartLoadedFor=null;
  document.getElementById('narrModalTk').textContent=tk;  // 로딩 중에는 ticker 만 표시
  document.getElementById('narrModalMeta').innerHTML='';
  _resetModalSections();
  document.getElementById('narrModalState').className='narr-modal-state';
  document.getElementById('narrModalState').innerHTML='<div class="spinner"></div><div>캐시 확인 중...</div>';
  document.getElementById('narrModal').classList.add('open');

  loadNarrChart(tk);

  fetch('/api/narratives/'+tk+Q).then(r=>r.json()).then(d=>{
    _narrHistory = Array.isArray(d.history) ? d.history : [];
    if(d.cached&&d.narrative){
      renderNarrative(d.narrative);
    }else{
      const stage1Hint = MARKET==='KR'
        ? '네이버 뉴스 API 로 최근 20일 기사를 수집하고 도메인 비전공자용 해설을 작성합니다.'
        : '최근 20일 이슈를 웹에서 검색하고 도메인 비전공자용 해설을 작성합니다.';
      document.getElementById('narrModalState').innerHTML=
        '<div style="font-size:1rem;font-weight:600;color:#1e293b;margin-bottom:.4rem">아직 생성되지 않은 종목입니다</div>'+
        '<div class="hint">'+stage1Hint+' 약 30~90초 소요.</div>'+
        '<div class="hint" style="margin-top:1rem">아래 "생성하기" 버튼을 눌러주세요.</div>';
      document.getElementById('narrModalGen').style.display='inline-block';
      document.getElementById('narrModalGen').disabled=false;
      document.getElementById('narrModalGen').textContent='생성하기';
      // history 가 비어 있지 않을 수도 있음 (DB 에는 있으나 ?date 필요한 경우)
      // → 미생성 안내와 함께 history selector 표시
      if(_narrHistory.length>0) renderHistory(null);
    }
  }).catch(()=>{
    document.getElementById('narrModalState').className='narr-modal-state error';
    document.getElementById('narrModalState').innerHTML='데이터 로드 실패';
  });
}

function loadNarrativeAt(date){
  const tk=_narrCurrent;
  if(!tk||!date)return;
  _resetModalSections();
  document.getElementById('narrModalState').className='narr-modal-state';
  document.getElementById('narrModalState').innerHTML='<div class="spinner"></div><div>'+escapeHtml(date)+' 일자 내러티브 불러오는 중...</div>';
  fetch('/api/narratives/'+tk+Q+'&date='+encodeURIComponent(date)).then(r=>r.json()).then(d=>{
    _narrHistory = Array.isArray(d.history) ? d.history : _narrHistory;
    if(d.narrative){
      renderNarrative(d.narrative);
    }else{
      document.getElementById('narrModalState').className='narr-modal-state error';
      document.getElementById('narrModalState').innerHTML='해당 일자 데이터가 없습니다';
    }
  }).catch(()=>{
    document.getElementById('narrModalState').className='narr-modal-state error';
    document.getElementById('narrModalState').innerHTML='데이터 로드 실패';
  });
}

function generateOne(force){
  const tk=_narrCurrent;
  if(!tk)return;
  document.getElementById('narrModalGen').style.display='none';
  document.getElementById('narrModalRegen').style.display='none';
  document.getElementById('narrModalMd').style.display='none';
  document.getElementById('narrModalIssues').style.display='none';
  document.getElementById('narrModalState').className='narr-modal-state';
  const stage1Label = MARKET==='KR' ? 'Naver News + Flash 정리' : 'Flash + Search';
  document.getElementById('narrModalState').innerHTML=
    '<div class="spinner"></div>'+
    '<div>Stage 1 ('+stage1Label+') → Stage 2 (Pro) 생성 중...</div>'+
    '<div class="hint">최대 2분 정도 걸릴 수 있습니다. 창을 닫지 마세요.</div>';
  setCardStatus(tk,'loading','생성 중...');

  const url='/api/narratives/'+tk+Q+(force?'&force=1':'');
  fetch(url,{method:'POST'}).then(r=>r.json()).then(d=>{
    if(d.ok&&d.narrative){
      renderNarrative(d.narrative);
      setCardStatus(tk,'ready','생성 완료');
    }else{
      document.getElementById('narrModalState').className='narr-modal-state error';
      document.getElementById('narrModalState').innerHTML=
        '<div style="font-weight:600">생성 실패</div>'+
        '<div class="hint">'+escapeHtml(d.msg||'알 수 없는 오류')+'</div>';
      setCardStatus(tk,'error','오류');
      document.getElementById('narrModalGen').style.display='inline-block';
      document.getElementById('narrModalGen').textContent='다시 시도';
    }
  }).catch(e=>{
    document.getElementById('narrModalState').className='narr-modal-state error';
    document.getElementById('narrModalState').innerHTML=
      '<div style="font-weight:600">네트워크 오류</div>'+
      '<div class="hint">'+escapeHtml(String(e))+'</div>';
    setCardStatus(tk,'error','오류');
    document.getElementById('narrModalGen').style.display='inline-block';
    document.getElementById('narrModalGen').textContent='다시 시도';
  });
}

function loadNarrChart(tk){
  const box=document.getElementById('narrModalChart');
  const loading=document.getElementById('narrChartLoading');
  box.style.display='block';
  loading.style.display='block';
  loading.textContent='차트 데이터 로드 중...';
  document.querySelector('#narrModalChart .narr-chart-wrap').style.display='none';
  document.querySelector('#narrModalChart .narr-rs-wrap').style.display='none';
  fetch('/api/chart/'+tk+Q).then(r=>r.json()).then(d=>{
    if(_narrCurrent!==tk)return;  // 중간에 다른 종목으로 바뀐 경우 무시
    if(d.error){loading.textContent=d.error;return;}
    loading.style.display='none';
    document.querySelector('#narrModalChart .narr-chart-wrap').style.display='block';
    document.querySelector('#narrModalChart .narr-rs-wrap').style.display='block';
    renderNarrCharts(d);
    _narrChartLoadedFor=tk;
  }).catch(()=>{loading.textContent='차트 데이터 로드 실패';});
}

function _narrSyncFrom(src){
  if(_narrSyncing)return;_narrSyncing=true;
  const tgt=src===_narrCP?_narrCR:_narrCP;
  if(!tgt){_narrSyncing=false;return;}
  const sx=src.scales.x;
  tgt.scales.x.options.min=sx.min;tgt.scales.x.options.max=sx.max;
  tgt.update('none');_narrSyncing=false;
}
function _narrZoomCfg(){return{
  zoom:{wheel:{enabled:true,speed:.08},pinch:{enabled:true},mode:'x',
    onZoomComplete:function(ctx){_narrSyncFrom(ctx.chart);}},
  pan:{enabled:true,mode:'x',threshold:5,
    onPanComplete:function(ctx){_narrSyncFrom(ctx.chart);}},
  limits:{x:{minRange:20}}
};}

function renderNarrCharts(d){
  _destroyNarrCharts();
  const lbl=d.dates;
  const skip=Math.max(1,Math.floor(lbl.length/60));
  const xCfg={type:'category',ticks:{maxTicksLimit:12,maxRotation:0,callback:function(v,i){return i%skip===0?lbl[i].slice(0,7):'';}},grid:{display:false}};
  const zp=_narrZoomCfg();
  const cur=d.currency||CURRENCY;
  const benchLbl=d.bench_name||BENCH_LABEL;

  _narrCP=new Chart(document.getElementById('narrCvPrice'),{type:'line',data:{
    labels:lbl,datasets:[
      {label:d.ticker,data:d.price,borderColor:'#2563eb',borderWidth:2,pointRadius:0,tension:.1,yAxisID:'y'},
      {label:benchLbl,data:d.spy,borderColor:'#f59e0b',borderWidth:1.5,pointRadius:0,borderDash:[5,3],tension:.1,yAxisID:'y1'}
    ]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
    plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:8,font:{size:11}}},
      title:{display:true,text:d.ticker+' vs '+benchLbl,font:{size:12}},
      tooltip:{callbacks:{label:function(ctx){return ctx.dataset.label+': '+cur+ctx.parsed.y.toLocaleString(undefined,{maximumFractionDigits:2});}}},
      zoom:zp},
    scales:{x:xCfg,
      y:{position:'left',title:{display:true,text:d.ticker+' ('+cur+')',color:'#2563eb',font:{weight:'bold',size:10}},
        ticks:{color:'#2563eb',font:{size:10}},grid:{color:'#f1f5f9'}},
      y1:{position:'right',title:{display:true,text:benchLbl+' ('+cur+')',color:'#f59e0b',font:{weight:'bold',size:10}},
        ticks:{color:'#f59e0b',font:{size:10}},grid:{drawOnChartArea:false}}
    }}});

  _narrCR=new Chart(document.getElementById('narrCvRS'),{type:'line',data:{
    labels:lbl,datasets:[
      {label:'RS 20d (%)',data:d.rs_20,borderColor:'#059669',borderWidth:1.8,pointRadius:0,tension:.1,
        fill:{target:{value:0},above:'rgba(5,150,105,.12)',below:'rgba(220,38,38,.1)'}},
      {label:'RS 60d (%)',data:d.rs_60,borderColor:'#6366f1',borderWidth:1.5,pointRadius:0,tension:.1,borderDash:[5,2]},
      {label:'',data:lbl.map(()=>0),borderColor:'#cbd5e1',borderWidth:1,pointRadius:0,borderDash:[3,3]}
    ]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
    spanGaps:true,
    plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:8,font:{size:10},
        filter:function(item){return item.text!=='';}}},
      title:{display:true,text:'Relative Strength vs '+benchLbl+' (Rolling Excess Return %)',font:{size:11}},
      tooltip:{callbacks:{label:function(ctx){if(!ctx.dataset.label)return null;
        return ctx.dataset.label+': '+(ctx.parsed.y!==null?ctx.parsed.y.toFixed(2)+'%':'N/A');}}},
      zoom:zp},
    scales:{x:xCfg,y:{ticks:{font:{size:10}},grid:{color:'#f1f5f9'}}}}});
}

function renderNarrative(it){
  const m=it.rs_metrics||{};
  const rsTxt=v=>v==null?'—':(v>=0?'+':'')+v.toFixed(2)+'%';
  const rsCls=v=>v==null?'muted':(v>=0?'p':'n');
  // 헤더: 종목명을 메인 (h2), 티커는 메타라인으로 빼서 가독성 ↑ (특히 KR 숫자 티커)
  const _name = it.name || it.ticker || '';
  document.getElementById('narrModalTk').textContent = _name;
  document.getElementById('narrModalMeta').innerHTML=
    `<span style="font-family:monospace;color:#475569;font-weight:600">${escapeHtml(it.ticker||'')}</span> &middot; `+
    `${escapeHtml(it.sector||'')} &middot; 생성: ${it.generated_at||''}`;

  // ── 상단 메트릭 스트립 (주가 + RS20 + RS60 + RS5/IR20) ──
  const priceTxt = (it.price==null) ? '—'
    : CURRENCY + Number(it.price).toLocaleString(undefined,{maximumFractionDigits:2});
  const stripEl=document.getElementById('narrModalMetric');
  stripEl.innerHTML =
    `<div class="narr-metric"><div class="lb">주가 (최근 종가)</div>`+
      `<div class="vl">${escapeHtml(priceTxt)}</div>`+
      `<div class="sub">${escapeHtml(BENCH_LABEL)} 대비 RS</div></div>`+
    `<div class="narr-metric"><div class="lb">RS 20d</div>`+
      `<div class="vl ${rsCls(m.rs_20)}">${rsTxt(m.rs_20)}</div></div>`+
    `<div class="narr-metric"><div class="lb">RS 60d</div>`+
      `<div class="vl ${rsCls(m.rs_60)}">${rsTxt(m.rs_60)}</div></div>`+
    `<div class="narr-metric"><div class="lb">RS 5d</div>`+
      `<div class="vl ${rsCls(m.rs_5)}">${rsTxt(m.rs_5)}</div></div>`+
    `<div class="narr-metric"><div class="lb">IR 20d</div>`+
      `<div class="vl ${rsCls(m.ir_20)}">${m.ir_20==null?'—':m.ir_20.toFixed(2)}</div></div>`;
  stripEl.style.display='flex';

  // ── 일자 선택기 (DB 누적 history) ──
  renderHistory(it.date||null);

  const md=it.narrative_md||'(해설 없음)';
  const mdEl=document.getElementById('narrModalMd');
  try{mdEl.innerHTML=marked.parse(md);}catch(e){mdEl.innerText=md;}
  mdEl.style.display='block';
  renderIssues(it.issues||[]);
  document.getElementById('narrModalIssues').style.display='block';
  document.getElementById('narrModalState').innerHTML='';
  document.getElementById('narrModalRegen').style.display='inline-block';
  document.getElementById('narrModalRegen').disabled=false;
  document.getElementById('narrModalRegen').textContent='재생성';
  document.getElementById('narrModalGen').style.display='none';
  if(it.ticker)setCardStatus(it.ticker,'ready','생성 완료');
}

function renderHistory(currentDate){
  const box=document.getElementById('narrModalHist');
  if(!_narrHistory||_narrHistory.length===0){
    box.style.display='none';box.innerHTML='';return;
  }
  const latest = _narrHistory[0]?_narrHistory[0].date:null;
  let opts='';
  _narrHistory.forEach(h=>{
    const sel=(currentDate&&h.date===currentDate)?' selected':'';
    const tag=(h.date===latest)?' (최신)':'';
    opts += `<option value="${escapeHtml(h.date)}"${sel}>${escapeHtml(h.date)}${tag} &middot; ${escapeHtml(h.generated_at||'')}</option>`;
  });
  const isOld = currentDate && latest && currentDate!==latest;
  box.innerHTML =
    `<label>일자별 누적 (${_narrHistory.length}건):</label>`+
    `<select onchange="loadNarrativeAt(this.value)">${opts}</select>`+
    (isOld?`<span class="badge-old">과거 버전 보는 중</span>`:'');
  box.style.display='flex';
}

function renderIssues(issues){
  const box=document.getElementById('narrModalIssues');
  if(!issues||issues.length===0){box.innerHTML='<h4>발견된 이슈 (0건)</h4><p style="color:#94a3b8;font-size:.8rem">최근 lookback 기간 내 material 이슈가 발견되지 않았습니다.</p>';return;}
  let html='<h4>발견된 이슈 ('+issues.length+'건)</h4>';
  issues.forEach(i=>{
    const imp=i.impact||'neutral';
    html+=`
      <div class="narr-issue">
        <div class="narr-issue-meta">
          <span class="narr-issue-date">${escapeHtml(i.date||'')}</span>
          <span class="narr-issue-cat">${escapeHtml(i.category||'')}</span>
          <span class="narr-issue-imp ${imp}">${escapeHtml(imp)}</span>
        </div>
        <div class="narr-issue-headline">${escapeHtml(i.headline||'')}</div>
        <div class="narr-issue-summary">${escapeHtml(i.summary||'')}</div>
        ${i.source?'<div class="narr-issue-src">출처: '+escapeHtml(i.source)+'</div>':''}
      </div>`;
  });
  box.innerHTML=html;
}

function setCardStatus(tk,status,label){
  const card=document.querySelector('.narr-card[data-tk="'+tk+'"]');
  if(!card)return;
  const st=card.querySelector('.narr-card-status');
  const rough=card.querySelector('.narr-card-rough');
  if(st){st.dataset.status=status;st.textContent=label;}
  if(rough){
    rough.dataset.status=status;
    if(status==='loading')rough.textContent='이슈 검색 + 해설 생성 중...';
    else if(status==='queued')rough.textContent='순차 생성 대기열에 있음...';
    else if(status==='ready')rough.textContent='클릭하여 결과 보기';
    else if(status==='error')rough.textContent='오류 발생 — 다시 시도하려면 클릭';
    else if(status==='skipped')rough.textContent='이미 생성됨 (스킵됨)';
  }
}

// ── 순차 생성 (위에서부터 카드 순서대로 단일 생성) ──
let _narrBatchRun = false;
let _narrBatchAbort = false;

function _setBatchUI(running, total){
  const btn=document.getElementById('narrBatchBtn');
  const stopBtn=document.getElementById('narrBatchStop');
  const prog=document.getElementById('narrBatchProg');
  if(running){
    btn.disabled=true;btn.textContent='실행 중...';
    stopBtn.style.display='inline-block';
    prog.style.display='flex';
    _updateBatchProgress(0,total,'준비 중...');
  }else{
    btn.disabled=false;btn.textContent='▶ 순차 생성 (전체)';
    stopBtn.style.display='none';
  }
}

function _updateBatchProgress(done,total,label){
  const pct = total>0 ? Math.round(done/total*100) : 0;
  const fill=document.getElementById('narrBatchProgFill');
  const txt=document.getElementById('narrBatchProgTxt');
  if(fill)fill.style.width=pct+'%';
  if(txt)txt.textContent=`[${done}/${total}] ${pct}% — ${label||''}`;
}

function stopBatchNarratives(){
  if(!_narrBatchRun)return;
  _narrBatchAbort = true;
  const txt=document.getElementById('narrBatchProgTxt');
  if(txt)txt.textContent += '  (중지 요청 — 현재 진행 중인 항목 완료 후 정지)';
}

async function runBatchNarratives(){
  if(_narrBatchRun)return;
  const force = document.getElementById('narrBatchForce').checked;
  // 카드를 DOM 순서 (위→아래) 대로 수집
  const cards = Array.from(document.querySelectorAll('.narr-card'));
  const tickers = cards.map(c=>c.dataset.tk).filter(Boolean);
  if(tickers.length===0){alert('Early Trend 종목이 없습니다.');return;}
  const total = tickers.length;
  if(!confirm(`총 ${total}개 종목을 순차 생성합니다. 종목당 30~90초 (${force?'캐시 무시':'캐시 있으면 스킵'}) — 약 ${Math.ceil(total*60/60)}분 예상.\n진행하시겠습니까?`))return;

  _narrBatchRun=true;_narrBatchAbort=false;
  _setBatchUI(true,total);

  // 시작 시 모든 미완료 카드를 'queued' 로 표시
  if(!force){
    for(const tk of tickers){
      const card=document.querySelector('.narr-card[data-tk="'+tk+'"]');
      const st=card&&card.querySelector('.narr-card-status');
      if(st && st.dataset.status!=='ready'){setCardStatus(tk,'queued','대기 중');}
    }
  }else{
    for(const tk of tickers){setCardStatus(tk,'queued','대기 중 (재생성)');}
  }

  let done=0, ok=0, fail=0, skip=0;
  for(let i=0;i<tickers.length;i++){
    if(_narrBatchAbort)break;
    const tk = tickers[i];
    done = i;
    _updateBatchProgress(done,total,`${tk} 처리 중... (성공 ${ok} / 실패 ${fail} / 스킵 ${skip})`);

    // 캐시 체크 (force 가 아닐 때만)
    if(!force){
      try{
        const r = await fetch('/api/narratives/'+tk+Q);
        const d = await r.json();
        if(d.cached){
          setCardStatus(tk,'ready','생성 완료');
          skip++;done++;
          _updateBatchProgress(done,total,`${tk} 스킵 (캐시 있음)`);
          continue;
        }
      }catch(e){/* 무시하고 생성 시도 */}
    }

    setCardStatus(tk,'loading','생성 중...');
    try{
      const url = '/api/narratives/'+tk+Q+(force?'&force=1':'');
      const r = await fetch(url,{method:'POST'});
      const d = await r.json();
      if(d.ok && d.narrative){
        setCardStatus(tk,'ready','생성 완료');
        ok++;
        // 모달이 이 종목으로 열려 있으면 같이 갱신
        if(_narrCurrent===tk){
          _narrHistory = Array.isArray(d.history) ? d.history : _narrHistory;
          renderNarrative(d.narrative);
        }
      }else{
        setCardStatus(tk,'error','오류');
        fail++;
      }
    }catch(e){
      setCardStatus(tk,'error','네트워크 오류');
      fail++;
    }
    done++;
    _updateBatchProgress(done,total,`완료: ${tk} (성공 ${ok} / 실패 ${fail} / 스킵 ${skip})`);
  }

  // 정리: 남아 있는 'queued' 표시 (중지된 경우) 를 idle 로 복귀
  document.querySelectorAll('.narr-card-status[data-status="queued"]').forEach(el=>{
    const card = el.closest('.narr-card');
    if(card)setCardStatus(card.dataset.tk,'idle','미생성');
  });

  const finalLabel = _narrBatchAbort
    ? `중지됨 — 처리 ${done}/${total} (성공 ${ok} / 실패 ${fail} / 스킵 ${skip})`
    : `완료 — ${total}개 처리 (성공 ${ok} / 실패 ${fail} / 스킵 ${skip})`;
  _updateBatchProgress(done,total,finalLabel);
  _narrBatchRun=false;_narrBatchAbort=false;
  _setBatchUI(false,total);
}

function closeNarrative(){
  document.getElementById('narrModal').classList.remove('open');
  _destroyNarrCharts();
  _narrChartLoadedFor=null;
  _narrCurrent=null;
}

function setLdistMode(mode,btn){
  document.querySelectorAll('.ldist-tbtn').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  const sec=document.getElementById('ldist-sector');
  const ind=document.getElementById('ldist-industry');
  const title=document.getElementById('ldist-title');
  const cl = {{ constituents_label | tojson }};
  if(mode==='industry'){
    sec.style.display='none';ind.style.display='block';
    title.textContent='Leader Distribution by Industry ('+cl+')';
  }else{
    sec.style.display='block';ind.style.display='none';
    title.textContent='Leader Distribution by Sector ('+cl+')';
  }
}

function escapeHtml(s){if(s==null)return '';return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

document.addEventListener('keydown',e=>{
  if(e.key==='Escape'&&document.getElementById('narrModal').classList.contains('open')){closeNarrative();}
});

// 페이지 로드 시 이미 캐시된 narrative가 있는 카드에 "생성 완료" 라벨 표시
window.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('.narr-card').forEach(card=>{
    const tk=card.dataset.tk;if(!tk)return;
    fetch('/api/narratives/'+tk+Q).then(r=>r.json()).then(d=>{
      if(d.cached)setCardStatus(tk,'ready','생성 완료');
    }).catch(()=>{});
  });
});

function switchTab(t,btn){
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('on'));
  document.getElementById('tab-'+t).classList.add('active');
  btn.classList.add('on');
  history.replaceState(null,'','#'+t);
}

window.addEventListener('DOMContentLoaded',()=>{
  const h=location.hash.slice(1);
  if(h){const b=document.querySelector('.tab[data-tab="'+h+'"]');if(b)switchTab(h,b);}
});

let SS={};
function srt(tid,ci,num){
  if(!SS[tid])SS[tid]={col:-1,asc:true};
  const cs=SS[tid],tb=document.querySelector('#'+tid+' tbody'),rows=Array.from(tb.rows);
  const asc=cs.col===ci?!cs.asc:!num;SS[tid]={col:ci,asc};
  rows.sort((a,b)=>{let va=a.cells[ci].textContent.trim(),vb=b.cells[ci].textContent.trim();
    if(num){va=parseFloat(va)||-1e9;vb=parseFloat(vb)||-1e9;return asc?va-vb:vb-va;}
    return asc?va.localeCompare(vb):vb.localeCompare(va);});
  rows.forEach(r=>tb.appendChild(r));
}
function filt(tid,st,btn){
  btn.parentElement.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('on'));btn.classList.add('on');
  document.querySelectorAll('#'+tid+' tbody tr').forEach(r=>{
    if(st==='all'){r.style.display='';return;}
    r.style.display=r.dataset.st===st?'':'none';});
}
function search(tid,q){q=q.toLowerCase();document.querySelectorAll('#'+tid+' tbody tr').forEach(r=>{
  r.style.display=r.dataset.sk.toLowerCase().includes(q)?'':'none';});}

let cP=null,cR=null,_syncing=false;

function syncFrom(src){
  if(_syncing)return;_syncing=true;
  const tgt=src===cP?cR:cP;
  if(!tgt){_syncing=false;return;}
  const sx=src.scales.x;
  tgt.scales.x.options.min=sx.min;
  tgt.scales.x.options.max=sx.max;
  tgt.update('none');
  _syncing=false;
}

function zoomCfg(){return{
  zoom:{wheel:{enabled:true,speed:.08},pinch:{enabled:true},mode:'x',
    onZoomComplete:function(ctx){syncFrom(ctx.chart);}},
  pan:{enabled:true,mode:'x',threshold:5,
    onPanComplete:function(ctx){syncFrom(ctx.chart);}},
  limits:{x:{minRange:20}}
};}

function showChart(tk){
  document.getElementById('chartModal').classList.add('open');
  document.getElementById('chartTicker').textContent=tk;
  document.getElementById('chartName').textContent='';
  document.getElementById('chartLoading').style.display='block';
  document.querySelector('.chart-wrap').style.display='none';
  document.querySelector('.rs-wrap').style.display='none';
  document.querySelector('.zoom-hint').style.display='none';
  fetch('/api/chart/'+tk+Q).then(r=>r.json()).then(d=>{
    if(d.error){document.getElementById('chartLoading').textContent=d.error;return;}
    document.getElementById('chartLoading').style.display='none';
    document.querySelector('.chart-wrap').style.display='block';
    document.querySelector('.rs-wrap').style.display='block';
    document.querySelector('.zoom-hint').style.display='block';
    if(d.name)document.getElementById('chartName').textContent=' \u2014 '+d.name;
    renderCharts(d);
  }).catch(()=>{document.getElementById('chartLoading').textContent='Failed to load data';});
}

function closeChart(){
  document.getElementById('chartModal').classList.remove('open');
  document.getElementById('chartBox').classList.remove('fs');
  document.getElementById('fsBtn').textContent='Fullscreen';
  if(cP){cP.destroy();cP=null;}
  if(cR){cR.destroy();cR=null;}
}

function toggleFS(){
  const box=document.getElementById('chartBox');
  box.classList.toggle('fs');
  const isFS=box.classList.contains('fs');
  document.getElementById('fsBtn').textContent=isFS?'Exit Fullscreen':'Fullscreen';
  setTimeout(()=>{if(cP)cP.resize();if(cR)cR.resize();},280);
}

function resetZoom(){
  if(cP)cP.resetZoom();
  if(cR)cR.resetZoom();
}

function renderCharts(d){
  if(cP)cP.destroy();if(cR)cR.destroy();
  const lbl=d.dates;
  const skip=Math.max(1,Math.floor(lbl.length/60));
  const xCfg={type:'category',ticks:{maxTicksLimit:14,maxRotation:0,callback:function(v,i){return i%skip===0?lbl[i].slice(0,7):'';}},grid:{display:false}};
  const zp=zoomCfg();

  const cur = d.currency || CURRENCY;
  const benchLbl = d.bench_name || BENCH_LABEL;
  cP=new Chart(document.getElementById('cvPrice'),{type:'line',data:{
    labels:lbl,datasets:[
      {label:d.ticker,data:d.price,borderColor:'#2563eb',borderWidth:2,pointRadius:0,tension:.1,yAxisID:'y'},
      {label:benchLbl,data:d.spy,borderColor:'#f59e0b',borderWidth:1.5,pointRadius:0,borderDash:[5,3],tension:.1,yAxisID:'y1'}
    ]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
    plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:8,font:{size:12}}},
      title:{display:true,text:d.ticker+' vs '+benchLbl,font:{size:13}},
      tooltip:{callbacks:{label:function(ctx){return ctx.dataset.label+': '+cur+ctx.parsed.y.toLocaleString(undefined,{maximumFractionDigits:2});}}},
      zoom:zp},
    scales:{x:xCfg,
      y:{position:'left',title:{display:true,text:d.ticker+' ('+cur+')',color:'#2563eb',font:{weight:'bold'}},
        ticks:{color:'#2563eb'},grid:{color:'#f1f5f9'}},
      y1:{position:'right',title:{display:true,text:benchLbl+' ('+cur+')',color:'#f59e0b',font:{weight:'bold'}},
        ticks:{color:'#f59e0b'},grid:{drawOnChartArea:false}}
    }}});

  cR=new Chart(document.getElementById('cvRS'),{type:'line',data:{
    labels:lbl,datasets:[
      {label:'RS 20d (%)',data:d.rs_20,borderColor:'#059669',borderWidth:1.8,pointRadius:0,tension:.1,
        fill:{target:{value:0},above:'rgba(5,150,105,.12)',below:'rgba(220,38,38,.1)'}},
      {label:'RS 60d (%)',data:d.rs_60,borderColor:'#6366f1',borderWidth:1.5,pointRadius:0,tension:.1,borderDash:[5,2]},
      {label:'',data:lbl.map(()=>0),borderColor:'#cbd5e1',borderWidth:1,pointRadius:0,borderDash:[3,3]}
    ]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
    spanGaps:true,
    plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:8,font:{size:11},
        filter:function(item){return item.text!=='';}}},
      title:{display:true,text:'Relative Strength vs '+benchLbl+' (Rolling Excess Return %)',font:{size:12}},
      tooltip:{callbacks:{label:function(ctx){if(!ctx.dataset.label)return null;
        return ctx.dataset.label+': '+(ctx.parsed.y!==null?ctx.parsed.y.toFixed(2)+'%':'N/A');}}},
      zoom:zp},
    scales:{x:xCfg,y:{grid:{color:'#f1f5f9'}}}}});
}

document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){
    const box=document.getElementById('chartBox');
    if(box.classList.contains('fs')){toggleFS();}
    else{closeChart();}
  }
});
</script></body></html>"""


# ═══════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════
def _request_market(default: str = "US") -> str:
    """request.args.get('market') 또는 path query → 정규화된 'US'/'KR'."""
    from flask import request
    m = (request.args.get("market") or default).upper()
    return m if m in MARKETS else default


def _ready() -> bool:
    """G 안에 최소 한 시장이라도 분석되어 있으면 True."""
    return any(m in G and "sector_results" in G[m] for m in MARKETS)


@app.route("/")
def index():
    if G_STATUS["state"] == "READY" and _ready():
        return redirect("/dashboard")
    return render_template_string(TEMPLATE_LANDING)


@app.route("/dashboard")
def dashboard():
    market = _request_market()
    state = _market_state(market)
    if not state or "sector_results" not in state:
        # 해당 시장이 아직 없으면 다른 시장으로 자동 폴백
        for alt in MARKETS:
            alt_state = _market_state(alt)
            if alt_state and "sector_results" in alt_state:
                return redirect(f"/dashboard?market={alt.lower()}")
        return redirect("/")

    mkt = MARKETS[market]
    kr_error = G.get("KR_ERROR") if market == "US" else None

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, (float, np.floating)):
            if np.isnan(v) or np.isinf(v):
                return None
            return round(float(v), 2)
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        return v

    def _build_rows(df, info, w52):
        rows = []
        for tk, row in df.iterrows():
            inf = info.get(tk, {})
            w = w52.get(tk, {})
            rows.append({
                "ticker": tk,
                "name": inf.get("name", ""),
                "category": inf.get("sector", ""),
                "industry": inf.get("industry", "") or inf.get("sector", ""),
                "w52_status": w.get("status"),
                "pct_from_high": w.get("pct_from_high"),
                **{k: _safe(row[k]) for k in [
                    "composite", "rs_60", "rs_20", "rs_10", "rs_5",
                    "ir_20", "alpha_ann", "early_trend", "status",
                ]},
            })
        return rows

    sector_rows = _build_rows(state["sector_results"], state["sector_info"], state["sector_w52"])
    stock_rows = _build_rows(state["sp500_results"], state["sp500_info"], state["sp500_w52"])

    s_cnt = state["sector_results"]["status"].value_counts()
    s_52h = sum(1 for v in state["sector_w52"].values() if v.get("status") == "52W_HIGH")
    s_52l = sum(1 for v in state["sector_w52"].values() if v.get("status") == "52W_LOW")

    sector_highs = [{"ticker": tk, "name": state["sector_info"].get(tk, {}).get("name", "")}
                    for tk, v in state["sector_w52"].items() if v.get("status") == "52W_HIGH"]
    sector_lows = [{"ticker": tk, "name": state["sector_info"].get(tk, {}).get("name", "")}
                   for tk, v in state["sector_w52"].items() if v.get("status") == "52W_LOW"]

    def _early_candidates(df, info, kind):
        out = []
        early = df[df["status"] == "Early Trend"].copy()
        early = early.sort_values("composite", ascending=False)
        for tk, row in early.iterrows():
            meta = info.get(tk, {})
            out.append({
                "ticker": tk,
                "name": meta.get("name", tk),
                "sector": meta.get("sector", "Unknown"),
                "industry": meta.get("industry", "") or meta.get("sector", "Unknown"),
                "kind": kind,
                "rs_5": _safe(row.get("rs_5")),
                "rs_20": _safe(row.get("rs_20")),
                "rs_60": _safe(row.get("rs_60")),
                "ir_20": _safe(row.get("ir_20")),
                "composite": _safe(row.get("composite")),
            })
        return out

    early_stocks = _early_candidates(state["sp500_results"], state["sp500_info"], "stock")
    early_etfs = _early_candidates(state["sector_results"], state["sector_info"], "etf")
    narr_candidates = early_stocks + early_etfs

    # 시장 스위처용 메타 — 어느 시장이 분석 완료/실패인지
    market_tabs = []
    for code, info_m in MARKETS.items():
        ready = code in G and "sector_results" in G[code]
        market_tabs.append({
            "code": code,
            "label": info_m["label"],
            "title": info_m["title"],
            "ready": ready,
            "active": code == market,
            "url": f"/dashboard?market={code.lower()}",
        })

    return render_template_string(
        TEMPLATE_DASHBOARD,
        sector_rows=sector_rows, stock_rows=stock_rows,
        s_total=len(state["sector_results"]),
        s_n_leader=int(s_cnt.get("LEADER", 0)),
        s_n_early=int(s_cnt.get("Early Trend", 0)),
        s_n_52h=s_52h, s_n_52l=s_52l,
        sector_highs=sector_highs, sector_lows=sector_lows,
        breadth=state["breadth"],
        bench=mkt["bench"], updated=state["updated"], beta_adj=state["beta_adj"],
        gemini_md=state.get("gemini_analysis", ""),
        gemini_updated=state.get("gemini_updated", ""),
        narr_candidates=narr_candidates,
        narr_lookback=NARRATIVE_LOOKBACK_DAYS,
        market=market,
        market_label=mkt["label"],
        market_title=mkt["title"],
        constituents_label=mkt["constituents_label"],
        etf_label=mkt["etf_label"],
        bench_name=mkt["bench_name"],
        currency=mkt["currency"],
        market_tabs=market_tabs,
        kr_error=kr_error,
    )


@app.route("/api/start", methods=["POST"])
def api_start():
    if G_STATUS["state"] == "LOADING":
        return jsonify({"ok": False, "msg": "Already running"})
    t = threading.Thread(target=_run_analysis, daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify(G_STATUS)


@app.route("/toggle_beta")
def toggle_beta():
    market = _request_market()
    state = _market_state(market)
    if not state or "close" not in state:
        return redirect("/")
    bench = MARKETS[market]["bench"]
    cons_label = MARKETS[market]["constituents_label"]
    state["beta_adj"] = not state["beta_adj"]
    _log(f"[*] Beta toggled ({market}) -> {state['beta_adj']}")

    sp500_cols = [bench] + [t for t in state["sp500_info"] if t in state["close"].columns]
    etf_cols = [bench] + [t for t in state["sector_info"] if t in state["close"].columns]
    state["sp500_results"] = screen(state["close"][sp500_cols],
                                    beta_adj=state["beta_adj"],
                                    label=f"{market} {cons_label}", bench=bench)
    state["sector_results"] = screen(state["close"][etf_cols],
                                     beta_adj=state["beta_adj"],
                                     label=f"{market} ETF", bench=bench)
    state["breadth"] = calc_breadth(state["close"], state["sp500_results"],
                                    state["sp500_info"], state["sp500_w52"],
                                    bench=bench)
    state["updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return redirect(f"/dashboard?market={market.lower()}")


@app.route("/api/market_ai", methods=["POST"])
def api_market_ai():
    market = _request_market()
    state = _market_state(market)
    if not state or "sector_results" not in state:
        return jsonify({"ok": False, "msg": f"{market} 분석을 먼저 실행하세요"}), 400
    _log(f"[*] Generating market AI analysis ({market}, on-demand) ...")
    md = gemini_analyze(
        state["sector_results"], state["sector_info"], state["sector_w52"],
        state["close"],
        sp500_results=state["sp500_results"], sp500_info=state["sp500_info"],
        breadth=state["breadth"],
        market=market,
    )
    state["gemini_analysis"] = md
    state["gemini_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return jsonify({"ok": True, "md": md, "updated": state["gemini_updated"]})


@app.route("/api/chart/<ticker>")
def api_chart(ticker):
    market = _request_market()
    state = _market_state(market)
    if not state or "close" not in state:
        return jsonify({"error": "No data"}), 404
    bench = MARKETS[market]["bench"]
    bench_name = MARKETS[market]["bench_name"]
    currency = MARKETS[market]["currency"]
    close = state["close"]
    if ticker not in close.columns or bench not in close.columns:
        return jsonify({"error": f"{ticker} not found"}), 404

    p = close[[ticker, bench]].dropna()
    if p.empty:
        return jsonify({"error": "No price data"}), 404

    r_tk = np.log(p[ticker] / p[ticker].shift(1))
    r_b = np.log(p[bench] / p[bench].shift(1))

    beta_adj = state.get("beta_adj", BETA_ADJ)
    if beta_adj and len(p) > W_BETA:
        mr_tk = r_tk.rolling(W_BETA).mean()
        mr_b = r_b.rolling(W_BETA).mean()
        cov_val = (r_tk * r_b).rolling(W_BETA).mean() - mr_tk * mr_b
        var_b = (r_b ** 2).rolling(W_BETA).mean() - mr_b ** 2
        bbeta = (cov_val / var_b).replace([np.inf, -np.inf], np.nan)
        excess = (r_tk - bbeta * r_b).fillna(r_tk - r_b)
    else:
        excess = r_tk - r_b

    def _safe_list(s):
        return [None if pd.isna(v) else round(float(v), 2) for v in s]

    rs_20 = (np.exp(excess.rolling(W_JUDGE).sum()) - 1) * 100
    rs_60 = (np.exp(excess.rolling(W_FILTER).sum()) - 1) * 100

    name = ""
    for src in ("sector_info", "sp500_info"):
        if src in state and ticker in state[src]:
            name = state[src][ticker].get("name", "")
            break

    return jsonify({
        "ticker": ticker,
        "name": name,
        "bench": bench,
        "bench_name": bench_name,
        "currency": currency,
        "dates": [d.strftime("%Y-%m-%d") for d in p.index],
        "price": _safe_list(p[ticker]),
        "spy": _safe_list(p[bench]),  # 차트 코드 호환 위해 키 이름 유지 (실제는 시장 벤치마크)
        "rs_20": _safe_list(rs_20),
        "rs_60": _safe_list(rs_60),
    })


@app.route("/api/narratives/<ticker>", methods=["GET"])
def api_narratives_get(ticker):
    """캐시된 narrative 조회. 없으면 cached=False 만 반환 (생성하지 않음).

    옵션:
      ?date=YYYY-MM-DD  → DB 의 해당 일자 버전 로드 (없으면 history 만 반환)
      ?date 미지정      → 메모리 → DB(최신) 순으로 조회
    응답에 항상 history (해당 ticker 의 일자 목록, 최신순) 포함."""
    from flask import request
    market = _request_market()
    date_q = (request.args.get("date") or "").strip() or None
    history = _load_narrative_history(ticker, market)

    narr = None
    if date_q:
        narr = _load_narrative_db(ticker, market, date=date_q)
    else:
        state = _market_state(market)
        if state:
            narr = state.get("narratives", {}).get(ticker)
        if not narr:
            narr = _load_narrative_db(ticker, market)

    if not narr:
        return jsonify({"ok": True, "cached": False, "narrative": None,
                        "history": history})
    return jsonify({"ok": True, "cached": True, "narrative": narr,
                    "history": history})


@app.route("/api/narratives/<ticker>", methods=["POST"])
def api_narratives_generate(ticker):
    """단일 종목 narrative 생성/재생성. force=1 이면 캐시 무시.
    market=us|kr 쿼리로 시장 선택 (기본 us)."""
    market = _request_market()
    state = _market_state(market)
    if not state or "sp500_results" not in state:
        return jsonify({"ok": False,
                        "msg": f"{market} 분석을 먼저 실행하세요"}), 400

    from flask import request
    force = request.args.get("force", "0") == "1"
    if not force:
        cached = state.get("narratives", {}).get(ticker)
        if cached:
            return jsonify({"ok": True, "cached": True, "narrative": cached})

    sp500 = state["sp500_results"]
    sector_results = state.get("sector_results")
    info = state["sp500_info"]
    sector_info = state.get("sector_info", {})

    row, meta, kind = None, None, None
    if ticker in sp500.index:
        row = sp500.loc[ticker]
        meta = info.get(ticker, {})
        kind = "stock"
    elif sector_results is not None and ticker in sector_results.index:
        row = sector_results.loc[ticker]
        meta = sector_info.get(ticker, {})
        kind = "etf"
    else:
        return jsonify({"ok": False, "msg": f"{ticker} not found"}), 404

    rs_metrics = {
        "rs_5": float(row["rs_5"]) if pd.notna(row.get("rs_5")) else None,
        "rs_20": float(row["rs_20"]) if pd.notna(row.get("rs_20")) else None,
        "rs_60": float(row["rs_60"]) if pd.notna(row.get("rs_60")) else None,
        "ir_20": float(row["ir_20"]) if pd.notna(row.get("ir_20")) else None,
    }

    # 생성 시점 종가 (DB / 모달 헤더용) — 거래일 기준 가장 최근 close
    price = None
    try:
        close_df = state.get("close")
        if close_df is not None and ticker in close_df.columns:
            s = close_df[ticker].dropna()
            if not s.empty:
                price = float(s.iloc[-1])
    except Exception as exc:
        _log(f"[!] price fetch failed for {ticker}: {exc}")

    _log(f"[*] Narrative on-demand: {market}/{ticker} ({kind})")
    result = _process_one_narrative(
        ticker, meta.get("name", ticker), meta.get("sector", "Unknown"),
        rs_metrics,
        industry=meta.get("industry", "") or meta.get("sector", ""),
        market=market,
    )
    result["kind"] = kind
    result["price"] = price
    # 생성 일자 (PK 의 date 부분)
    result["date"] = (result.get("generated_at") or
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M")).split(" ")[0]

    state.setdefault("narratives", {})[ticker] = result
    _save_narrative_db(result, market)
    history = _load_narrative_history(ticker, market)
    return jsonify({"ok": True, "cached": False, "narrative": result,
                    "history": history})


@app.route("/reset")
def reset():
    G_STATUS.update(state="IDLE", step="", detail="", progress=0)
    G.clear()
    return redirect("/")


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Market Regime Analyzer")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    _log(f"\n* Server starting -> http://127.0.0.1:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
