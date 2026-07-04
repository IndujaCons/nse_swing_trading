=== Mom500 — Nifty500 Universe, 2-Monthly Rebalance, β≤1.2 | Regime ON [EMA200] | Sector≤4 ===
    top_n=20 buffer_in=15 buffer_out=40 beta_cap=1.2
Loading PIT universe...
  886 unique PIT tickers across all periods
Loading EPS data...
  871 stocks with EPS data
  Sector map loaded: 34 PIT dates
Loading cached data from /Users/jay/dev/relative_strength/data/cache/mom500_daily.pkl...
Fetching Nifty 500 (beta)...
  3128 bars
Fetching Nifty 500 (regime filter)...
  3128 bars
  Trading days in backtest: 2840 (2015-01-01 → 2026-07-01)
  Rebalance dates: 69

=====================================================================================================================
  MOM500 PIT BACKTEST  |  NAV/20 slot  |  2-Month Rebalance  |  Nifty500 Universe  |  Beta≤1.2  |  Regime ON [EMA200]
=====================================================================================================================

========================================================================
  REBALANCE #01  —  02 Feb 2015
  NAV: ₹2,000,000  |  Slot: ₹100,000  |  Cash: ₹2,000,000
========================================================================
  [SECTOR CAP≤4] dropped: ASHOKLEY

  EXITS (0)
    —

  ENTRIES (19)
  [52w filter blocked 1: BHARATRAS(-25.4%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BOSCHLTD    1      AUTO       4.158    0.40   +163.6%   +67.5%    22,437.0    4      ₹89,748       +14.3%    
  EVEREADY    2      CON DUR    3.848    0.50   +462.6%   +52.4%    200.4       499    ₹100,000      +3.3%     
  RATNAMANI   3      METAL      3.604    0.40   +451.3%   +59.8%    456.1       219    ₹99,895       +8.0%     
  CENTURYPLY  4      CON DUR    3.357    0.71   +631.4%   +45.2%    176.2       567    ₹99,932       +5.6%     
  HONAUT      5      MFG        3.246    0.42   +184.0%   +40.5%    7,224.6     13     ₹93,919       +3.3%     
  AXISBANK    6      PVT BNK    3.173    0.38   +168.1%   +43.5%    597.6       167    ₹99,792       +11.9%    
  BEL         7      DEFENCE    3.149    0.49   +245.9%   +67.9%    28.5        3505   ₹99,988       +6.7%     
  SUNDRMFAST  8      AUTO       3.148    0.65   +372.2%   +38.6%    189.7       527    ₹99,968       +5.6%     
  WHIRLPOOL   9      CON DUR    3.087    0.59   +277.4%   +50.5%    693.1       144    ₹99,799       +3.3%     
  ZFCVINDIA   10     AUTO       3.033    0.50   +176.2%   +45.9%    868.1       115    ₹99,830       +5.8%     
  ORIENTCEM   11     MFG        3.030    0.54   +418.8%   +34.8%    167.0       598    ₹99,895       +2.6%     
  SYMPHONY    13     CON DUR    2.831    0.26   +468.3%   +19.6%    1,040.9     96     ₹99,930       +7.5%     
  WELSPUNLIV  14     CONSUMP    2.767    0.70   +336.8%   +30.9%    35.2        2839   ₹99,989       +11.6%    
  MOTILALOFS  15     FIN SVC    2.745    0.77   +262.2%   +54.9%    311.2       321    ₹99,895       +10.3%    
  EICHERMOT   16     AUTO       2.735    0.60   +229.2%   +32.2%    1,530.4     65     ₹99,477       +6.3%     
  BAJFINANCE  17     FIN SVC    2.731    0.34   +167.1%   +48.8%    39.9        2507   ₹99,987       +7.9%     
  SHREECEM    18     INFRA      2.703    0.55   +147.9%   +25.9%    10,437.7    9      ₹93,940       +6.1%     
  BBTC        19     FMCG       2.657    0.89   +322.4%   +84.3%    438.1       228    ₹99,890       +1.1%     
  CANFINHOME  21     FIN SVC    2.636    0.50   +315.4%   +40.6%    115.3       867    ₹99,982       +6.6%     

  HOLDS (0)
    —

  AFTER: Invested ₹1,875,858 | Cash ₹121,914 | Total ₹1,997,773 | Positions 19/20 | Slot ₹100,000

========================================================================
  REBALANCE #02  —  01 Apr 2015
  NAV: ₹2,070,634  |  Slot: ₹103,532  |  Cash: ₹121,914
========================================================================

  EXITS (12)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  WHIRLPOOL   49     CON DUR    02-Feb-15   693.1       709.4       144    ₹2,351        +2.4%     58d   
  BEL         42     DEFENCE    02-Feb-15   28.5        29.1        3505   ₹2,078        +2.1%     58d   
  SHREECEM    48     INFRA      02-Feb-15   10,437.7    10,419.0    9      ₹-169         -0.2%     58d   
  BAJFINANCE  57     FIN SVC    02-Feb-15   39.9        39.7        2507   ₹-482         -0.5%     58d   
  BBTC        55     FMCG       02-Feb-15   438.1       435.4       228    ₹-619         -0.6%     58d   
  ORIENTCEM   40     MFG        02-Feb-15   167.0       162.7       598    ₹-2,610       -2.6%     58d   
  WELSPUNLIV  32     CONSUMP    02-Feb-15   35.2        34.2        2839   ₹-2,809       -2.8%     58d   
  EICHERMOT   78     AUTO       02-Feb-15   1,530.4     1,485.0     65     ₹-2,954       -3.0%     58d   
  RATNAMANI   124    METAL      02-Feb-15   456.1       426.4       219    ₹-6,505       -6.5%     58d   
  AXISBANK    87     PVT BNK    02-Feb-15   597.6       551.4       167    ₹-7,703       -7.7%     58d   
  MOTILALOFS  65     FIN SVC    02-Feb-15   311.2       286.0       321    ₹-8,099       -8.1%     58d   
  SUNDRMFAST  73     AUTO       02-Feb-15   189.7       168.3       527    ₹-11,279      -11.3%    58d   

  ENTRIES (12)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  DYNAMATECH  2      DEFENCE    4.283    0.66   +559.8%   +106.4%   3,893.2     26     ₹101,223      +15.3%    
  VIVIDHA     4      MFG        4.125    -0.10  +1082.1%  +32.6%    72.5        1428   ₹103,523      +25.5%    
  SPARC       5      HEALTH     4.086    0.76   +228.9%   +169.3%   500.0       207    ₹103,500      +5.2%     
  MBLINFRA    6      INFRA      3.504    0.76   +407.6%   +57.9%    306.4       337    ₹103,248      +4.2%     
  LUPIN       7      HEALTH     3.473    0.42   +116.3%   +42.7%    1,917.4     53     ₹101,625      +6.5%     
  WOCKPHARMA  8      HEALTH     3.412    0.53   +327.9%   +85.3%    1,701.9     60     ₹102,116      +3.9%     
  TATAELXSI   9      IT         3.332    0.60   +123.8%   +116.2%   552.9       187    ₹103,392      +10.4%    
  GILLETTE    10     FMCG       3.260    0.22   +158.9%   +50.5%    4,160.0     24     ₹99,840       +6.4%     
  KRBL        11     FMCG       3.218    0.64   +251.4%   +72.0%    153.6       673    ₹103,379      +12.0%    
  BHARATFORG  12     DEFENCE    3.082    0.62   +232.3%   +41.5%    610.1       169    ₹103,099      +4.3%     
  JKIL        13     INFRA      2.921    0.50   +302.1%   +41.8%    319.6       323    ₹103,243      +9.6%     
  ASHOKLEY    14     AUTO       2.842    0.96   +319.5%   +43.9%    30.2        3432   ₹103,529      +4.6%     

  HOLDS (7)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  EVEREADY    3      CON DUR    02-Feb-15   200.4       265.4       499    ₹32,411       +32.4%      +6.1%     
  CENTURYPLY  1      CON DUR    02-Feb-15   176.2       233.1       567    ₹32,252       +32.3%      +4.2%     
  SYMPHONY    17     CON DUR    02-Feb-15   1,040.9     1,213.6     96     ₹16,579       +16.6%      +12.4%    
  HONAUT      19     MFG        02-Feb-15   7,224.6     8,387.7     13     ₹15,121       +16.1%      -0.9%     
  ZFCVINDIA   27     AUTO       02-Feb-15   868.1       956.1       115    ₹10,120       +10.1%      +5.1%     
  CANFINHOME  24     FIN SVC    02-Feb-15   115.3       118.8       867    ₹3,036        +3.0%       +8.6%     
  BOSCHLTD    22     AUTO       02-Feb-15   22,437.0    22,972.6    4      ₹2,142        +2.4%       -1.7%     

  AFTER: Invested ₹2,026,720 | Cash ₹42,452 | Total ₹2,069,172 | Positions 19/20 | Slot ₹103,532

========================================================================
  REBALANCE #03  —  01 Jun 2015
  NAV: ₹1,964,833  |  Slot: ₹98,242  |  Cash: ₹42,452
========================================================================
  [SECTOR CAP≤4] dropped: AUROPHARMA, EMAMILTD

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  CENTURYPLY  34     CON DUR    02-Feb-15   176.2       198.8       567    ₹12,760       +12.8%    119d  
  SYMPHONY    48     CON DUR    02-Feb-15   1,040.9     1,135.5     96     ₹9,077        +9.1%     119d  
  HONAUT      233    MFG        02-Feb-15   7,224.6     7,173.1     13     ₹-669         -0.7%     119d  
  ASHOKLEY    61     AUTO       01-Apr-15   30.2        29.3        3432   ₹-2,936       -2.8%     61d   
  BOSCHLTD    141    AUTO       02-Feb-15   22,437.0    20,961.6    4      ₹-5,902       -6.6%     119d  
  BHARATFORG  74     DEFENCE    01-Apr-15   610.1       560.0       169    ₹-8,458       -8.2%     61d   
  MBLINFRA    54     INFRA      01-Apr-15   306.4       268.5       337    ₹-12,766      -12.4%    61d   
  SPARC       80     HEALTH     01-Apr-15   500.0       398.2       207    ₹-21,064      -20.4%    61d   
  WOCKPHARMA  167    HEALTH     01-Apr-15   1,701.9     1,341.2     60     ₹-21,645      -21.2%    61d   
  DYNAMATECH  88     DEFENCE    01-Apr-15   3,893.2     2,783.2     26     ₹-28,861      -28.5%    61d   

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  WELSPUNLIV  3      CONSUMP    4.450    0.88   +341.4%   +61.0%    51.7        1901   ₹98,222       +7.6%     
  PAGEIND     4      MFG        3.924    0.22   +185.4%   +45.2%    15,087.9    6      ₹90,527       +14.7%    
  AEGISLOG    5      INFRA      3.838    0.92   +278.1%   +68.4%    62.2        1578   ₹98,230       -0.4%     
  AJANTPHARM  6      HEALTH     3.828    0.41   +285.4%   +46.7%    976.8       100    ₹97,680       +15.1%    
  BRITANNIA   7      FMCG       3.500    0.50   +214.6%   +24.1%    1,115.5     88     ₹98,166       +6.8%     
  NATCOPHARM  8      HEALTH     3.406    1.05   +212.4%   +63.3%    407.7       240    ₹97,843       -0.5%     
  EICHERMOT   10     AUTO       2.908    1.08   +196.3%   +20.8%    1,740.7     56     ₹97,477       +5.2%     
  MARICO      11     FMCG       2.883    0.41   +101.2%   +25.6%    190.3       516    ₹98,212       +8.3%     
  APLLTD      12     HEALTH     2.846    0.67   +133.9%   +35.4%    481.2       204    ₹98,174       +4.4%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  EVEREADY    2      CON DUR    02-Feb-15   200.4       307.9       499    ₹53,635       +53.6%      +11.8%    
  VIVIDHA     1      MFG        01-Apr-15   72.5        96.8        1428   ₹34,733       +33.6%      +8.2%     
  CANFINHOME  43     FIN SVC    02-Feb-15   115.3       125.6       867    ₹8,940        +8.9%       -1.7%     
  ZFCVINDIA   19     AUTO       02-Feb-15   868.1       932.7       115    ₹7,430        +7.4%       +1.2%     
  KRBL        16     FMCG       01-Apr-15   153.6       164.0       673    ₹6,989        +6.8%       +3.8%     
  TATAELXSI   32     IT         01-Apr-15   552.9       549.6       187    ₹-609         -0.6%       +5.2%     
  JKIL        10     INFRA      01-Apr-15   319.6       311.7       323    ₹-2,554       -2.5%       +6.3%     
  GILLETTE    28     FMCG       01-Apr-15   4,160.0     3,804.3     24     ₹-8,536       -8.5%       -0.1%     
  LUPIN       33     HEALTH     01-Apr-15   1,917.4     1,686.5     53     ₹-12,242      -12.0%      +1.8%     

  AFTER: Invested ₹1,877,133 | Cash ₹86,662 | Total ₹1,963,795 | Positions 18/20 | Slot ₹98,242

========================================================================
  REBALANCE #04  —  03 Aug 2015
  NAV: ₹2,113,527  |  Slot: ₹105,676  |  Cash: ₹86,662
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  EVEREADY    6      CON DUR    02-Feb-15   200.4       319.4       499    ₹59,388       +59.4%    182d  
  TATAELXSI   9      IT         01-Apr-15   552.9       796.4       187    ₹45,536       +44.0%    124d  
  CANFINHOME  85     FIN SVC    02-Feb-15   115.3       149.0       867    ₹29,176       +29.2%    182d  
  ZFCVINDIA   68     AUTO       02-Feb-15   868.1       1,050.2     115    ₹20,942       +21.0%    182d  
  KRBL        80     FMCG       01-Apr-15   153.6       166.3       673    ₹8,569        +8.3%     124d  
  NATCOPHARM  96     HEALTH     01-Jun-15   407.7       425.0       240    ₹4,167        +4.3%     63d   
  MARICO      62     FMCG       01-Jun-15   190.3       184.9       516    ₹-2,823       -2.9%     63d   
  LUPIN       185    HEALTH     01-Apr-15   1,917.4     1,572.2     53     ₹-18,298      -18.0%    124d  
  PAGEIND     165    MFG        01-Jun-15   15,087.9    12,198.5    6      ₹-17,336      -19.2%    63d   
  VIVIDHA     170    MFG        01-Apr-15   72.5        37.1        1428   ₹-50,586      -48.9%    124d  

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RAJESHEXPO  2      CON DUR    4.664    0.97   +239.0%   +148.2%   561.3       188    ₹105,521      +23.9%    
  NIITLTD     4      IT         3.242    1.19   +69.4%    +121.1%   67.5        1566   ₹105,655      +25.5%    
  EMAMILTD    5      FMCG       3.228    0.57   +154.5%   +46.2%    562.9       187    ₹105,267      +9.9%     
  HINDPETRO   6      OIL&GAS    3.142    0.82   +142.5%   +53.0%    81.6        1295   ₹105,666      +3.8%     
  BAJFINANCE  7      FIN SVC    3.140    0.73   +162.1%   +33.1%    54.7        1933   ₹105,668      +7.3%     
  SAMMAANCAP  9      FIN SVC    2.853    0.91   +127.5%   +36.8%    503.0       210    ₹105,630      +9.1%     
  CHENNPETRO  10     OIL&GAS    2.835    1.10   +100.6%   +108.9%   139.4       758    ₹105,657      +0.5%     
  BHARATRAS   11     FMCG       2.829    0.83   +218.9%   +59.0%    320.5       329    ₹105,447      +10.3%    
  MARUTI      12     AUTO       2.804    0.69   +75.0%    +21.8%    4,018.0     26     ₹104,467      +5.9%     
  PRAJIND     13     ENERGY     2.674    1.07   +73.4%    +89.0%    95.5        1106   ₹105,611      +6.5%     
  EROSMEDIA   14     MEDIA      2.661    0.46   +157.0%   +53.6%    584.2       180    ₹105,165      +0.5%     

  HOLDS (8)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  WELSPUNLIV  1      CONSUMP    01-Jun-15   51.7        86.5        1901   ₹66,175       +67.4%      +19.0%    
  APLLTD      10     HEALTH     01-Jun-15   481.2       657.5       204    ₹35,965       +36.6%      +0.6%     
  BRITANNIA   3      FMCG       01-Jun-15   1,115.5     1,370.3     88     ₹22,416       +22.8%      +7.7%     
  AEGISLOG    34     INFRA      01-Jun-15   62.2        76.0        1578   ₹21,712       +22.1%      -0.8%     
  GILLETTE    25     FMCG       01-Apr-15   4,160.0     4,521.0     24     ₹8,663        +8.7%       +7.7%     
  JKIL        38     INFRA      01-Apr-15   319.6       338.5       323    ₹6,103        +5.9%       -0.4%     
  EICHERMOT   28     AUTO       01-Jun-15   1,740.7     1,769.0     56     ₹1,584        +1.6%       -4.1%     
  AJANTPHARM  49     HEALTH     01-Jun-15   976.8       938.5       100    ₹-3,832       -3.9%       -1.9%     

  AFTER: Invested ₹2,109,575 | Cash ₹2,576 | Total ₹2,112,150 | Positions 19/20 | Slot ₹105,676

========================================================================
  REBALANCE #05  —  01 Oct 2015
  NAV: ₹2,001,847  |  Slot: ₹100,092  |  Cash: ₹2,576
========================================================================
  [SECTOR CAP≤4] dropped: UNICHEMLAB

  [REGIME OFF] Nifty 500 6,654.5 < EMA200 6,749.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  WELSPUNLIV  3      CONSUMP    01-Jun-15   51.7        77.7        1901   ₹49,535       +50.4%      +2.6%     
  APLLTD      56     HEALTH     01-Jun-15   481.2       626.0       204    ₹29,537       +30.1%      +2.5%     
  AEGISLOG    44     INFRA      01-Jun-15   62.2        78.6        1578   ₹25,867       +26.3%      +1.2%     
  BRITANNIA   6      FMCG       01-Jun-15   1,115.5     1,364.1     88     ₹21,876       +22.3%      +4.4%     
  CHENNPETRO  29     OIL&GAS    03-Aug-15   139.4       169.4       758    ₹22,718       +21.5%      -0.8%     
  JKIL        14     INFRA      01-Apr-15   319.6       362.4       323    ₹13,807       +13.4%      +4.2%     
  SAMMAANCAP  11     FIN SVC    03-Aug-15   503.0       530.3       210    ₹5,729        +5.4%       +7.0%     
  MARUTI      16     AUTO       03-Aug-15   4,018.0     4,182.0     26     ₹4,265        +4.1%       +3.1%     
  GILLETTE    27     FMCG       01-Apr-15   4,160.0     4,231.7     24     ₹1,720        +1.7%       +3.3%     
  RAJESHEXPO  2      CON DUR    03-Aug-15   561.3       548.4       188    ₹-2,413       -2.3%       +12.6%    
  EICHERMOT   132    AUTO       01-Jun-15   1,740.7     1,695.1     56     ₹-2,552       -2.6%       +0.2%     
  AJANTPHARM  47     HEALTH     01-Jun-15   976.8       913.9       100    ₹-6,288       -6.4%       +3.4%     
  NIITLTD     19     IT         03-Aug-15   67.5        61.5        1566   ₹-9,374       -8.9%       +4.4%     
  BAJFINANCE  53     FIN SVC    03-Aug-15   54.7        49.1        1933   ₹-10,689      -10.1%      +1.3%     
  HINDPETRO   68     OIL&GAS    03-Aug-15   81.6        71.8        1295   ₹-12,626      -11.9%      -1.0%     
  EMAMILTD    75     FMCG       03-Aug-15   562.9       496.2       187    ₹-12,487      -11.9%      +0.6%     
  EROSMEDIA   88     MEDIA      03-Aug-15   584.2       503.5       180    ₹-14,535      -13.8%      -1.8%     
  PRAJIND     171    ENERGY     03-Aug-15   95.5        73.1        1106   ₹-24,786      -23.5%      +3.2%     
  BHARATRAS   193    FMCG       03-Aug-15   320.5       226.8       329    ₹-30,822      -29.2% ⚠    -1.7%     
  ⚠  WAZ < 0 (momentum below universe mean): BHARATRAS

  AFTER: Invested ₹1,999,271 | Cash ₹2,576 | Total ₹2,001,847 | Positions 19/20 | Slot ₹100,092

========================================================================
  REBALANCE #06  —  01 Dec 2015
  NAV: ₹1,998,727  |  Slot: ₹99,936  |  Cash: ₹2,576
========================================================================
  [SECTOR CAP≤4] dropped: HERITGFOOD

  [REGIME OFF] Nifty 500 6,712.5 < EMA200 6,742.0 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  AEGISLOG    9      INFRA      01-Jun-15   62.2        97.9        1578   ₹56,223       +57.2%      +7.5%     
  WELSPUNLIV  22     CONSUMP    01-Jun-15   51.7        80.0        1901   ₹53,855       +54.8%      +8.8%     
  APLLTD      74     HEALTH     01-Jun-15   481.2       629.3       204    ₹30,211       +30.8%      +3.8%     
  RAJESHEXPO  2      CON DUR    03-Aug-15   561.3       698.5       188    ₹25,790       +24.4%      +4.8%     
  NIITLTD     27     IT         03-Aug-15   67.5        80.8        1566   ₹20,897       +19.8%      +4.6%     
  BRITANNIA   36     FMCG       01-Jun-15   1,115.5     1,298.0     88     ₹16,058       +16.4%      -0.8%     
  CHENNPETRO  106    OIL&GAS    03-Aug-15   139.4       156.6       758    ₹13,036       +12.3%      +4.4%     
  JKIL        41     INFRA      01-Apr-15   319.6       340.3       323    ₹6,664        +6.5%       +0.5%     
  MARUTI      39     AUTO       03-Aug-15   4,018.0     4,159.9     26     ₹3,692        +3.5%       -0.5%     
  BAJFINANCE  24     FIN SVC    03-Aug-15   54.7        53.9        1933   ₹-1,461       -1.4%       +3.7%     
  GILLETTE    97     FMCG       01-Apr-15   4,160.0     4,018.2     24     ₹-3,403       -3.4%       -0.5%     
  SAMMAANCAP  71     FIN SVC    03-Aug-15   503.0       484.6       210    ₹-3,863       -3.7%       +4.8%     
  HINDPETRO   64     OIL&GAS    03-Aug-15   81.6        78.4        1295   ₹-4,142       -3.9%       +6.0%     
  EICHERMOT   342    AUTO       01-Jun-15   1,740.7     1,505.2     56     ₹-13,188      -13.5%      -2.6%     
  AJANTPHARM  201    HEALTH     01-Jun-15   976.8       814.9       100    ₹-16,187      -16.6%      -3.0%     
  BHARATRAS   203    FMCG       03-Aug-15   320.5       265.0       329    ₹-18,275      -17.3% ⚠    +6.6%     
  PRAJIND     207    ENERGY     03-Aug-15   95.5        71.8        1106   ₹-26,192      -24.8%      -0.6%     
  EMAMILTD    354    FMCG       03-Aug-15   562.9       400.9       187    ₹-30,300      -28.8% ⚠    -4.8%     
  EROSMEDIA   406    MEDIA      03-Aug-15   584.2       228.4       180    ₹-64,053      -60.9% ⚠    -12.4%    
  ⚠  WAZ < 0 (momentum below universe mean): BHARATRAS, EMAMILTD, EROSMEDIA

  AFTER: Invested ₹1,996,151 | Cash ₹2,576 | Total ₹1,998,727 | Positions 19/20 | Slot ₹99,936

========================================================================
  REBALANCE #07  —  01 Feb 2016
  NAV: ₹1,926,747  |  Slot: ₹96,337  |  Cash: ₹2,576
========================================================================
  [SECTOR CAP≤4] dropped: MARICO

  [REGIME OFF] Nifty 500 6,341.6 < EMA200 6,646.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  WELSPUNLIV  9      CONSUMP    01-Jun-15   51.7        78.0        1901   ₹50,052       +51.0%      +1.1%     
  AEGISLOG    19     INFRA      01-Jun-15   62.2        90.3        1578   ₹44,210       +45.0%      +2.6%     
  RAJESHEXPO  1      CON DUR    03-Aug-15   561.3       720.9       188    ₹30,001       +28.4%      +2.5%     
  APLLTD      103    HEALTH     01-Jun-15   481.2       560.0       204    ₹16,072       +16.4%      -1.1%     
  BRITANNIA   92     FMCG       01-Jun-15   1,115.5     1,223.3     88     ₹9,487        +9.7%       +1.0%     
  BAJFINANCE  10     FIN SVC    03-Aug-15   54.7        57.8        1933   ₹5,966        +5.6%       +2.0%     
  NIITLTD     57     IT         03-Aug-15   67.5        70.8        1566   ₹5,208        +4.9%       -2.4%     
  JKIL        63     INFRA      01-Apr-15   319.6       327.5       323    ₹2,552        +2.5%       +1.5%     
  CHENNPETRO  32     OIL&GAS    03-Aug-15   139.4       142.2       758    ₹2,129        +2.0%       -3.3%     
  SAMMAANCAP  86     FIN SVC    03-Aug-15   503.0       484.1       210    ₹-3,965       -3.8%       +1.8%     
  HINDPETRO   41     OIL&GAS    03-Aug-15   81.6        74.6        1295   ₹-9,019       -8.5%       -2.4%     
  GILLETTE    115    FMCG       01-Apr-15   4,160.0     3,802.3     24     ₹-8,586       -8.6%       -2.2%     
  EICHERMOT   127    AUTO       01-Jun-15   1,740.7     1,582.7     56     ₹-8,846       -9.1%       +3.8%     
  MARUTI      246    AUTO       03-Aug-15   4,018.0     3,604.9     26     ₹-10,740      -10.3% ⚠    -6.0%     
  PRAJIND     25     ENERGY     03-Aug-15   95.5        82.4        1106   ₹-14,431      -13.7%      +11.2%    
  AJANTPHARM  168    HEALTH     01-Jun-15   976.8       786.7       100    ₹-19,015      -19.5%      +4.6%     
  EMAMILTD    121    FMCG       03-Aug-15   562.9       437.8       187    ₹-23,407      -22.2%      +3.4%     
  BHARATRAS   144    FMCG       03-Aug-15   320.5       243.9       329    ₹-25,219      -23.9%      -4.1%     
  EROSMEDIA   379    MEDIA      03-Aug-15   584.2       200.6       180    ₹-69,066      -65.7% ⚠    +1.1%     
  ⚠  WAZ < 0 (momentum below universe mean): MARUTI, EROSMEDIA

  AFTER: Invested ₹1,924,171 | Cash ₹2,576 | Total ₹1,926,747 | Positions 19/20 | Slot ₹96,337

========================================================================
  REBALANCE #08  —  01 Apr 2016
  NAV: ₹1,898,508  |  Slot: ₹94,925  |  Cash: ₹2,576
========================================================================

  [REGIME OFF] Nifty 500 6,445.5 < EMA200 6,485.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  WELSPUNLIV  3      CONSUMP    01-Jun-15   51.7        91.2        1901   ₹75,146       +76.5%      +3.5%     
  AEGISLOG    61     INFRA      01-Jun-15   62.2        90.2        1578   ₹44,067       +44.9%      +1.1%     
  BAJFINANCE  6      FIN SVC    03-Aug-15   54.7        66.3        1933   ₹22,581       +21.4%      +4.6%     
  CHENNPETRO  10     OIL&GAS    03-Aug-15   139.4       156.1       758    ₹12,657       +12.0%      +4.0%     
  APLLTD      170    HEALTH     01-Jun-15   481.2       529.0       204    ₹9,744        +9.9%       -3.0%     
  RAJESHEXPO  9      CON DUR    03-Aug-15   561.3       615.3       188    ₹10,153       +9.6%       -1.8%     
  BRITANNIA   140    FMCG       01-Jun-15   1,115.5     1,170.3     88     ₹4,818        +4.9%       -1.3%     
  EICHERMOT   27     AUTO       01-Jun-15   1,740.7     1,794.6     56     ₹3,019        +3.1%       +2.5%     
  NIITLTD     —      OTHER      03-Aug-15   67.5        69.2        1566   ₹2,734        +2.6%       +5.1%     
  HINDPETRO   79     OIL&GAS    03-Aug-15   81.6        75.4        1295   ₹-8,072       -7.6%       +5.7%     
  GILLETTE    265    FMCG       01-Apr-15   4,160.0     3,749.4     24     ₹-9,855       -9.9% ⚠     -1.0%     
  SAMMAANCAP  139    FIN SVC    03-Aug-15   503.0       448.3       210    ₹-11,493      -10.9%      +2.1%     
  AJANTPHARM  67     HEALTH     01-Jun-15   976.8       847.5       100    ₹-12,927      -13.2%      +1.7%     
  MARUTI      334    AUTO       03-Aug-15   4,018.0     3,399.4     26     ₹-16,083      -15.4% ⚠    +1.5%     
  PRAJIND     54     ENERGY     03-Aug-15   95.5        79.0        1106   ₹-18,283      -17.3%      +8.0%     
  JKIL        369    INFRA      01-Apr-15   319.6       245.2       323    ₹-24,048      -23.3% ⚠    -6.5%     
  EMAMILTD    226    FMCG       03-Aug-15   562.9       391.6       187    ₹-32,030      -30.4% ⚠    -2.4%     
  BHARATRAS   249    FMCG       03-Aug-15   320.5       219.8       329    ₹-33,131      -31.4% ⚠    -1.0%     
  EROSMEDIA   348    MEDIA      03-Aug-15   584.2       173.9       180    ₹-73,854      -70.2%      +5.9%     
  ⚠  WAZ < 0 (momentum below universe mean): EMAMILTD, BHARATRAS, GILLETTE, MARUTI, JKIL

  AFTER: Invested ₹1,895,932 | Cash ₹2,576 | Total ₹1,898,508 | Positions 19/20 | Slot ₹94,925

========================================================================
  REBALANCE #09  —  01 Jun 2016
  NAV: ₹1,987,338  |  Slot: ₹99,367  |  Cash: ₹2,576
========================================================================

  EXITS (15)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  WELSPUNLIV  29     CONSUMP    01-Jun-15   51.7        97.9        1901   ₹87,800       +89.4%    366d  
  CHENNPETRO  —      OIL&GAS    03-Aug-15   139.4       152.8       758    ₹10,149       +9.6%     303d  
  BRITANNIA   169    FMCG       01-Jun-15   1,115.5     1,207.9     88     ₹8,126        +8.3%     366d  
  APLLTD      232    HEALTH     01-Jun-15   481.2       485.3       204    ₹830          +0.8%     366d  
  EICHERMOT   209    AUTO       01-Jun-15   1,740.7     1,744.2     56     ₹197          +0.2%     366d  
  SAMMAANCAP  53     FIN SVC    03-Aug-15   503.0       503.4       210    ₹93           +0.1%     303d  
  NIITLTD     —      OTHER      03-Aug-15   67.5        66.4        1566   ₹-1,693       -1.6%     303d  
  GILLETTE    167    FMCG       01-Apr-15   4,160.0     3,995.7     24     ₹-3,944       -4.0%     427d  
  AJANTPHARM  120    HEALTH     01-Jun-15   976.8       930.8       100    ₹-4,604       -4.7%     366d  
  MARUTI      75     AUTO       03-Aug-15   4,018.0     3,803.1     26     ₹-5,587       -5.3%     303d  
  BHARATRAS   97     FMCG       03-Aug-15   320.5       269.5       329    ₹-16,786      -15.9%    303d  
  EMAMILTD    —      FMCG       03-Aug-15   562.9       437.7       187    ₹-23,415      -22.2%    303d  
  PRAJIND     109    ENERGY     03-Aug-15   95.5        73.3        1106   ₹-24,589      -23.3%    303d  
  JKIL        240    INFRA      01-Apr-15   319.6       226.1       323    ₹-30,227      -29.3%    427d  
  EROSMEDIA   —      MEDIA      03-Aug-15   584.2       206.3       180    ₹-68,031      -64.7%    303d  

  ENTRIES (14)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BIOCON      1      HEALTH     3.687    0.80   +60.9%    +57.9%    117.0       848    ₹99,253       +10.2%    
  CHOLAFIN    2      FIN SVC    3.636    0.33   +70.6%    +55.4%    189.3       524    ₹99,191       +9.5%     
  VGUARD      3      CON DUR    3.456    0.60   +46.9%    +66.5%    93.1        1067   ₹99,357       +13.7%    
  MANAPPURAM  4      FIN SVC    3.016    1.19   +75.4%    +84.2%    44.9        2215   ₹99,343       +10.3%    
  RAMCOCEM    5      MFG        2.818    0.74   +62.3%    +35.7%    473.9       209    ₹99,036       +0.7%     
  OMAXE       6      REALTY     2.806    0.38   +15.0%    +15.5%    153.5       647    ₹99,328       +2.6%     
  UNOMINDA    7      AUTO       2.657    1.15   +110.9%   +50.0%    37.6        2643   ₹99,366       +4.7%     
  INDUSINDBK  9      PVT BNK    2.502    1.01   +29.8%    +36.6%    1,042.1     95     ₹98,996       +2.9%     
  FINCABLES   10     ENERGY     2.467    0.76   +39.1%    +46.3%    316.5       313    ₹99,069       +13.7%    
  ATUL        12     MFG        2.253    1.13   +63.4%    +37.9%    1,800.4     55     ₹99,024       +0.2%     
  HDFC        13     PVT BNK    2.247    0.77   +15.7%    +24.7%    265.8       373    ₹99,129       +1.9%     
  HDFCBANK    14     PVT BNK    2.247    0.77   +15.7%    +24.7%    265.8       373    ₹99,129       +1.9%     
  EPL         15     MFG        2.231    1.01   +45.4%    +37.6%    78.8        1261   ₹99,315       -0.4%     
  PVRINOX     16     MEDIA      2.128    0.73   +38.2%    +38.4%    878.7       113    ₹99,296       +6.6%     

  HOLDS (4)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  AEGISLOG    47     INFRA      01-Jun-15   62.2        102.6       1578   ₹63,635       +64.8%      +0.3%     
  BAJFINANCE  12     FIN SVC    03-Aug-15   54.7        73.7        1933   ₹36,845       +34.9%      +1.7%     
  HINDPETRO   15     OIL&GAS    03-Aug-15   81.6        86.0        1295   ₹5,691        +5.4%       +5.4%     
  RAJESHEXPO  56     CON DUR    03-Aug-15   561.3       558.5       188    ₹-516         -0.5%       -0.4%     

  AFTER: Invested ₹1,909,573 | Cash ₹76,116 | Total ₹1,985,689 | Positions 18/20 | Slot ₹99,367

========================================================================
  REBALANCE #10  —  01 Aug 2016
  NAV: ₹2,303,293  |  Slot: ₹115,165  |  Cash: ₹76,116
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJFINSV, MOTILALOFS, CHOLAHLDNG

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  AEGISLOG    102    INFRA      01-Jun-15   62.2        111.9       1578   ₹78,383       +79.8%    427d  
  EPL         65     MFG        01-Jun-16   78.8        90.7        1261   ₹15,001       +15.1%    61d   
  HDFC        67     PVT BNK    01-Jun-16   265.8       283.3       373    ₹6,525        +6.6%     61d   
  HDFCBANK    68     PVT BNK    01-Jun-16   265.8       283.3       373    ₹6,525        +6.6%     61d   
  ATUL        77     MFG        01-Jun-16   1,800.4     1,855.5     55     ₹3,026        +3.1%     61d   
  RAJESHEXPO  244    CON DUR    03-Aug-15   561.3       436.5       188    ₹-23,455      -22.2%    364d  

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  DALMIABHA   4      MFG        3.725    0.20   +103.9%   +69.7%    1,443.9     79     ₹114,068      +10.0%    
  MUTHOOTFIN  5      FIN SVC    3.632    0.67   +84.3%    +81.3%    298.8       385    ₹115,050      +20.6%    
  FINPIPE     10     INFRA      2.555    0.73   +78.6%    +33.4%    80.4        1431   ₹115,124      +5.3%     
  KAJARIACER  13     CON DUR    2.333    0.78   +65.8%    +20.0%    587.2       196    ₹115,098      +3.3%     
  GREENPLY    14     CON DUR    2.329    0.01   +613.7%   +20.1%    248.8       462    ₹114,963      +0.1%     
  SOMANYCERA  17     CON DUR    2.247    0.87   +54.1%    +45.9%    558.2       206    ₹114,999      +1.6%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  2      FIN SVC    03-Aug-15   54.7        108.4       1933   ₹103,847      +98.3%      +22.0%    
  MANAPPURAM  1      FIN SVC    01-Jun-16   44.9        71.2        2215   ₹58,378       +58.8%      +13.2%    
  HINDPETRO   12     OIL&GAS    03-Aug-15   81.6        121.9       1295   ₹52,250       +49.4%      +12.3%    
  PVRINOX     37     MEDIA      01-Jun-16   878.7       1,092.7     113    ₹24,174       +24.3%      +5.3%     
  VGUARD      3      CON DUR    01-Jun-16   93.1        113.9       1067   ₹22,124       +22.3%      +11.9%    
  CHOLAFIN    8      FIN SVC    01-Jun-16   189.3       222.8       524    ₹17,550       +17.7%      +12.9%    
  BIOCON      7      HEALTH     01-Jun-16   117.0       135.3       848    ₹15,495       +15.6%      +7.7%     
  FINCABLES   11     ENERGY     01-Jun-16   316.5       365.2       313    ₹15,242       +15.4%      +5.3%     
  RAMCOCEM    23     MFG        01-Jun-16   473.9       544.0       209    ₹14,655       +14.8%      +0.2%     
  INDUSINDBK  51     PVT BNK    01-Jun-16   1,042.1     1,138.1     95     ₹9,128        +9.2%       +5.1%     
  OMAXE       18     REALTY     01-Jun-16   153.5       160.3       647    ₹4,369        +4.4%       +1.1%     
  UNOMINDA    29     AUTO       01-Jun-16   37.6        37.6        2643   ₹42           +0.0%       +0.4%     

  AFTER: Invested ₹2,230,125 | Cash ₹72,350 | Total ₹2,302,474 | Positions 18/20 | Slot ₹115,165

========================================================================
  REBALANCE #11  —  03 Oct 2016
  NAV: ₹2,485,430  |  Slot: ₹124,272  |  Cash: ₹72,350
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJFINSV

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MANAPPURAM  2      FIN SVC    01-Jun-16   44.9        78.5        2215   ₹74,585       +75.1%    124d  
  PVRINOX     72     MEDIA      01-Jun-16   878.7       1,204.0     113    ₹36,755       +37.0%    124d  
  INDUSINDBK  87     PVT BNK    01-Jun-16   1,042.1     1,168.8     95     ₹12,042       +12.2%    124d  
  OMAXE       50     REALTY     01-Jun-16   153.5       169.5       647    ₹10,332       +10.4%    124d  
  GREENPLY    31     CON DUR    01-Aug-16   248.8       263.5       462    ₹6,757        +5.9%     63d   
  SOMANYCERA  95     CON DUR    01-Aug-16   558.2       579.8       206    ₹4,441        +3.9%     63d   
  FINPIPE     75     INFRA      01-Aug-16   80.4        77.8        1431   ₹-3,757       -3.3%     63d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HINDZINC    2      METAL      2.981    1.10   +120.3%   +40.9%    113.8       1092   ₹124,247      +11.7%    
  MRF         5      MFG        2.751    0.96   +25.7%    +57.3%    51,226.3    2      ₹102,453      +16.3%    
  IOC         8      OIL&GAS    2.608    0.93   +60.9%    +38.9%    53.5        2321   ₹124,223      +4.8%     
  CANFINHOME  9      FIN SVC    2.557    1.18   +105.2%   +38.8%    315.7       393    ₹124,088      +3.8%     
  TVSSRICHAK  11     AUTO       2.524    0.94   +46.3%    +62.5%    3,459.0     35     ₹121,066      +24.1%    
  VSTIND      12     FMCG       2.507    0.61   +60.3%    +40.0%    156.9       792    ₹124,246      +1.9%     
  BLUESTARCO  14     CON DUR    2.496    0.78   +73.1%    +31.4%    263.3       472    ₹124,270      +8.7%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  5      FIN SVC    03-Aug-15   54.7        104.9       1933   ₹97,142       +91.9%      +0.1%     
  HINDPETRO   28     OIL&GAS    03-Aug-15   81.6        124.8       1295   ₹55,924       +52.9%      +4.3%     
  VGUARD      7      CON DUR    01-Jun-16   93.1        127.4       1067   ₹36,537       +36.8%      +0.4%     
  BIOCON      4      HEALTH     01-Jun-16   117.0       157.9       848    ₹34,656       +34.9%      +3.1%     
  UNOMINDA    8      AUTO       01-Jun-16   37.6        50.3        2643   ₹33,576       +33.8%      +7.2%     
  DALMIABHA   1      MFG        01-Aug-16   1,443.9     1,919.0     79     ₹37,531       +32.9%      +7.9%     
  FINCABLES   26     ENERGY     01-Jun-16   316.5       402.5       313    ₹26,924       +27.2%      +1.3%     
  RAMCOCEM    27     MFG        01-Jun-16   473.9       596.8       209    ₹25,702       +26.0%      +4.3%     
  CHOLAFIN    12     FIN SVC    01-Jun-16   189.3       228.4       524    ₹20,479       +20.6%      +5.2%     
  KAJARIACER  23     CON DUR    01-Aug-16   587.2       648.4       196    ₹11,993       +10.4%      +3.9%     
  MUTHOOTFIN  32     FIN SVC    01-Aug-16   298.8       295.2       385    ₹-1,408       -1.2%       -1.6%     

  AFTER: Invested ₹2,374,471 | Cash ₹109,957 | Total ₹2,484,427 | Positions 18/20 | Slot ₹124,272

========================================================================
  REBALANCE #12  —  01 Dec 2016
  NAV: ₹2,296,498  |  Slot: ₹114,825  |  Cash: ₹109,957
========================================================================

  [REGIME OFF] Nifty 500 7,040.2 < EMA200 7,046.7 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  84     FIN SVC    03-Aug-15   54.7        88.3        1933   ₹64,948       +61.5%      -0.3%     
  HINDPETRO   15     OIL&GAS    03-Aug-15   81.6        129.0       1295   ₹61,341       +58.1%      -1.9%     
  UNOMINDA    20     AUTO       01-Jun-16   37.6        49.0        2643   ₹30,166       +30.4%      -3.4%     
  BIOCON      9      HEALTH     01-Jun-16   117.0       151.6       848    ₹29,267       +29.5%      +3.4%     
  VGUARD      34     CON DUR    01-Jun-16   93.1        116.3       1067   ₹24,710       +24.9%      -6.0%     
  RAMCOCEM    24     MFG        01-Jun-16   473.9       581.0       209    ₹22,388       +22.6%      +2.6%     
  FINCABLES   56     ENERGY     01-Jun-16   316.5       378.2       313    ₹19,294       +19.5%      +1.3%     
  DALMIABHA   27     MFG        01-Aug-16   1,443.9     1,595.5     79     ₹11,979       +10.5%      -5.1%     
  HINDZINC    1      METAL      03-Oct-16   113.8       123.8       1092   ₹10,996       +8.9%       +4.4%     
  CANFINHOME  26     FIN SVC    03-Oct-16   315.7       317.9       393    ₹852          +0.7%       +5.5%     
  IOC         37     OIL&GAS    03-Oct-16   53.5        52.8        2321   ₹-1,646       -1.3%       -1.5%     
  CHOLAFIN    118    FIN SVC    01-Jun-16   189.3       186.0       524    ₹-1,721       -1.7% ⚠     -5.0%     
  VSTIND      83     FMCG       03-Oct-16   156.9       151.1       792    ₹-4,567       -3.7%       -2.3%     
  MRF         10     MFG        03-Oct-16   51,226.3    48,256.4    2      ₹-5,940       -5.8%       -0.3%     
  TVSSRICHAK  17     AUTO       03-Oct-16   3,459.0     3,031.8     35     ₹-14,952      -12.4%      -0.6%     
  MUTHOOTFIN  74     FIN SVC    01-Aug-16   298.8       253.8       385    ₹-17,333      -15.1%      -3.8%     
  BLUESTARCO  63     CON DUR    03-Oct-16   263.3       222.9       472    ₹-19,056      -15.3%      -0.8%     
  KAJARIACER  198    CON DUR    01-Aug-16   587.2       487.2       196    ₹-19,599      -17.0% ⚠    -1.3%     
  ⚠  WAZ < 0 (momentum below universe mean): CHOLAFIN, KAJARIACER

  AFTER: Invested ₹2,186,542 | Cash ₹109,957 | Total ₹2,296,498 | Positions 18/20 | Slot ₹114,825

========================================================================
  REBALANCE #13  —  01 Feb 2017
  NAV: ₹2,588,066  |  Slot: ₹129,403  |  Cash: ₹109,957
========================================================================

  EXITS (11)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HINDPETRO   5      OIL&GAS    03-Aug-15   81.6        155.5       1295   ₹95,658       +90.5%    548d  
  BAJFINANCE  57     FIN SVC    03-Aug-15   54.7        102.9       1933   ₹93,307       +88.3%    548d  
  UNOMINDA    63     AUTO       01-Jun-16   37.6        57.1        2643   ₹51,566       +51.9%    245d  
  CANFINHOME  62     FIN SVC    03-Oct-16   315.7       356.7       393    ₹16,077       +13.0%    121d  
  VSTIND      71     FMCG       03-Oct-16   156.9       167.7       792    ₹8,567        +6.9%     121d  
  CHOLAFIN    117    FIN SVC    01-Jun-16   189.3       198.7       524    ₹4,951        +5.0%     245d  
  MRF         67     MFG        03-Oct-16   51,226.3    51,910.7    2      ₹1,369        +1.3%     121d  
  BLUESTARCO  99     CON DUR    03-Oct-16   263.3       241.9       472    ₹-10,095      -8.1%     121d  
  KAJARIACER  179    CON DUR    01-Aug-16   587.2       537.2       196    ₹-9,803       -8.5%     184d  
  MUTHOOTFIN  93     FIN SVC    01-Aug-16   298.8       272.1       385    ₹-10,277      -8.9%     184d  
  TVSSRICHAK  187    AUTO       03-Oct-16   3,459.0     2,914.4     35     ₹-19,062      -15.7%    121d  

  ENTRIES (12)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HATSUN      2      FMCG       3.129    0.28   +53.7%    +27.2%    297.5       435    ₹129,397      +5.6%     
  VINATIORGA  3      MFG        2.948    0.74   +84.4%    +23.4%    360.0       359    ₹129,249      +7.7%     
  VAKRANGEE   5      IT         2.810    0.53   +83.6%    +24.3%    137.7       940    ₹129,399      +4.5%     
  VTL         6      CONSUMP    2.807    0.71   +84.8%    +14.4%    232.3       557    ₹129,374      +8.7%     
  POWERGRID   8      ENERGY     2.791    0.74   +56.8%    +17.5%    73.8        1754   ₹129,385      +3.7%     
  JSWHL       10     FIN SVC    2.678    0.61   +51.1%    +26.5%    1,484.2     87     ₹129,125      +1.5%     
  FLFL        11     CONSUMP    2.660    1.06   +115.1%   +21.3%    172.6       749    ₹129,284      +10.5%    
  IGL         12     OIL&GAS    2.518    0.79   +70.6%    +12.3%    84.2        1537   ₹129,368      +3.1%     
  CCL         13     FMCG       2.465    0.62   +70.4%    +21.5%    274.5       471    ₹129,268      +6.0%     
  SHARDACROP  14     FMCG       2.423    0.95   +115.3%   +12.4%    424.7       304    ₹129,101      +0.9%     
  TATACOMM    15     CONSUMP    2.389    1.02   +81.8%    +15.4%    395.5       327    ₹129,342      +5.7%     
  APLAPOLLO   16     METAL      2.273    0.97   +55.6%    +19.7%    103.1       1255   ₹129,340      +6.3%     

  HOLDS (7)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VGUARD      24     CON DUR    01-Jun-16   93.1        141.1       1067   ₹51,228       +51.6%      +12.8%    
  RAMCOCEM    12     MFG        01-Jun-16   473.9       693.0       209    ₹45,809       +46.3%      +13.1%    
  BIOCON      10     HEALTH     01-Jun-16   117.0       166.6       848    ₹42,039       +42.4%      +2.4%     
  DALMIABHA   34     MFG        01-Aug-16   1,443.9     1,900.0     79     ₹36,031       +31.6%      +8.3%     
  FINCABLES   32     ENERGY     01-Jun-16   316.5       407.4       313    ₹28,456       +28.7%      +5.0%     
  IOC         6      OIL&GAS    03-Oct-16   53.5        66.6        2321   ₹30,289       +24.4%      +5.1%     
  HINDZINC    1      METAL      03-Oct-16   113.8       138.1       1092   ₹26,586       +21.4%      +6.3%     

  AFTER: Invested ₹2,571,324 | Cash ₹14,900 | Total ₹2,586,224 | Positions 19/20 | Slot ₹129,403

========================================================================
  REBALANCE #14  —  03 Apr 2017
  NAV: ₹2,826,947  |  Slot: ₹141,347  |  Cash: ₹14,900
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  FLFL        —      OTHER      01-Feb-17   172.6       270.2       749    ₹73,112       +56.6%    61d   
  RAMCOCEM    85     MFG        01-Jun-16   473.9       654.3       209    ₹37,711       +38.1%    306d  
  HINDZINC    38     METAL      03-Oct-16   113.8       143.4       1092   ₹32,372       +26.1%    182d  
  CCL         61     FMCG       01-Feb-17   274.5       306.1       471    ₹14,916       +11.5%    61d   
  VAKRANGEE   57     IT         01-Feb-17   137.7       147.3       940    ₹9,088        +7.0%     61d   
  SHARDACROP  77     FMCG       01-Feb-17   424.7       452.0       304    ₹8,315        +6.4%     61d   
  IGL         62     OIL&GAS    01-Feb-17   84.2        88.9        1537   ₹7,222        +5.6%     61d   
  JSWHL       —      OTHER      01-Feb-17   1,484.2     1,524.8     87     ₹3,532        +2.7%     61d   
  TATACOMM    69     CONSUMP    01-Feb-17   395.5       401.3       327    ₹1,873        +1.4%     61d   
  POWERGRID   114    ENERGY     01-Feb-17   73.8        70.7        1754   ₹-5,420       -4.2%     61d   

  ENTRIES (10)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BLUESTARCO  2      CON DUR    3.139    0.83   +87.5%    +50.3%    327.9       431    ₹141,338      +15.7%    
  BAJAJFINSV  3      FIN SVC    2.991    1.07   +137.6%   +41.6%    407.9       346    ₹141,135      +2.7%     
  BHARATRAS   4      FMCG       2.955    0.98   +223.1%   +69.9%    753.1       187    ₹140,825      +9.5%     
  AVANTIFEED  6      FMCG       2.718    0.97   +96.5%    +58.6%    233.4       605    ₹141,199      +9.4%     
  UNOMINDA    7      AUTO       2.647    1.13   +132.7%   +46.5%    71.3        1983   ₹141,299      +2.1%     
  FINPIPE     9      INFRA      2.519    0.68   +70.3%    +35.7%    101.0       1400   ₹141,344      +7.4%     
  GUJGASLTD   11     OIL&GAS    2.408    0.54   +44.0%    +44.8%    142.3       992    ₹141,207      +4.2%     
  NATCOPHARM  12     HEALTH     2.356    0.58   +85.9%    +50.4%    803.3       175    ₹140,572      +8.1%     
  KARURVYSYA  13     PVT BNK    2.344    0.64   +36.7%    +42.4%    73.3        1927   ₹141,311      +10.0%    
  AEGISLOG    14     INFRA      2.331    1.08   +107.3%   +52.7%    180.0       785    ₹141,285      -1.8%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VGUARD      2      CON DUR    01-Jun-16   93.1        168.4       1067   ₹80,353       +80.9%      +2.5%     
  BIOCON      19     HEALTH     01-Jun-16   117.0       185.3       848    ₹57,874       +58.3%      +0.9%     
  FINCABLES   37     ENERGY     01-Jun-16   316.5       486.1       313    ₹53,094       +53.6%      +10.3%    
  DALMIABHA   13     MFG        01-Aug-16   1,443.9     1,969.1     79     ₹41,489       +36.4%      +2.4%     
  IOC         30     OIL&GAS    03-Oct-16   53.5        70.8        2321   ₹40,138       +32.3%      +2.1%     
  HATSUN      8      FMCG       01-Feb-17   297.5       348.3       435    ₹22,103       +17.1%      +2.5%     
  APLAPOLLO   31     METAL      01-Feb-17   103.1       115.5       1255   ₹15,616       +12.1%      +5.3%     
  VTL         41     CONSUMP    01-Feb-17   232.3       245.7       557    ₹7,478        +5.8%       +2.5%     
  VINATIORGA  54     MFG        01-Feb-17   360.0       360.8       359    ₹295          +0.2%       +1.5%     

  AFTER: Invested ₹2,783,285 | Cash ₹41,986 | Total ₹2,825,271 | Positions 19/20 | Slot ₹141,347

========================================================================
  REBALANCE #15  —  01 Jun 2017
  NAV: ₹3,127,904  |  Slot: ₹156,395  |  Cash: ₹41,986
========================================================================
  [SECTOR CAP≤4] dropped: BALKRISIND

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  FINCABLES   99     ENERGY     01-Jun-16   316.5       458.0       313    ₹44,275       +44.7%    365d  
  UNOMINDA    8      AUTO       03-Apr-17   71.3        101.3       1983   ₹59,539       +42.1%    59d   
  BIOCON      241    HEALTH     01-Jun-16   117.0       156.7       848    ₹33,656       +33.9%    365d  
  AEGISLOG    115    INFRA      03-Apr-17   180.0       198.3       785    ₹14,412       +10.2%    59d   
  FINPIPE     58     INFRA      03-Apr-17   101.0       107.4       1400   ₹9,076        +6.4%     59d   
  KARURVYSYA  107    PVT BNK    03-Apr-17   73.3        74.6        1927   ₹2,488        +1.8%     59d   
  VTL         183    CONSUMP    01-Feb-17   232.3       234.7       557    ₹1,365        +1.1%     120d  
  GUJGASLTD   67     OIL&GAS    03-Apr-17   142.3       142.8       992    ₹466          +0.3%     59d   
  BHARATRAS   80     FMCG       03-Apr-17   753.1       749.6       187    ₹-656         -0.5%     59d   
  BLUESTARCO  118    CON DUR    03-Apr-17   327.9       292.4       431    ₹-15,328      -10.8%    59d   

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  VAKRANGEE   2      IT         3.117    0.34   +124.2%   +22.6%    174.1       898    ₹156,356      +9.6%     
  KEC         3      INFRA      3.111    1.16   +108.3%   +59.7%    249.3       627    ₹156,342      +9.6%     
  BIRLACORPN  4      MFG        2.893    0.81   +129.2%   +26.5%    825.7       189    ₹156,057      +12.1%    
  SWARAJENG   6      AUTO       2.602    0.68   +75.0%    +34.6%    1,398.1     111    ₹155,189      +3.5%     
  HDFC        8      PVT BNK    2.444    0.66   +39.9%    +17.2%    371.3       421    ₹156,331      +2.9%     
  HDFCBANK    9      PVT BNK    2.444    0.66   +39.9%    +17.2%    371.3       421    ₹156,331      +2.9%     
  TVSMOTOR    10     AUTO       2.382    0.90   +81.3%    +24.9%    508.7       307    ₹156,164      +2.1%     
  CHAMBLFERT  11     MFG        2.329    1.08   +91.0%    +48.4%    100.5       1556   ₹156,369      +5.4%     
  HINDUNILVR  13     FMCG       2.274    0.52   +32.9%    +26.5%    940.4       166    ₹156,114      +7.3%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VGUARD      60     CON DUR    01-Jun-16   93.1        176.8       1067   ₹89,249       +89.8%      -3.8%     
  AVANTIFEED  1      FMCG       03-Apr-17   233.4       427.4       605    ₹117,387      +83.1%      +14.8%    
  DALMIABHA   15     MFG        01-Aug-16   1,443.9     2,419.4     79     ₹77,062       +67.6%      +1.8%     
  IOC         40     OIL&GAS    03-Oct-16   53.5        76.9        2321   ₹54,239       +43.7%      -3.3%     
  APLAPOLLO   23     METAL      01-Feb-17   103.1       139.3       1255   ₹45,492       +35.2%      +3.3%     
  HATSUN      13     FMCG       01-Feb-17   297.5       397.6       435    ₹43,542       +33.7%      -0.9%     
  VINATIORGA  39     MFG        01-Feb-17   360.0       438.5       359    ₹28,164       +21.8%      +5.6%     
  NATCOPHARM  46     HEALTH     03-Apr-17   803.3       877.3       175    ₹12,947       +9.2%       +3.1%     
  BAJAJFINSV  50     FIN SVC    03-Apr-17   407.9       418.6       346    ₹3,697        +2.6%       -0.1%     

  AFTER: Invested ₹3,025,572 | Cash ₹100,664 | Total ₹3,126,236 | Positions 18/20 | Slot ₹156,395

========================================================================
  REBALANCE #16  —  01 Aug 2017
  NAV: ₹3,413,803  |  Slot: ₹170,690  |  Cash: ₹100,664
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VGUARD      113    CON DUR    01-Jun-16   93.1        174.9       1067   ₹87,294       +87.9%    426d  
  IOC         209    OIL&GAS    03-Oct-16   53.5        68.8        2321   ₹35,481       +28.6%    302d  
  BAJAJFINSV  38     FIN SVC    03-Apr-17   407.9       504.6       346    ₹33,444       +23.7%    120d  
  CHAMBLFERT  23     MFG        01-Jun-17   100.5       111.0       1556   ₹16,282       +10.4%    61d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RAJESHEXPO  8      CON DUR    2.754    0.75   +64.8%    +16.7%    714.7       238    ₹170,093      +4.4%     
  IGL         9      OIL&GAS    2.720    0.80   +93.2%    +12.9%    104.2       1637   ₹170,612      +4.8%     
  MHRIL       13     CONSUMP    2.366    0.94   +40.9%    +35.8%    262.9       649    ₹170,591      -1.3%     
  RELIANCE    14     OIL&GAS    2.327    0.69   +58.0%    +18.0%    352.5       484    ₹170,634      +3.9%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  AVANTIFEED  1      FMCG       03-Apr-17   233.4       511.0       605    ₹167,980      +119.0%     +7.1%     
  DALMIABHA   57     MFG        01-Aug-16   1,443.9     2,588.4     79     ₹90,414       +79.3%      -1.1%     
  APLAPOLLO   67     METAL      01-Feb-17   103.1       151.0       1255   ₹60,156       +46.5%      -1.0%     
  HATSUN      43     FMCG       01-Feb-17   297.5       420.1       435    ₹53,326       +41.2%      +0.1%     
  VINATIORGA  22     MFG        01-Feb-17   360.0       500.6       359    ₹50,464       +39.0%      +2.4%     
  KEC         6      INFRA      01-Jun-17   249.3       288.3       627    ₹24,412       +15.6%      +5.8%     
  VAKRANGEE   3      IT         01-Jun-17   174.1       199.9       898    ₹23,132       +14.8%      +1.3%     
  SWARAJENG   27     AUTO       01-Jun-17   1,398.1     1,604.2     111    ₹22,877       +14.7%      -2.2%     
  NATCOPHARM  76     HEALTH     03-Apr-17   803.3       899.3       175    ₹16,808       +12.0%      -1.8%     
  TVSMOTOR    12     AUTO       01-Jun-17   508.7       568.4       307    ₹18,330       +11.7%      +4.4%     
  HDFC        8      PVT BNK    01-Jun-17   371.3       412.5       421    ₹17,330       +11.1%      +4.2%     
  HDFCBANK    9      PVT BNK    01-Jun-17   371.3       412.5       421    ₹17,330       +11.1%      +4.2%     
  HINDUNILVR  14     FMCG       01-Jun-17   940.4       1,017.0     166    ₹12,705       +8.1%       +2.8%     
  BIRLACORPN  19     MFG        01-Jun-17   825.7       887.0       189    ₹11,581       +7.4%       +2.5%     

  AFTER: Invested ₹3,301,483 | Cash ₹111,509 | Total ₹3,412,993 | Positions 18/20 | Slot ₹170,690

========================================================================
  REBALANCE #17  —  03 Oct 2017
  NAV: ₹3,602,513  |  Slot: ₹180,126  |  Cash: ₹111,509
========================================================================

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  AVANTIFEED  2      FMCG       03-Apr-17   233.4       679.6       605    ₹269,967      +191.2%   183d  
  DALMIABHA   94     MFG        01-Aug-16   1,443.9     2,715.0     79     ₹100,417      +88.0%    428d  
  VINATIORGA  111    MFG        01-Feb-17   360.0       466.3       359    ₹38,166       +29.5%    244d  
  KEC         19     INFRA      01-Jun-17   249.3       291.4       627    ₹26,357       +16.9%    124d  
  SWARAJENG   148    AUTO       01-Jun-17   1,398.1     1,518.3     111    ₹13,341       +8.6%     124d  
  BIRLACORPN  109    MFG        01-Jun-17   825.7       848.8       189    ₹4,358        +2.8%     124d  
  NATCOPHARM  248    HEALTH     03-Apr-17   803.3       738.5       175    ₹-11,338      -8.1%     183d  
  MHRIL       268    CONSUMP    01-Aug-17   262.9       223.9       649    ₹-25,309      -14.8%    63d   

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  AHLUCONT    3      INFRA      3.351    -0.23  +188.5%   +188.5%   300.4       599    ₹179,930      +10.8%    
  BAJAJHLDNG  6      FIN SVC    2.904    0.80   +55.8%    +39.2%    2,533.4     71     ₹179,871      +3.8%     
  KRBL        7      FMCG       2.896    1.02   +101.7%   +30.9%    465.9       386    ₹179,839      +4.6%     
  GUJALKALI   8      MFG        2.816    1.18   +76.7%    +42.0%    475.4       378    ₹179,708      +8.1%     
  BLUESTARCO  13     CON DUR    2.349    0.85   +46.1%    +30.9%    360.1       500    ₹180,032      +1.6%     
  GAIL        14     OIL&GAS    2.296    0.82   +58.0%    +19.5%    76.7        2348   ₹180,121      +8.2%     
  HERITGFOOD  15     FMCG       2.226    1.07   +61.7%    +28.4%    339.8       530    ₹180,106      -1.3%     
  GODREJPROP  16     REALTY     2.185    1.18   +84.8%    +17.2%    614.5       293    ₹180,063      +2.2%     
  MGL         17     OIL&GAS    2.182    0.81   +77.3%    +10.1%    877.9       205    ₹179,979      -0.5%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  APLAPOLLO   27     METAL      01-Feb-17   103.1       173.6       1255   ₹88,589       +68.5%      +5.0%     
  HATSUN      10     FMCG       01-Feb-17   297.5       474.6       435    ₹77,076       +59.6%      +5.4%     
  VAKRANGEE   13     IT         01-Jun-17   174.1       222.4       898    ₹43,401       +27.8%      +0.6%     
  IGL         3      OIL&GAS    01-Aug-17   104.2       129.7       1637   ₹41,686       +24.4%      +4.3%     
  TVSMOTOR    20     AUTO       01-Jun-17   508.7       622.8       307    ₹35,036       +22.4%      +2.2%     
  RAJESHEXPO  9      CON DUR    01-Aug-17   714.7       815.3       238    ₹23,954       +14.1%      +6.7%     
  HDFC        23     PVT BNK    01-Jun-17   371.3       415.2       421    ₹18,456       +11.8%      +0.2%     
  HDFCBANK    24     PVT BNK    01-Jun-17   371.3       415.2       421    ₹18,456       +11.8%      +0.2%     
  HINDUNILVR  62     FMCG       01-Jun-17   940.4       1,027.7     166    ₹14,488       +9.3%       -2.7%     
  RELIANCE    41     OIL&GAS    01-Aug-17   352.5       351.0       484    ₹-734         -0.4%       -1.7%     

  AFTER: Invested ₹3,531,429 | Cash ₹69,161 | Total ₹3,600,590 | Positions 19/20 | Slot ₹180,126

========================================================================
  REBALANCE #18  —  01 Dec 2017
  NAV: ₹4,026,889  |  Slot: ₹201,344  |  Cash: ₹69,161
========================================================================

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  GODREJPROP  32     REALTY     03-Oct-17   614.5       721.2       293    ₹31,234       +17.3%    59d   
  HINDUNILVR  92     FMCG       01-Jun-17   940.4       1,090.3     166    ₹24,884       +15.9%    183d  
  HERITGFOOD  123    FMCG       03-Oct-17   339.8       362.1       530    ₹11,791       +6.5%     59d   
  RAJESHEXPO  80     CON DUR    01-Aug-17   714.7       751.9       238    ₹8,854        +5.2%     122d  
  GAIL        77     OIL&GAS    03-Oct-17   76.7        80.3        2348   ₹8,439        +4.7%     59d   
  MGL         134    OIL&GAS    03-Oct-17   877.9       888.1       205    ₹2,078        +1.2%     59d   
  BAJAJHLDNG  172    FIN SVC    03-Oct-17   2,533.4     2,538.6     71     ₹371          +0.2%     59d   
  BLUESTARCO  167    CON DUR    03-Oct-17   360.1       345.2       500    ₹-7,453       -4.1%     59d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IFBIND      1      CON DUR    5.134    0.48   +244.0%   +105.3%   1,468.3     137    ₹201,157      +19.0%    
  UNOMINDA    2      AUTO       4.532    1.01   +312.5%   +42.2%    202.1       996    ₹201,307      +14.1%    
  BBTC        5      FMCG       3.179    1.06   +197.5%   +48.2%    1,478.3     136    ₹201,055      -0.9%     
  SOLARINDS   8      DEFENCE    2.865    0.60   +72.9%    +30.5%    1,121.0     179    ₹200,657      +2.0%     
  EVEREADY    9      CON DUR    2.755    0.68   +102.3%   +42.0%    428.7       469    ₹201,061      +9.1%     
  GILLETTE    10     FMCG       2.642    0.43   +58.2%    +22.0%    5,834.9     34     ₹198,386      +6.2%     
  SOBHA       15     REALTY     2.278    1.10   +139.1%   +52.1%    547.0       368    ₹201,291      +9.6%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HATSUN      13     FMCG       01-Feb-17   297.5       596.2       435    ₹129,968      +100.4%     +3.9%     
  APLAPOLLO   22     METAL      01-Feb-17   103.1       191.3       1255   ₹110,690      +85.6%      +4.2%     
  VAKRANGEE   8      IT         01-Jun-17   174.1       320.5       898    ₹131,439      +84.1%      +7.4%     
  GUJALKALI   14     MFG        03-Oct-17   475.4       656.0       378    ₹68,258       +38.0%      +2.0%     
  TVSMOTOR    23     AUTO       01-Jun-17   508.7       692.6       307    ₹56,454       +36.2%      +0.9%     
  IGL         24     OIL&GAS    01-Aug-17   104.2       139.5       1637   ₹57,793       +33.9%      +1.1%     
  AHLUCONT    9      INFRA      03-Oct-17   300.4       373.5       599    ₹43,812       +24.3%      +6.6%     
  KRBL        21     FMCG       03-Oct-17   465.9       570.9       386    ₹40,530       +22.5%      -1.3%     
  HDFC        41     PVT BNK    01-Jun-17   371.3       424.2       421    ₹22,258       +14.2%      +0.4%     
  HDFCBANK    42     PVT BNK    01-Jun-17   371.3       424.2       421    ₹22,258       +14.2%      +0.4%     
  RELIANCE    51     OIL&GAS    01-Aug-17   352.5       400.2       484    ₹23,048       +13.5%      -1.3%     

  AFTER: Invested ₹3,876,065 | Cash ₹149,156 | Total ₹4,025,221 | Positions 18/20 | Slot ₹201,344

========================================================================
  REBALANCE #19  —  01 Feb 2018
  NAV: ₹3,880,612  |  Slot: ₹194,031  |  Cash: ₹149,156
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VAKRANGEE   100    IT         01-Jun-17   174.1       262.4       898    ₹79,274       +50.7%    245d  
  GUJALKALI   131    MFG        03-Oct-17   475.4       646.3       378    ₹64,610       +36.0%    121d  
  IGL         139    OIL&GAS    01-Aug-17   104.2       133.1       1637   ₹47,290       +27.7%    184d  
  TVSMOTOR    110    AUTO       01-Jun-17   508.7       641.5       307    ₹40,771       +26.1%    245d  
  KRBL        172    FMCG       03-Oct-17   465.9       560.9       386    ₹36,672       +20.4%    121d  
  RELIANCE    74     OIL&GAS    01-Aug-17   352.5       415.0       484    ₹30,230       +17.7%    184d  
  AHLUCONT    94     INFRA      03-Oct-17   300.4       352.1       599    ₹30,988       +17.2%    121d  
  SOBHA       83     REALTY     01-Dec-17   547.0       524.0       368    ₹-8,472       -4.2%     62d   
  SOLARINDS   81     DEFENCE    01-Dec-17   1,121.0     1,054.9     179    ₹-11,832      -5.9%     62d   
  BBTC        93     FMCG       01-Dec-17   1,478.3     1,383.9     136    ₹-12,846      -6.4%     62d   

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HGS         2      IT         3.212    0.82   +83.9%    +63.1%    403.7       480    ₹193,776      +2.9%     
  BIOCON      4      HEALTH     2.803    1.15   +82.6%    +62.8%    305.0       636    ₹193,954      +6.0%     
  BALKRISIND  5      MFG        2.711    0.80   +100.6%   +29.2%    1,047.3     185    ₹193,742      -1.5%     
  MARUTI      8      AUTO       2.512    0.99   +60.5%    +14.3%    8,730.3     22     ₹192,067      -0.1%     
  LT          9      INFRA      2.502    1.17   +55.4%    +20.2%    1,278.9     151    ₹193,109      +6.1%     
  MPHASIS     10     IT         2.436    0.28   +61.5%    +26.3%    722.3       268    ₹193,574      +9.6%     
  3MINDIA     11     MNC        2.407    0.80   +54.6%    +33.2%    17,913.1    10     ₹179,131      +4.1%     
  HINDUNILVR  12     FMCG       2.353    0.66   +62.7%    +10.5%    1,195.6     162    ₹193,693      +0.3%     
  BHARATRAS   14     FMCG       2.266    0.84   +86.6%    +44.6%    1,034.2     187    ₹193,398      -2.6%     
  PHOENIXLTD  16     REALTY     2.154    0.53   +86.5%    +25.4%    331.6       585    ₹193,980      +8.0%     
  RAJESHEXPO  18     CON DUR    2.003    0.34   +63.5%    +5.8%     814.1       238    ₹193,752      -0.1%     

  HOLDS (8)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  APLAPOLLO   39     METAL      01-Feb-17   103.1       205.6       1255   ₹128,629      +99.5%      -6.5%     
  HATSUN      62     FMCG       01-Feb-17   297.5       535.3       435    ₹103,478      +80.0%      -3.2%     
  HDFC        12     PVT BNK    01-Jun-17   371.3       457.0       421    ₹36,072       +23.1%      +2.7%     
  HDFCBANK    13     PVT BNK    01-Jun-17   371.3       457.0       421    ₹36,072       +23.1%      +2.7%     
  UNOMINDA    1      AUTO       01-Dec-17   202.1       205.6       996    ₹3,486        +1.7%       +0.2%     
  GILLETTE    29     FMCG       01-Dec-17   5,834.9     5,816.3     34     ₹-634         -0.3%       -1.5%     
  EVEREADY    30     CON DUR    01-Dec-17   428.7       409.1       469    ₹-9,171       -4.6%       -4.3%     
  IFBIND      8      CON DUR    01-Dec-17   1,468.3     1,229.5     137    ₹-32,716      -16.3%      -9.9%     

  AFTER: Invested ₹3,752,702 | Cash ₹125,400 | Total ₹3,878,102 | Positions 19/20 | Slot ₹194,031

========================================================================
  REBALANCE #20  —  02 Apr 2018
  NAV: ₹3,689,011  |  Slot: ₹184,451  |  Cash: ₹125,400
========================================================================
  [SECTOR CAP≤4] dropped: VBL

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HATSUN      179    FMCG       01-Feb-17   297.5       471.9       435    ₹75,883       +58.6%    425d  
  BALKRISIND  84     MFG        01-Feb-18   1,047.3     1,002.1     185    ₹-8,355       -4.3%     60d   
  PHOENIXLTD  62     REALTY     01-Feb-18   331.6       299.5       585    ₹-18,747      -9.7%     60d   
  RAJESHEXPO  131    CON DUR    01-Feb-18   814.1       733.3       238    ₹-19,231      -9.9%     60d   
  UNOMINDA    28     AUTO       01-Dec-17   202.1       179.6       996    ₹-22,420      -11.1%    122d  
  EVEREADY    126    CON DUR    01-Dec-17   428.7       373.6       469    ₹-25,822      -12.8%    122d  
  HGS         111    IT         01-Feb-18   403.7       351.3       480    ₹-25,154      -13.0%    60d   
  IFBIND      79     CON DUR    01-Dec-17   1,468.3     1,161.9     137    ₹-41,970      -20.9%    122d  

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BRITANNIA   1      FMCG       2.962    0.66   +55.7%    +7.9%     2,259.4     81     ₹183,011      +4.5%     
  TITAN       2      CON DUR    2.878    1.17   +111.8%   +11.2%    917.9       200    ₹183,575      +6.9%     
  CYIENT      3      IT         2.746    0.80   +48.6%    +21.5%    562.3       328    ₹184,437      +5.5%     
  ASTRAL      8      MFG        2.558    0.64   +63.4%    +10.0%    410.2       449    ₹184,175      +7.2%     
  CHOLAFIN    9      FIN SVC    2.552    0.97   +49.0%    +14.9%    287.4       641    ₹184,215      +2.6%     
  SUNTECK     10     REALTY     2.552    1.08   +124.4%   -0.1%     407.3       452    ₹184,103      +2.6%     
  INDUSINDBK  12     PVT BNK    2.417    0.69   +30.4%    +9.3%     1,717.3     107    ₹183,753      +3.7%     
  PGHL        15     HEALTH     2.324    0.72   +49.0%    +17.0%    1,115.5     165    ₹184,060      +1.0%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  APLAPOLLO   41     METAL      01-Feb-17   103.1       198.0       1255   ₹119,153      +92.1%      +3.6%     
  HDFC        16     PVT BNK    01-Jun-17   371.3       443.3       421    ₹30,279       +19.4%      +2.9%     
  HDFCBANK    17     PVT BNK    01-Jun-17   371.3       443.3       421    ₹30,279       +19.4%      +2.9%     
  BHARATRAS   55     FMCG       01-Feb-18   1,034.2     1,117.2     187    ₹15,512       +8.0%       +1.7%     
  3MINDIA     33     MNC        01-Feb-18   17,913.1    18,555.8    10     ₹6,427        +3.6%       +1.2%     
  GILLETTE    23     FMCG       01-Dec-17   5,834.9     5,855.7     34     ₹706          +0.4%       +0.8%     
  HINDUNILVR  32     FMCG       01-Feb-18   1,195.6     1,178.2     162    ₹-2,818       -1.5%       +2.2%     
  BIOCON      48     HEALTH     01-Feb-18   305.0       294.8       636    ₹-6,454       -3.3%       +0.9%     
  MPHASIS     14     IT         01-Feb-18   722.3       695.3       268    ₹-7,229       -3.7%       +0.0%     
  MARUTI      63     AUTO       01-Feb-18   8,730.3     8,364.8     22     ₹-8,042       -4.2%       +2.1%     
  LT          42     INFRA      01-Feb-18   1,278.9     1,173.7     151    ₹-15,874      -8.2%       +2.5%     

  AFTER: Invested ₹3,612,583 | Cash ₹74,681 | Total ₹3,687,264 | Positions 19/20 | Slot ₹184,451

========================================================================
  REBALANCE #21  —  01 Jun 2018
  NAV: ₹4,035,919  |  Slot: ₹201,796  |  Cash: ₹74,681
========================================================================
  [SECTOR CAP≤4] dropped: COLPAL

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  APLAPOLLO   155    METAL      01-Feb-17   103.1       175.6       1255   ₹91,034       +70.4%    485d  
  BIOCON      —      HEALTH     01-Feb-18   305.0       319.7       636    ₹9,367        +4.8%     120d  
  3MINDIA     116    MNC        01-Feb-18   17,913.1    17,979.5    10     ₹664          +0.4%     120d  
  GILLETTE    46     FMCG       01-Dec-17   5,834.9     5,814.6     34     ₹-691         -0.3%     182d  
  LT          83     INFRA      01-Feb-18   1,278.9     1,205.8     151    ₹-11,038      -5.7%     120d  
  MARUTI      89     AUTO       01-Feb-18   8,730.3     8,179.7     22     ₹-12,114      -6.3%     120d  
  SUNTECK     —      REALTY     02-Apr-18   407.3       379.8       452    ₹-12,417      -6.7%     60d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    2      CONSUMP    3.574    0.90   +166.5%   +23.0%    245.3       822    ₹201,633      -1.3%     
  COFORGE     5      IT         2.891    0.92   +120.6%   +32.7%    201.6       1001   ₹201,754      +3.3%     
  PIDILITIND  6      MFG        2.809    0.81   +49.2%    +24.2%    538.3       374    ₹201,319      +0.5%     
  TECHM       7      IT         2.705    0.26   +89.2%    +14.4%    516.6       390    ₹201,479      +2.4%     
  DABUR       9      FMCG       2.693    0.59   +39.8%    +19.0%    354.9       568    ₹201,610      +3.3%     
  KOTAKBANK   10     PVT BNK    2.656    0.76   +36.2%    +20.9%    262.2       769    ₹201,653      +3.5%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  PGHL        2      HEALTH     02-Apr-18   1,115.5     1,901.3     165    ₹129,650      +70.4%      +22.0%    
  BHARATRAS   26     FMCG       01-Feb-18   1,034.2     1,449.7     187    ₹77,691       +40.2%      +1.3%     
  HDFC        19     PVT BNK    01-Jun-17   371.3       487.5       421    ₹48,917       +31.3%      +4.9%     
  HDFCBANK    20     PVT BNK    01-Jun-17   371.3       487.5       421    ₹48,917       +31.3%      +4.9%     
  MPHASIS     12     IT         01-Feb-18   722.3       868.5       268    ₹39,170       +20.2%      +0.6%     
  HINDUNILVR  5      FMCG       01-Feb-18   1,195.6     1,385.4     162    ₹30,737       +15.9%      +2.3%     
  BRITANNIA   4      FMCG       02-Apr-18   2,259.4     2,571.5     81     ₹25,278       +13.8%      +2.9%     
  ASTRAL      25     MFG        02-Apr-18   410.2       446.0       449    ₹16,082       +8.7%       +4.1%     
  CHOLAFIN    43     FIN SVC    02-Apr-18   287.4       308.5       641    ₹13,515       +7.3%       +1.4%     
  CYIENT      40     IT         02-Apr-18   562.3       602.3       328    ₹13,128       +7.1%       -4.5%     
  INDUSINDBK  31     PVT BNK    02-Apr-18   1,717.3     1,822.8     107    ₹11,282       +6.1%       +0.9%     
  TITAN       34     CON DUR    02-Apr-18   917.9       875.0       200    ₹-8,575       -4.7%       -2.9%     

  AFTER: Invested ₹3,835,793 | Cash ₹198,690 | Total ₹4,034,483 | Positions 18/20 | Slot ₹201,796

========================================================================
  REBALANCE #22  —  01 Aug 2018
  NAV: ₹4,266,018  |  Slot: ₹213,301  |  Cash: ₹198,690
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BHARATRAS   24     FMCG       01-Feb-18   1,034.2     1,954.0     187    ₹172,008      +88.9%    181d  
  PGHL        61     HEALTH     02-Apr-18   1,115.5     1,645.2     165    ₹87,403       +47.5%    121d  
  INDUSINDBK  55     PVT BNK    02-Apr-18   1,717.3     1,919.4     107    ₹21,627       +11.8%    121d  
  CYIENT      130    IT         02-Apr-18   562.3       573.0       328    ₹3,506        +1.9%     121d  
  KOTAKBANK   46     PVT BNK    01-Jun-18   262.2       261.3       769    ₹-708         -0.4%     61d   
  CHOLAFIN    138    FIN SVC    02-Apr-18   287.4       285.5       641    ₹-1,237       -0.7%     121d  
  TITAN       57     CON DUR    02-Apr-18   917.9       895.8       200    ₹-4,421       -2.4%     121d  

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BAJFINANCE  2      FIN SVC    3.152    1.17   +60.3%    +46.8%    264.0       807    ₹213,060      +5.8%     
  GLAXO       3      HEALTH     3.109    0.53   +32.0%    +36.8%    1,303.6     163    ₹212,479      +7.0%     
  SANOFI      7      HEALTH     2.858    0.55   +42.6%    +23.9%    4,304.8     49     ₹210,937      +6.9%     
  RELIANCE    8      OIL&GAS    2.841    1.07   +50.6%    +25.8%    527.8       404    ₹213,223      +8.4%     
  TCS         9      IT         2.629    0.15   +61.6%    +14.7%    1,618.1     131    ₹211,965      +1.6%     
  PAGEIND     10     MFG        2.610    1.15   +82.6%    +26.9%    27,338.4    7      ₹191,369      +4.9%     
  RHIM        11     METAL      2.590    1.01   +59.2%    +46.0%    219.2       972    ₹213,095      +23.6%    
  RELAXO      13     CON DUR    2.568    0.59   +70.7%    +19.8%    398.8       534    ₹212,965      +5.1%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MPHASIS     9      IT         01-Feb-18   722.3       1,006.5     268    ₹76,165       +39.3%      +6.6%     
  HDFC        36     PVT BNK    01-Jun-17   371.3       498.6       421    ₹53,594       +34.3%      +0.0%     
  HDFCBANK    37     PVT BNK    01-Jun-17   371.3       498.6       421    ₹53,594       +34.3%      +0.0%     
  BRITANNIA   2      FMCG       02-Apr-18   2,259.4     2,882.3     81     ₹50,457       +27.6%      +2.0%     
  HINDUNILVR  6      FMCG       01-Feb-18   1,195.6     1,523.6     162    ₹53,130       +27.4%      +3.2%     
  ASTRAL      22     MFG        02-Apr-18   410.2       487.9       449    ₹34,870       +18.9%      +2.6%     
  DABUR       21     FMCG       01-Jun-18   354.9       403.3       568    ₹27,440       +13.6%      +11.8%    
  COFORGE     11     IT         01-Jun-18   201.6       227.3       1001   ₹25,821       +12.8%      +6.2%     
  JUBLFOOD    20     CONSUMP    01-Jun-18   245.3       273.8       822    ₹23,403       +11.6%      -1.1%     
  PIDILITIND  40     MFG        01-Jun-18   538.3       543.6       374    ₹1,996        +1.0%       +3.8%     
  TECHM       29     IT         01-Jun-18   516.6       513.2       390    ₹-1,323       -0.7%       +5.1%     

  AFTER: Invested ₹4,153,152 | Cash ₹110,872 | Total ₹4,264,024 | Positions 19/20 | Slot ₹213,301

========================================================================
  REBALANCE #23  —  01 Oct 2018
  NAV: ₹4,123,152  |  Slot: ₹206,158  |  Cash: ₹110,872
========================================================================
  [SECTOR CAP≤4] dropped: INFY, LTTS, ZENSARTECH, WIPRO

  [REGIME OFF] Nifty 500 9,165.5 < EMA200 9,315.0 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MPHASIS     15     IT         01-Feb-18   722.3       968.9       268    ₹66,091       +34.1%      -3.5%     
  HDFC        121    PVT BNK    01-Jun-17   371.3       470.2       421    ₹41,608       +26.6% ⚠    +1.2%     
  HDFCBANK    122    PVT BNK    01-Jun-17   371.3       470.2       421    ₹41,608       +26.6% ⚠    +1.2%     
  HINDUNILVR  37     FMCG       01-Feb-18   1,195.6     1,442.8     162    ₹40,037       +20.7%      -0.1%     
  DABUR       17     FMCG       01-Jun-18   354.9       409.8       568    ₹31,149       +15.5%      -2.0%     
  BRITANNIA   52     FMCG       02-Apr-18   2,259.4     2,595.3     81     ₹27,207       +14.9%      -2.7%     
  TCS         1      IT         01-Aug-18   1,618.1     1,846.5     131    ₹29,926       +14.1%      +6.3%     
  PAGEIND     11     MFG        01-Aug-18   27,338.4    30,462.0    7      ₹21,865       +11.4%      +0.7%     
  TECHM       13     IT         01-Jun-18   516.6       572.3       390    ₹21,731       +10.8%      +1.7%     
  COFORGE     14     IT         01-Jun-18   201.6       214.9       1001   ₹13,369       +6.6%       -6.0%     
  SANOFI      6      HEALTH     01-Aug-18   4,304.8     4,484.0     49     ₹8,778        +4.2%       -2.5%     
  ASTRAL      79     MFG        02-Apr-18   410.2       424.7       449    ₹6,499        +3.5%       -8.7%     
  RELIANCE    4      OIL&GAS    01-Aug-18   527.8       545.2       404    ₹7,037        +3.3%       -0.5%     
  RHIM        12     METAL      01-Aug-18   219.2       222.1       972    ₹2,815        +1.3%       -2.9%     
  JUBLFOOD    38     CONSUMP    01-Jun-18   245.3       243.8       822    ₹-1,211       -0.6%       -7.2%     
  PIDILITIND  50     MFG        01-Jun-18   538.3       503.3       374    ₹-13,069      -6.5%       -5.4%     
  GLAXO       49     HEALTH     01-Aug-18   1,303.6     1,187.5     163    ₹-18,920      -8.9%       -8.9%     
  RELAXO      45     CON DUR    01-Aug-18   398.8       346.4       534    ₹-28,001      -13.1%      -11.0%    
  BAJFINANCE  98     FIN SVC    01-Aug-18   264.0       214.1       807    ₹-40,244      -18.9%      -10.5%    
  ⚠  WAZ < 0 (momentum below universe mean): HDFC, HDFCBANK

  AFTER: Invested ₹4,012,280 | Cash ₹110,872 | Total ₹4,123,152 | Positions 19/20 | Slot ₹206,158

========================================================================
  REBALANCE #24  —  03 Dec 2018
  NAV: ₹4,115,659  |  Slot: ₹205,783  |  Cash: ₹110,872
========================================================================
  [SECTOR CAP≤4] dropped: VINATIORGA

  [REGIME OFF] Nifty 500 9,126.9 < EMA200 9,151.6 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  4      FMCG       01-Feb-18   1,195.6     1,613.0     162    ₹67,607       +34.9%      +7.1%     
  HDFC        31     PVT BNK    01-Jun-17   371.3       488.1       421    ₹49,179       +31.5%      +4.0%     
  HDFCBANK    32     PVT BNK    01-Jun-17   371.3       488.1       421    ₹49,179       +31.5%      +4.0%     
  BRITANNIA   60     FMCG       02-Apr-18   2,259.4     2,748.7     81     ₹39,635       +21.7%      +4.2%     
  MPHASIS     86     IT         01-Feb-18   722.3       834.0       268    ₹29,948       +15.5%      +3.6%     
  ASTRAL      55     MFG        02-Apr-18   410.2       462.6       449    ₹23,535       +12.8%      +1.6%     
  RHIM        28     METAL      01-Aug-18   219.2       241.3       972    ₹21,447       +10.1%      +6.5%     
  DABUR       103    FMCG       01-Jun-18   354.9       386.6       568    ₹17,956       +8.9%       +2.7%     
  JUBLFOOD    48     CONSUMP    01-Jun-18   245.3       261.4       822    ₹13,197       +6.5%       +10.2%    
  SANOFI      9      HEALTH     01-Aug-18   4,304.8     4,503.1     49     ₹9,716        +4.6%       +3.2%     
  TECHM       20     IT         01-Jun-18   516.6       537.0       390    ₹7,963        +4.0%       +1.4%     
  PIDILITIND  19     MFG        01-Jun-18   538.3       556.8       374    ₹6,928        +3.4%       +3.7%     
  COFORGE     44     IT         01-Jun-18   201.6       203.2       1001   ₹1,610        +0.8%       -1.8%     
  TCS         14     IT         01-Aug-18   1,618.1     1,626.3     131    ₹1,081        +0.5%       +3.4%     
  RELIANCE    91     OIL&GAS    01-Aug-18   527.8       511.9       404    ₹-6,420       -3.0%       +2.7%     
  RELAXO      81     CON DUR    01-Aug-18   398.8       369.0       534    ₹-15,923      -7.5%       -1.7%     
  BAJFINANCE  66     FIN SVC    01-Aug-18   264.0       243.3       807    ₹-16,757      -7.9%       +4.3%     
  PAGEIND     161    MFG        01-Aug-18   27,338.4    24,355.1    7      ₹-20,883      -10.9% ⚠    -3.0%     
  GLAXO       165    HEALTH     01-Aug-18   1,303.6     1,130.5     163    ₹-28,214      -13.3% ⚠    +1.2%     
  ⚠  WAZ < 0 (momentum below universe mean): PAGEIND, GLAXO

  AFTER: Invested ₹4,004,787 | Cash ₹110,872 | Total ₹4,115,659 | Positions 19/20 | Slot ₹205,783

========================================================================
  REBALANCE #25  —  01 Feb 2019
  NAV: ₹4,185,846  |  Slot: ₹209,292  |  Cash: ₹110,872
========================================================================
  [SECTOR CAP≤4] dropped: MARICO

  [REGIME OFF] Nifty 500 9,056.3 < EMA200 9,121.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  8      FMCG       01-Feb-18   1,195.6     1,589.7     162    ₹63,842       +33.0%      +1.8%     
  HDFC        29     PVT BNK    01-Jun-17   371.3       482.9       421    ₹46,957       +30.0%      -0.3%     
  HDFCBANK    30     PVT BNK    01-Jun-17   371.3       482.9       421    ₹46,957       +30.0%      -0.3%     
  BRITANNIA   2      FMCG       02-Apr-18   2,259.4     2,887.7     81     ₹50,894       +27.8%      +2.4%     
  COFORGE     22     IT         01-Jun-18   201.6       243.0       1001   ₹41,440       +20.5%      +5.6%     
  ASTRAL      27     MFG        02-Apr-18   410.2       492.5       449    ₹36,961       +20.1%      +0.5%     
  DABUR       7      FMCG       01-Jun-18   354.9       423.0       568    ₹38,639       +19.2%      +5.0%     
  MPHASIS     56     IT         01-Feb-18   722.3       843.5       268    ₹32,477       +16.8%      +6.9%     
  SANOFI      3      HEALTH     01-Aug-18   4,304.8     4,707.5     49     ₹19,729       +9.4%       +2.8%     
  TECHM       46     IT         01-Jun-18   516.6       562.6       390    ₹17,926       +8.9%       +3.9%     
  JUBLFOOD    17     CONSUMP    01-Jun-18   245.3       266.9       822    ₹17,720       +8.8%       +10.6%    
  RELIANCE    11     OIL&GAS    01-Aug-18   527.8       553.3       404    ₹10,300       +4.8%       +5.0%     
  TCS         31     IT         01-Aug-18   1,618.1     1,668.9     131    ₹6,658        +3.1%       +4.9%     
  PIDILITIND  14     MFG        01-Jun-18   538.3       542.8       374    ₹1,685        +0.8%       -0.2%     
  BAJFINANCE  16     FIN SVC    01-Aug-18   264.0       254.9       807    ₹-7,387       -3.5%       +2.2%     
  RELAXO      73     CON DUR    01-Aug-18   398.8       366.1       534    ₹-17,479      -8.2%       +1.3%     
  GLAXO       68     HEALTH     01-Aug-18   1,303.6     1,152.2     163    ₹-24,671      -11.6%      -1.2%     
  RHIM        93     METAL      01-Aug-18   219.2       193.5       972    ₹-24,971      -11.7%      -5.1%     
  PAGEIND     157    MFG        01-Aug-18   27,338.4    22,094.4    7      ₹-36,708      -19.2% ⚠    +3.1%     
  ⚠  WAZ < 0 (momentum below universe mean): PAGEIND

  AFTER: Invested ₹4,074,974 | Cash ₹110,872 | Total ₹4,185,846 | Positions 19/20 | Slot ₹209,292

========================================================================
  REBALANCE #26  —  01 Apr 2019
  NAV: ₹4,318,576  |  Slot: ₹215,929  |  Cash: ₹110,872
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HINDUNILVR  109    FMCG       01-Feb-18   1,195.6     1,493.2     162    ₹48,198       +24.9%    424d  
  BRITANNIA   87     FMCG       02-Apr-18   2,259.4     2,709.0     81     ₹36,422       +19.9%    364d  
  RELIANCE    4      OIL&GAS    01-Aug-18   527.8       616.1       404    ₹35,676       +16.7%    243d  
  MPHASIS     119    IT         01-Feb-18   722.3       824.3       268    ₹27,328       +14.1%    424d  
  BAJFINANCE  8      FIN SVC    01-Aug-18   264.0       291.0       807    ₹21,804       +10.2%    243d  
  DABUR       134    FMCG       01-Jun-18   354.9       375.3       568    ₹11,575       +5.7%     304d  
  RHIM        63     METAL      01-Aug-18   219.2       231.0       972    ₹11,414       +5.4%     243d  
  SANOFI      166    HEALTH     01-Aug-18   4,304.8     4,202.2     49     ₹-5,028       -2.4%     243d  
  PAGEIND     82     MFG        01-Aug-18   27,338.4    23,593.7    7      ₹-26,213      -13.7%    243d  
  GLAXO       214    HEALTH     01-Aug-18   1,303.6     1,063.8     163    ₹-39,075      -18.4%    243d  

  ENTRIES (10)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BATAINDIA   1      CON DUR    3.724    1.07   +97.2%    +22.8%    1,295.6     166    ₹215,078      +3.3%     
  ASTRAZEN    2      HEALTH     3.444    0.82   +112.3%   +33.7%    1,979.9     109    ₹215,806      +3.2%     
  IPCALAB     3      HEALTH     2.791    0.53   +46.5%    +24.1%    474.3       455    ₹215,788      +7.8%     
  TITAN       4      CON DUR    2.639    0.91   +29.0%    +26.2%    1,094.0     197    ₹215,517      +2.9%     
  VBL         5      FMCG       2.573    0.39   +46.0%    +21.6%    52.0        4153   ₹215,882      +9.5%     
  VINATIORGA  6      MFG        2.495    1.13   +112.1%   +4.0%     814.1       265    ₹215,729      +2.9%     
  INFY        7      IT         2.451    0.46   +34.7%    +15.6%    618.5       349    ₹215,842      +3.3%     
  TIINDIA     10     AUTO       2.363    0.85   +59.9%    +17.7%    382.0       565    ₹215,811      +3.6%     
  GREAVESCOT  12     AUTO       2.288    0.92   +29.2%    +25.8%    141.0       1531   ₹215,840      +7.1%     
  ASIANPAINT  14     CONSUMP    2.178    0.94   +36.2%    +8.8%     1,397.3     154    ₹215,188      +3.0%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HDFC        15     PVT BNK    01-Jun-17   371.3       534.0       421    ₹68,482       +43.8%      +3.5%     
  HDFCBANK    16     PVT BNK    01-Jun-17   371.3       534.0       421    ₹68,482       +43.8%      +3.5%     
  ASTRAL      53     MFG        02-Apr-18   410.2       533.3       449    ₹55,261       +30.0%      +4.7%     
  COFORGE     39     IT         01-Jun-18   201.6       245.6       1001   ₹44,113       +21.9%      +1.1%     
  JUBLFOOD    36     CONSUMP    01-Jun-18   245.3       287.0       822    ₹34,253       +17.0%      +5.1%     
  TECHM       47     IT         01-Jun-18   516.6       591.9       390    ₹29,357       +14.6%      -0.5%     
  PIDILITIND  23     MFG        01-Jun-18   538.3       605.6       374    ₹25,162       +12.5%      +6.2%     
  TCS         25     IT         01-Aug-18   1,618.1     1,670.3     131    ₹6,841        +3.2%       +1.5%     
  RELAXO      19     CON DUR    01-Aug-18   398.8       398.1       534    ₹-355         -0.2%       +7.1%     

  AFTER: Invested ₹4,216,030 | Cash ₹99,985 | Total ₹4,316,015 | Positions 19/20 | Slot ₹215,929

========================================================================
  REBALANCE #27  —  03 Jun 2019
  NAV: ₹4,461,969  |  Slot: ₹223,098  |  Cash: ₹99,985
========================================================================
  [SECTOR CAP≤4] dropped: HEIDELBERG, PIIND, POLYPLEX

  EXITS (11)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ASTRAL      49     MFG        02-Apr-18   410.2       595.0       449    ₹82,960       +45.0%    427d  
  COFORGE     135    IT         01-Jun-18   201.6       241.5       1001   ₹39,984       +19.8%    367d  
  PIDILITIND  —      MFG        01-Jun-18   538.3       626.4       374    ₹32,945       +16.4%    367d  
  TCS         52     IT         01-Aug-18   1,618.1     1,843.5     131    ₹29,528       +13.9%    306d  
  TECHM       202    IT         01-Jun-18   516.6       570.9       390    ₹21,158       +10.5%    367d  
  JUBLFOOD    151    CONSUMP    01-Jun-18   245.3       263.8       822    ₹15,224       +7.6%     367d  
  GREAVESCOT  —      AUTO       01-Apr-19   141.0       147.9       1531   ₹10,547       +4.9%     63d   
  RELAXO      62     CON DUR    01-Aug-18   398.8       416.0       534    ₹9,175        +4.3%     306d  
  INFY        —      IT         01-Apr-19   618.5       609.9       349    ₹-2,987       -1.4%     63d   
  ASIANPAINT  125    CONSUMP    01-Apr-19   1,397.3     1,366.0     154    ₹-4,829       -2.2%     63d   
  ASTRAZEN    72     HEALTH     01-Apr-19   1,979.9     1,914.8     109    ₹-7,088       -3.3%     63d   

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  PGHL        1      HEALTH     3.263    1.10   +125.3%   +52.6%    3,295.3     67     ₹220,784      +5.4%     
  ICICIGI     3      FIN SVC    2.999    0.61   +67.9%    +33.3%    1,164.0     191    ₹222,315      +8.1%     
  ESABINDIA   4      MFG        2.969    1.07   +102.7%   +37.9%    997.7       223    ₹222,487      +8.6%     
  NAUKRI      5      IT         2.931    0.57   +97.5%    +39.0%    451.4       494    ₹222,985      +15.9%    
  HONAUT      6      MFG        2.790    0.63   +44.5%    +22.9%    26,282.6    8      ₹210,261      +7.2%     
  GUJGASLTD   7      OIL&GAS    2.728    0.87   +10.2%    +55.1%    177.2       1259   ₹223,068      +9.8%     
  ATUL        8      MFG        2.604    0.47   +42.0%    +19.8%    3,884.6     57     ₹221,421      +5.1%     
  PNCINFRA    10     INFRA      2.530    1.04   +21.8%    +50.1%    196.5       1135   ₹222,976      +13.6%    
  AXISBANK    14     PVT BNK    2.380    1.15   +55.3%    +15.5%    808.3       276    ₹223,082      +4.0%     
  BALRAMCHIN  17     FMCG       2.307    1.12   +117.5%   +19.7%    145.9       1529   ₹223,078      +4.1%     
  LT          19     INFRA      2.184    1.04   +19.6%    +22.2%    1,387.6     160    ₹222,009      +5.7%     

  HOLDS (8)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HDFC        15     PVT BNK    01-Jun-17   371.3       567.6       421    ₹82,612       +52.8%      +3.3%     
  HDFCBANK    16     PVT BNK    01-Jun-17   371.3       567.6       421    ₹82,612       +52.8%      +3.3%     
  VINATIORGA  2      MFG        01-Apr-19   814.1       1,047.3     265    ₹61,810       +28.7%      +11.1%    
  TITAN       25     CON DUR    01-Apr-19   1,094.0     1,236.5     197    ₹28,066       +13.0%      +5.1%     
  VBL         43     FMCG       01-Apr-19   52.0        54.8        4153   ₹11,702       +5.4%       +3.0%     
  TIINDIA     31     AUTO       01-Apr-19   382.0       379.4       565    ₹-1,470       -0.7%       +0.5%     
  BATAINDIA   21     CON DUR    01-Apr-19   1,295.6     1,272.7     166    ₹-3,806       -1.8%       +0.7%     
  IPCALAB     42     HEALTH     01-Apr-19   474.3       451.0       455    ₹-10,593      -4.9%       -0.9%     

  AFTER: Invested ₹4,291,868 | Cash ₹167,211 | Total ₹4,459,079 | Positions 19/20 | Slot ₹223,098

========================================================================
  REBALANCE #28  —  01 Aug 2019
  NAV: ₹4,149,556  |  Slot: ₹207,478  |  Cash: ₹167,211
========================================================================
  [SECTOR CAP≤4] dropped: PIIND

  [REGIME OFF] Nifty 500 8,935.7 < EMA200 9,362.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HDFC        137    PVT BNK    01-Jun-17   371.3       517.5       421    ₹61,557       +39.4% ⚠    -4.1%     
  HDFCBANK    138    PVT BNK    01-Jun-17   371.3       517.5       421    ₹61,557       +39.4% ⚠    -4.1%     
  PGHL        1      HEALTH     03-Jun-19   3,295.3     3,897.3     67     ₹40,334       +18.3%      +4.1%     
  VINATIORGA  15     MFG        01-Apr-19   814.1       883.0       265    ₹18,262       +8.5%       -7.9%     
  VBL         33     FMCG       01-Apr-19   52.0        54.2        4153   ₹9,270        +4.3%       -0.8%     
  IPCALAB     38     HEALTH     01-Apr-19   474.3       472.3       455    ₹-910         -0.4%       +2.2%     
  NAUKRI      5      IT         03-Jun-19   451.4       433.1       494    ₹-9,032       -4.1%       +0.1%     
  TITAN       91     CON DUR    01-Apr-19   1,094.0     1,037.1     197    ₹-11,205      -5.2%       -5.1%     
  ICICIGI     16     FIN SVC    03-Jun-19   1,164.0     1,101.5     191    ₹-11,928      -5.4%       +2.9%     
  BATAINDIA   27     CON DUR    01-Apr-19   1,295.6     1,221.7     166    ₹-12,284      -5.7%       -2.1%     
  BALRAMCHIN  6      FMCG       03-Jun-19   145.9       135.0       1529   ₹-16,681      -7.5%       -3.4%     
  PNCINFRA    10     INFRA      03-Jun-19   196.5       181.3       1135   ₹-17,169      -7.7%       -4.0%     
  TIINDIA     23     AUTO       01-Apr-19   382.0       351.7       565    ₹-17,090      -7.9%       -3.9%     
  GUJGASLTD   47     OIL&GAS    03-Jun-19   177.2       161.8       1259   ₹-19,340      -8.7%       +2.8%     
  ESABINDIA   3      MFG        03-Jun-19   997.7       909.4       223    ₹-19,692      -8.9%       -2.8%     
  ATUL        18     MFG        03-Jun-19   3,884.6     3,533.6     57     ₹-20,005      -9.0%       -2.9%     
  LT          74     INFRA      03-Jun-19   1,387.6     1,224.0     160    ₹-26,165      -11.8%      -4.1%     
  HONAUT      67     MFG        03-Jun-19   26,282.6    22,664.6    8      ₹-28,944      -13.8%      -0.7%     
  AXISBANK    96     PVT BNK    03-Jun-19   808.3       666.5       276    ₹-39,123      -17.5%      -8.5%     
  ⚠  WAZ < 0 (momentum below universe mean): HDFC, HDFCBANK

  AFTER: Invested ₹3,982,345 | Cash ₹167,211 | Total ₹4,149,556 | Positions 19/20 | Slot ₹207,478

========================================================================
  REBALANCE #29  —  01 Oct 2019
  NAV: ₹4,482,053  |  Slot: ₹224,103  |  Cash: ₹167,211
========================================================================
  [SECTOR CAP≤4] dropped: HINDUNILVR

  [REGIME OFF] Nifty 500 9,236.4 < EMA200 9,253.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HDFC        35     PVT BNK    01-Jun-17   371.3       581.8       421    ₹88,604       +56.7%      +5.4%     
  HDFCBANK    36     PVT BNK    01-Jun-17   371.3       581.8       421    ₹88,604       +56.7%      +5.4%     
  VINATIORGA  20     MFG        01-Apr-19   814.1       1,070.1     265    ₹67,850       +31.5%      -1.3%     
  BATAINDIA   3      CON DUR    01-Apr-19   1,295.6     1,630.0     166    ₹55,504       +25.8%      +6.5%     
  TITAN       31     CON DUR    01-Apr-19   1,094.0     1,256.2     197    ₹31,945       +14.8%      +6.3%     
  PGHL        18     HEALTH     03-Jun-19   3,295.3     3,733.5     67     ₹29,361       +13.3%      +1.2%     
  ESABINDIA   43     MFG        03-Jun-19   997.7       1,074.2     223    ₹17,069       +7.7%       +9.4%     
  HONAUT      21     MFG        03-Jun-19   26,282.6    27,751.2    8      ₹11,749       +5.6%       +4.4%     
  BALRAMCHIN  24     FMCG       03-Jun-19   145.9       149.5       1529   ₹5,513        +2.5%       +4.2%     
  VBL         110    FMCG       01-Apr-19   52.0        52.7        4153   ₹2,810        +1.3%       -3.6%     
  ATUL        50     MFG        03-Jun-19   3,884.6     3,886.1     57     ₹87           +0.0%       +4.3%     
  ICICIGI     38     FIN SVC    03-Jun-19   1,164.0     1,151.0     191    ₹-2,472       -1.1%       +1.4%     
  TIINDIA     67     AUTO       01-Apr-19   382.0       374.7       565    ₹-4,083       -1.9%       +4.6%     
  LT          114    INFRA      03-Jun-19   1,387.6     1,319.6     160    ₹-10,879      -4.9%       +3.8%     
  PNCINFRA    116    INFRA      03-Jun-19   196.5       184.1       1135   ₹-14,071      -6.3%       +0.5%     
  GUJGASLTD   83     OIL&GAS    03-Jun-19   177.2       163.7       1259   ₹-16,982      -7.6%       -1.8%     
  IPCALAB     88     HEALTH     01-Apr-19   474.3       437.7       455    ₹-16,655      -7.7%       -4.1%     
  NAUKRI      97     IT         03-Jun-19   451.4       403.6       494    ₹-23,629      -10.6%      +1.3%     
  AXISBANK    174    PVT BNK    03-Jun-19   808.3       676.3       276    ₹-36,416      -16.3%      -0.2%     

  AFTER: Invested ₹4,314,843 | Cash ₹167,211 | Total ₹4,482,053 | Positions 19/20 | Slot ₹224,103

========================================================================
  REBALANCE #30  —  02 Dec 2019
  NAV: ₹4,708,520  |  Slot: ₹235,426  |  Cash: ₹167,211
========================================================================
  [SECTOR CAP≤4] dropped: ASTRAZEN

  EXITS (13)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HDFC        46     PVT BNK    01-Jun-17   371.3       589.7       421    ₹91,929       +58.8%    914d  
  HDFCBANK    47     PVT BNK    01-Jun-17   371.3       589.7       421    ₹91,929       +58.8%    914d  
  VBL         73     FMCG       01-Apr-19   52.0        63.9        4153   ₹49,601       +23.0%    245d  
  VINATIORGA  177    MFG        01-Apr-19   814.1       964.8       265    ₹39,952       +18.5%    245d  
  ESABINDIA   52     MFG        03-Jun-19   997.7       1,110.2     223    ₹25,084       +11.3%    182d  
  PGHL        91     HEALTH     03-Jun-19   3,295.3     3,564.0     67     ₹18,006       +8.2%     182d  
  TITAN       130    CON DUR    01-Apr-19   1,094.0     1,131.8     197    ₹7,442        +3.5%     245d  
  HONAUT      67     MFG        03-Jun-19   26,282.6    26,924.4    8      ₹5,134        +2.4%     182d  
  ATUL        68     MFG        03-Jun-19   3,884.6     3,949.5     57     ₹3,698        +1.7%     182d  
  BALRAMCHIN  84     FMCG       03-Jun-19   145.9       148.0       1529   ₹3,209        +1.4%     182d  
  PNCINFRA    105    INFRA      03-Jun-19   196.5       193.3       1135   ₹-3,584       -1.6%     182d  
  AXISBANK    101    PVT BNK    03-Jun-19   808.3       741.2       276    ₹-18,509      -8.3%     182d  
  LT          229    INFRA      03-Jun-19   1,387.6     1,201.8     160    ₹-29,727      -13.4%    182d  

  ENTRIES (13)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIGREEN  1      ENERGY     5.987    0.18   +210.0%   +192.1%   131.3       1793   ₹235,421      +21.4%    
  HDFCAMC     2      FIN SVC    3.967    0.47   +146.4%   +41.1%    1,498.3     157    ₹235,233      -0.9%     
  CREDITACC   3      FIN SVC    3.642    0.13   +125.3%   +55.7%    809.9       290    ₹234,879      +8.6%     
  GLAXO       4      HEALTH     3.543    0.27   +25.7%    +39.2%    1,389.4     169    ₹234,801      +2.2%     
  ABBOTINDIA  5      HEALTH     3.502    0.24   +66.6%    +38.0%    11,580.2    20     ₹231,603      +2.8%     
  AAVAS       6      FIN SVC    3.351    0.20   +159.5%   +13.5%    1,751.9     134    ₹234,755      +3.6%     
  PIIND       7      MFG        3.263    0.05   +80.8%    +34.6%    1,478.3     159    ₹235,051      +5.3%     
  LALPATHLAB  8      HEALTH     2.976    0.13   +93.3%    +36.6%    760.0       309    ₹234,829      +2.5%     
  EPL         9      MFG        2.905    0.16   +77.3%    +66.9%    133.7       1760   ₹235,390      +8.8%     
  BERGEPAINT  10     CON DUR    2.867    0.33   +56.8%    +34.4%    400.4       587    ₹235,031      +1.3%     
  WHIRLPOOL   11     CON DUR    2.865    0.33   +52.3%    +40.2%    2,121.8     110    ₹233,401      -0.8%     
  RELAXO      12     CON DUR    2.836    0.16   +47.8%    +31.3%    575.8       408    ₹234,942      +3.8%     
  IGL         15     OIL&GAS    2.531    0.14   +56.4%    +26.9%    185.5       1269   ₹235,372      +2.4%     

  HOLDS (6)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TIINDIA     13     AUTO       01-Apr-19   382.0       481.0       565    ₹55,978       +25.9%      +9.6%     
  BATAINDIA   34     CON DUR    01-Apr-19   1,295.6     1,526.7     166    ₹38,353       +17.8%      -2.5%     
  IPCALAB     38     HEALTH     01-Apr-19   474.3       553.3       455    ₹35,952       +16.7%      +3.5%     
  GUJGASLTD   21     OIL&GAS    03-Jun-19   177.2       206.7       1259   ₹37,106       +16.6%      +8.8%     
  ICICIGI     39     FIN SVC    03-Jun-19   1,164.0     1,314.4     191    ₹28,739       +12.9%      +2.1%     
  NAUKRI      30     IT         03-Jun-19   451.4       492.0       494    ₹20,085       +9.0%       -1.4%     

  AFTER: Invested ₹4,581,965 | Cash ₹122,932 | Total ₹4,704,898 | Positions 19/20 | Slot ₹235,426

========================================================================
  REBALANCE #31  —  03 Feb 2020
  NAV: ₹5,191,340  |  Slot: ₹259,567  |  Cash: ₹122,932
========================================================================
  [SECTOR CAP≤4] dropped: ALKYLAMINE, SRF, FINEORG

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BATAINDIA   28     CON DUR    01-Apr-19   1,295.6     1,736.3     166    ₹73,146       +34.0%    308d  
  TIINDIA     60     AUTO       01-Apr-19   382.0       480.1       565    ₹55,450       +25.7%    308d  
  NAUKRI      43     IT         03-Jun-19   451.4       550.2       494    ₹48,794       +21.9%    245d  
  WHIRLPOOL   42     CON DUR    02-Dec-19   2,121.8     2,361.3     110    ₹26,344       +11.3%    63d   
  ICICIGI     85     FIN SVC    03-Jun-19   1,164.0     1,247.4     191    ₹15,943       +7.2%     245d  
  ABBOTINDIA  36     HEALTH     02-Dec-19   11,580.2    11,694.8    20     ₹2,294        +1.0%     63d   
  GLAXO       111    HEALTH     02-Dec-19   1,389.4     1,371.2     169    ₹-3,068       -1.3%     63d   
  CREDITACC   32     FIN SVC    02-Dec-19   809.9       754.6       290    ₹-16,055      -6.8%     63d   
  HDFCAMC     27     FIN SVC    02-Dec-19   1,498.3     1,354.2     157    ₹-22,631      -9.6%     63d   

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  GMMPFAUDLR  2      MFG        4.843    0.06   +147.3%   +99.3%    923.4       281    ₹259,470      +21.3%    
  DIXON       3      CON DUR    3.704    0.08   +118.0%   +57.4%    946.1       274    ₹259,223      +11.2%    
  AMBER       7      CON DUR    3.345    0.69   +74.6%    +52.6%    1,532.1     169    ₹258,933      +12.6%    
  JBCHEPHARM  8      HEALTH     3.050    -0.07  +61.5%    +34.5%    228.6       1135   ₹259,482      +3.2%     
  COROMANDEL  9      MFG        3.035    0.11   +45.3%    +35.1%    585.9       442    ₹258,985      +6.6%     
  NESCO       12     CONSUMP    2.926    0.33   +63.8%    +31.3%    715.7       362    ₹259,082      +2.6%     
  NH          13     HEALTH     2.917    0.23   +93.7%    +30.8%    356.2       728    ₹259,339      +1.9%     
  RATNAMANI   16     METAL      2.736    0.23   +37.6%    +25.4%    798.1       325    ₹259,372      +4.3%     
  JUBLFOOD    18     CONSUMP    2.419    0.36   +66.3%    +25.1%    385.8       672    ₹259,258      +9.9%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GUJGASLTD   6      OIL&GAS    03-Jun-19   177.2       269.8       1259   ₹116,587      +52.3%      +2.3%     
  ADANIGREEN  1      ENERGY     02-Dec-19   131.3       196.1       1793   ₹116,276      +49.4%      +4.1%     
  IPCALAB     23     HEALTH     01-Apr-19   474.3       583.3       455    ₹49,604       +23.0%      -1.9%     
  RELAXO      4      CON DUR    02-Dec-19   575.8       706.9       408    ₹53,472       +22.8%      +5.2%     
  IGL         10     OIL&GAS    02-Dec-19   185.5       227.3       1269   ₹53,022       +22.5%      +6.7%     
  EPL         20     MFG        02-Dec-19   133.7       155.1       1760   ₹37,554       +16.0%      -0.2%     
  BERGEPAINT  17     CON DUR    02-Dec-19   400.4       461.5       587    ₹35,863       +15.3%      +4.1%     
  AAVAS       5      FIN SVC    02-Dec-19   1,751.9     1,988.8     134    ₹31,745       +13.5%      +1.3%     
  LALPATHLAB  26     HEALTH     02-Dec-19   760.0       830.1       309    ₹21,676       +9.2%       +4.0%     
  PIIND       19     MFG        02-Dec-19   1,478.3     1,520.1     159    ₹6,639        +2.8%       +3.5%     

  AFTER: Invested ₹5,175,228 | Cash ₹13,342 | Total ₹5,188,570 | Positions 19/20 | Slot ₹259,567

========================================================================
  REBALANCE #32  —  01 Apr 2020
  NAV: ₹4,197,868  |  Slot: ₹209,893  |  Cash: ₹13,342
========================================================================
  [SECTOR CAP≤4] dropped: ALKEM, GRANULES

  [REGIME OFF] Nifty 500 6,761.9 < EMA200 9,209.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  IPCALAB     4      HEALTH     01-Apr-19   474.3       686.1       455    ₹96,405       +44.7%      +4.0%     
  GUJGASLTD   19     OIL&GAS    03-Jun-19   177.2       216.2       1259   ₹49,112       +22.0%      -6.3%     
  ADANIGREEN  1      ENERGY     02-Dec-19   131.3       152.9       1793   ₹38,729       +16.5%      +3.3%     
  JBCHEPHARM  5      HEALTH     03-Feb-20   228.6       238.8       1135   ₹11,563       +4.5%       -1.9%     
  EPL         31     MFG        02-Dec-19   133.7       135.8       1760   ₹3,562        +1.5%       -2.5%     
  RELAXO      11     CON DUR    02-Dec-19   575.8       574.3       408    ₹-618         -0.3%       -4.6%     
  BERGEPAINT  12     CON DUR    02-Dec-19   400.4       392.5       587    ₹-4,619       -2.0%       +0.5%     
  IGL         28     OIL&GAS    02-Dec-19   185.5       174.2       1269   ₹-14,373      -6.1%       +2.5%     
  GMMPFAUDLR  3      MFG        03-Feb-20   923.4       826.2       281    ₹-27,296      -10.5%      -0.4%     
  LALPATHLAB  26     HEALTH     02-Dec-19   760.0       664.5       309    ₹-29,489      -12.6%      -5.6%     
  COROMANDEL  22     MFG        03-Feb-20   585.9       497.7       442    ₹-39,013      -15.1%      -3.2%     
  AMBER       8      CON DUR    03-Feb-20   1,532.1     1,267.9     169    ₹-44,658      -17.2%      +1.2%     
  PIIND       52     MFG        02-Dec-19   1,478.3     1,175.0     159    ₹-48,224      -20.5%      -3.7%     
  DIXON       20     CON DUR    03-Feb-20   946.1       713.0       274    ₹-63,851      -24.6%      -1.5%     
  RATNAMANI   73     METAL      03-Feb-20   798.1       581.0       325    ₹-70,546      -27.2%      -13.7%    
  JUBLFOOD    65     CONSUMP    03-Feb-20   385.8       273.9       672    ₹-75,226      -29.0%      -5.2%     
  NH          61     HEALTH     03-Feb-20   356.2       247.5       728    ₹-79,189      -30.5%      -5.8%     
  AAVAS       109    FIN SVC    02-Dec-19   1,751.9     1,192.0     134    ₹-75,027      -32.0%      -12.4%    
  NESCO       93     CONSUMP    03-Feb-20   715.7       451.9       362    ₹-95,506      -36.9%      -12.4%    

  AFTER: Invested ₹4,184,527 | Cash ₹13,342 | Total ₹4,197,868 | Positions 19/20 | Slot ₹209,893

========================================================================
  REBALANCE #33  —  01 Jun 2020
  NAV: ₹5,054,422  |  Slot: ₹252,721  |  Cash: ₹13,342
========================================================================
  [SECTOR CAP≤4] dropped: ALKYLAMINE, CIPLA, DRREDDY, AARTIDRUGS, INDIACEM, APLLTD, DIVISLAB, SANOFI

  [REGIME OFF] Nifty 500 8,020.1 < EMA200 8,686.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  —      ENERGY     02-Dec-19   131.3       260.8       1793   ₹232,104      +98.6%      +13.0%    
  IPCALAB     17     HEALTH     01-Apr-19   474.3       749.6       455    ₹125,262      +58.0%      -2.1%     
  JBCHEPHARM  2      HEALTH     03-Feb-20   228.6       324.7       1135   ₹109,104      +42.0%      +8.5%     
  GMMPFAUDLR  1      MFG        03-Feb-20   923.4       1,272.7     281    ₹98,146       +37.8%      +6.6%     
  GUJGASLTD   61     OIL&GAS    03-Jun-19   177.2       233.9       1259   ₹71,444       +32.0%      +0.9%     
  RELAXO      27     CON DUR    02-Dec-19   575.8       698.2       408    ₹49,908       +21.2%      +11.4%    
  IGL         26     OIL&GAS    02-Dec-19   185.5       209.8       1269   ₹30,896       +13.1%      +2.0%     
  EPL         75     MFG        02-Dec-19   133.7       150.8       1760   ₹30,043       +12.8%      +0.0%     
  DIXON       9      CON DUR    03-Feb-20   946.1       1,008.1     274    ₹16,998       +6.6%       +13.3%    
  PIIND       33     MFG        02-Dec-19   1,478.3     1,552.3     159    ₹11,764       +5.0%       +4.1%     
  COROMANDEL  16     MFG        03-Feb-20   585.9       605.6       442    ₹8,699        +3.4%       +4.7%     
  BERGEPAINT  47     CON DUR    02-Dec-19   400.4       397.9       587    ₹-1,473       -0.6%       +4.6%     
  LALPATHLAB  52     HEALTH     02-Dec-19   760.0       728.6       309    ₹-9,689       -4.1%       -1.4%     
  AMBER       35     CON DUR    03-Feb-20   1,532.1     1,338.6     169    ₹-32,718      -12.6%      +13.6%    
  JUBLFOOD    —      CONSUMP    03-Feb-20   385.8       331.8       672    ₹-36,260      -14.0%      +4.1%     
  NH          54     HEALTH     03-Feb-20   356.2       292.8       728    ₹-46,195      -17.8%      +9.4%     
  RATNAMANI   190    METAL      03-Feb-20   798.1       576.2       325    ₹-72,118      -27.8% ⚠    +2.9%     
  AAVAS       215    FIN SVC    02-Dec-19   1,751.9     1,063.1     134    ₹-92,306      -39.3% ⚠    +0.1%     
  NESCO       194    CONSUMP    03-Feb-20   715.7       424.8       362    ₹-105,319     -40.7% ⚠    +4.0%     
  ⚠  WAZ < 0 (momentum below universe mean): RATNAMANI, NESCO, AAVAS

  AFTER: Invested ₹5,041,081 | Cash ₹13,342 | Total ₹5,054,422 | Positions 19/20 | Slot ₹252,721

========================================================================
  REBALANCE #34  —  03 Aug 2020
  NAV: ₹5,913,615  |  Slot: ₹295,681  |  Cash: ₹13,342
========================================================================
  [SECTOR CAP≤4] dropped: LAURUSLABS, GRANULES, JUBLPHARMA, POLYMED, SYNGENE

  EXITS (12)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIGREEN  —      ENERGY     02-Dec-19   131.3       339.8       1793   ₹373,840      +158.8%   245d  
  EPL         44     MFG        02-Dec-19   133.7       215.5       1760   ₹143,876      +61.1%    245d  
  PIIND       51     MFG        02-Dec-19   1,478.3     1,809.4     159    ₹52,642       +22.4%    245d  
  LALPATHLAB  50     HEALTH     02-Dec-19   760.0       892.7       309    ₹41,003       +17.5%    245d  
  BERGEPAINT  60     CON DUR    02-Dec-19   400.4       426.3       587    ₹15,232       +6.5%     245d  
  RELAXO      136    CON DUR    02-Dec-19   575.8       584.3       408    ₹3,472        +1.5%     245d  
  IGL         216    OIL&GAS    02-Dec-19   185.5       175.4       1269   ₹-12,760      -5.4%     245d  
  RATNAMANI   66     METAL      03-Feb-20   798.1       731.2       325    ₹-21,726      -8.4%     182d  
  JUBLFOOD    —      CONSUMP    03-Feb-20   385.8       344.3       672    ₹-27,918      -10.8%    182d  
  NH          119    HEALTH     03-Feb-20   356.2       292.2       728    ₹-46,591      -18.0%    182d  
  AAVAS       147    FIN SVC    02-Dec-19   1,751.9     1,407.2     134    ₹-46,183      -19.7%    245d  
  NESCO       222    CONSUMP    03-Feb-20   715.7       422.1       362    ₹-106,286     -41.0%    182d  

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  AARTIDRUGS  1      HEALTH     5.191    0.68   +261.3%   +146.9%   429.0       689    ₹295,575      +14.0%    
  IOLCP       2      HEALTH     4.198    0.45   +277.1%   +145.9%   140.5       2105   ₹295,675      +9.1%     
  PERSISTENT  5      IT         3.745    0.45   +71.8%    +108.7%   466.7       633    ₹295,426      +21.7%    
  MASTEK      8      IT         3.124    0.71   +62.6%    +168.0%   613.2       482    ₹295,585      +35.5%    
  TATACOMM    10     CONSUMP    2.954    0.41   +170.8%   +78.0%    733.8       402    ₹294,975      +16.5%    
  DHANUKA     11     FMCG       2.814    0.74   +119.2%   +84.9%    781.7       378    ₹295,474      +1.9%     
  EIDPARRY    12     FMCG       2.477    0.96   +81.5%    +102.6%   282.5       1046   ₹295,481      +5.5%     
  HAL         13     DEFENCE    2.383    0.38   +40.8%    +77.5%    399.6       739    ₹295,315      +1.5%     
  ALKYLAMINE  17     MFG        2.337    0.95   +201.8%   +28.3%    893.0       331    ₹295,598      +1.4%     
  NAVINFLUOR  18     MFG        2.313    0.73   +203.3%   +15.0%    1,726.7     171    ₹295,267      +1.1%     
  MUTHOOTFIN  19     FIN SVC    2.252    1.17   +115.6%   +64.0%    1,172.9     252    ₹295,572      +4.0%     

  HOLDS (7)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  IPCALAB     30     HEALTH     01-Apr-19   474.3       927.0       455    ₹205,982      +95.5%      +7.8%     
  DIXON       4      CON DUR    03-Feb-20   946.1       1,590.2     274    ₹176,496      +68.1%      +14.6%    
  GUJGASLTD   35     OIL&GAS    03-Jun-19   177.2       291.7       1259   ₹144,202      +64.6%      +3.6%     
  JBCHEPHARM  25     HEALTH     03-Feb-20   228.6       344.7       1135   ₹131,699      +50.8%      +1.5%     
  GMMPFAUDLR  22     MFG        03-Feb-20   923.4       1,355.4     281    ₹121,394      +46.8%      -0.2%     
  COROMANDEL  15     MFG        03-Feb-20   585.9       737.5       442    ₹66,990       +25.9%      +1.3%     
  AMBER       16     CON DUR    03-Feb-20   1,532.1     1,722.2     169    ₹32,118       +12.4%      +14.4%    

  AFTER: Invested ₹5,863,774 | Cash ₹45,982 | Total ₹5,909,756 | Positions 18/20 | Slot ₹295,681

========================================================================
  REBALANCE #35  —  01 Oct 2020
  NAV: ₹6,770,777  |  Slot: ₹338,539  |  Cash: ₹45,982
========================================================================
  [SECTOR CAP≤4] dropped: LAURUSLABS, GRANULES, ADVENZYMES, SOLARA

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  GUJGASLTD   76     OIL&GAS    03-Jun-19   177.2       297.7       1259   ₹151,765      +68.0%    486d  
  GMMPFAUDLR  65     MFG        03-Feb-20   923.4       1,270.6     281    ₹97,575       +37.6%    241d  
  COROMANDEL  45     MFG        03-Feb-20   585.9       749.7       442    ₹72,361       +27.9%    241d  
  EIDPARRY    98     FMCG       03-Aug-20   282.5       266.5       1046   ₹-16,772      -5.7%     59d   
  DHANUKA     47     FMCG       03-Aug-20   781.7       733.9       378    ₹-18,072      -6.1%     59d   
  MUTHOOTFIN  96     FIN SVC    03-Aug-20   1,172.9     1,055.2     252    ₹-29,652      -10.0%    59d   
  HAL         189    DEFENCE    03-Aug-20   399.6       357.3       739    ₹-31,295      -10.6%    59d   

  ENTRIES (6)
  [52w filter blocked 1: EPL(-21.2%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  CGPOWER     8      ENERGY     2.937    0.43   +57.4%    +139.5%   24.3        13959  ₹338,527      +8.2%     
  AFFLE       12     IT         2.648    0.56   +143.5%   +78.7%    572.0       591    ₹338,064      +0.5%     
  ADANIENT    13     METAL      2.642    1.18   +103.2%   +87.7%    307.3       1101   ₹338,300      +8.0%     
  DEEPAKNTR   16     MFG        2.582    1.18   +164.4%   +58.8%    805.5       420    ₹338,305      +2.8%     
  APLAPOLLO   17     METAL      2.582    1.13   +117.6%   +67.7%    291.4       1161   ₹338,327      +11.1%    
  VAIBHAVGBL  20     CON DUR    2.298    0.91   +133.3%   +47.9%    344.5       982    ₹338,307      +5.0%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  IPCALAB     15     HEALTH     01-Apr-19   474.3       1,106.3     455    ₹287,584      +133.3%     +7.1%     
  JBCHEPHARM  9      HEALTH     03-Feb-20   228.6       468.5       1135   ₹272,298      +104.9%     +6.0%     
  DIXON       7      CON DUR    03-Feb-20   946.1       1,766.2     274    ₹224,714      +86.7%      -0.7%     
  AARTIDRUGS  1      HEALTH     03-Aug-20   429.0       795.5       689    ₹252,499      +85.4%      +10.9%    
  ALKYLAMINE  5      MFG        03-Aug-20   893.0       1,267.0     331    ₹123,765      +41.9%      +2.1%     
  PERSISTENT  3      IT         03-Aug-20   466.7       623.6       633    ₹99,288       +33.6%      +10.5%    
  AMBER       24     CON DUR    03-Feb-20   1,532.1     2,043.8     169    ₹86,477       +33.4%      +4.4%     
  MASTEK      6      IT         03-Aug-20   613.2       817.2       482    ₹98,283       +33.3%      +4.6%     
  NAVINFLUOR  21     MFG        03-Aug-20   1,726.7     2,095.0     171    ₹62,978       +21.3%      +3.6%     
  TATACOMM    33     CONSUMP    03-Aug-20   733.8       774.6       402    ₹16,409       +5.6%       -1.0%     
  IOLCP       10     HEALTH     03-Aug-20   140.5       135.6       2105   ₹-10,303      -3.5%       -4.1%     

  AFTER: Invested ₹6,605,351 | Cash ₹163,016 | Total ₹6,768,367 | Positions 17/20 | Slot ₹338,539

========================================================================
  REBALANCE #36  —  01 Dec 2020
  NAV: ₹7,815,692  |  Slot: ₹390,785  |  Cash: ₹163,016
========================================================================
  [SECTOR CAP≤4] dropped: GRANULES, ADVENZYMES

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  IPCALAB     47     HEALTH     01-Apr-19   474.3       1,120.5     455    ₹294,023      +136.3%   610d  
  MASTEK      28     IT         03-Aug-20   613.2       912.0       482    ₹143,992      +48.7%    120d  
  TATACOMM    39     CONSUMP    03-Aug-20   733.8       959.9       402    ₹90,916       +30.8%    120d  
  PERSISTENT  32     IT         03-Aug-20   466.7       577.6       633    ₹70,193       +23.8%    120d  
  DEEPAKNTR   27     MFG        01-Oct-20   805.5       849.6       420    ₹18,517       +5.5%     61d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  LAURUSLABS  1      HEALTH     4.287    -0.09  +366.3%   +44.2%    317.1       1232   ₹390,728      +9.9%     
  ATGL        3      OIL&GAS    3.619    0.17   +142.2%   +103.7%   360.4       1084   ₹390,623      +22.2%    
  TTML        6      CONSUMP    3.093    0.23   +154.5%   +100.0%   7.0         55826  ₹390,782      +3.4%     
  RATNAMANI   10     METAL      2.720    0.13   +74.0%    +43.1%    1,095.9     356    ₹390,137      +11.3%    
  TATAELXSI   13     IT         2.550    0.04   +107.4%   +48.4%    1,527.0     255    ₹389,388      +5.9%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  DIXON       4      CON DUR    03-Feb-20   946.1       2,301.9     274    ₹371,491      +143.3%     +9.6%     
  JBCHEPHARM  11     HEALTH     03-Feb-20   228.6       475.4       1135   ₹280,078      +107.9%     +3.3%     
  CGPOWER     2      ENERGY     01-Oct-20   24.3        42.7        13959  ₹257,860      +76.2%      +20.4%    
  ALKYLAMINE  8      MFG        03-Aug-20   893.0       1,546.5     331    ₹216,281      +73.2%      +13.3%    
  AARTIDRUGS  5      HEALTH     03-Aug-20   429.0       695.2       689    ₹183,403      +62.0%      +2.0%     
  AMBER       22     CON DUR    03-Feb-20   1,532.1     2,439.1     169    ₹153,275      +59.2%      +8.4%     
  NAVINFLUOR  7      MFG        03-Aug-20   1,726.7     2,638.5     171    ₹155,920      +52.8%      +6.2%     
  ADANIENT    21     METAL      01-Oct-20   307.3       420.7       1101   ₹124,879      +36.9%      +10.8%    
  AFFLE       23     IT         01-Oct-20   572.0       717.9       591    ₹86,203       +25.5%      +16.0%    
  APLAPOLLO   17     METAL      01-Oct-20   291.4       340.0       1161   ₹56,400       +16.7%      +6.6%     
  VAIBHAVGBL  18     CON DUR    01-Oct-20   344.5       388.9       982    ₹43,585       +12.9%      +4.1%     
  IOLCP       14     HEALTH     03-Aug-20   140.5       147.3       2105   ₹14,301       +4.8%       +10.4%    

  AFTER: Invested ₹7,546,612 | Cash ₹266,763 | Total ₹7,813,375 | Positions 17/20 | Slot ₹390,785

========================================================================
  REBALANCE #37  —  01 Feb 2021
  NAV: ₹9,378,168  |  Slot: ₹468,908  |  Cash: ₹266,763
========================================================================
  [SECTOR CAP≤4] dropped: SOLARA

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  JBCHEPHARM  115    HEALTH     03-Feb-20   228.6       466.7       1135   ₹270,205      +104.1%   364d  
  AMBER       88     CON DUR    03-Feb-20   1,532.1     2,624.2     169    ₹184,565      +71.3%    364d  
  NAVINFLUOR  108    MFG        03-Aug-20   1,726.7     2,307.2     171    ₹99,264       +33.6%    182d  
  RATNAMANI   87     METAL      01-Dec-20   1,095.9     996.8       356    ₹-35,289      -9.0%     62d   

  ENTRIES (4)
  [52w filter blocked 1: SUZLON(-21.9%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TATACHEM    12     MFG        2.313    0.13   +55.0%    +53.3%    453.6       1033   ₹468,613      -1.5%     
  PGHL        13     HEALTH     2.294    0.15   +67.5%    +44.6%    6,114.1     76     ₹464,675      +2.5%     
  PERSISTENT  14     IT         2.250    0.17   +117.9%   +33.5%    739.7       633    ₹468,253      +0.5%     
  SFL         16     CON DUR    2.226    0.17   +33.5%    +50.3%    983.1       476    ₹467,968      +1.1%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  DIXON       3      CON DUR    03-Feb-20   946.1       3,029.4     274    ₹570,843      +220.2%     +6.0%     
  TTML        1      CONSUMP    01-Dec-20   7.0         18.0        55826  ₹614,086      +157.1%     +39.3%    
  ALKYLAMINE  4      MFG        03-Aug-20   893.0       1,934.4     331    ₹344,679      +116.6%     +5.9%     
  TATAELXSI   2      IT         01-Dec-20   1,527.0     2,765.8     255    ₹315,890      +81.1%      +21.6%    
  ADANIENT    11     METAL      01-Oct-20   307.3       535.7       1101   ₹251,461      +74.3%      +4.7%     
  CGPOWER     7      ENERGY     01-Oct-20   24.3        40.0        13959  ₹219,250      +64.8%      -2.1%     
  APLAPOLLO   15     METAL      01-Oct-20   291.4       459.6       1161   ₹195,300      +57.7%      +2.0%     
  AARTIDRUGS  10     HEALTH     03-Aug-20   429.0       668.4       689    ₹164,924      +55.8%      -3.5%     
  VAIBHAVGBL  8      CON DUR    01-Oct-20   344.5       508.2       982    ₹160,784      +47.5%      +8.2%     
  AFFLE       17     IT         01-Oct-20   572.0       763.0       591    ₹112,887      +33.4%      +1.7%     
  LAURUSLABS  6      HEALTH     01-Dec-20   317.1       343.9       1232   ₹32,914       +8.4%       -2.1%     
  ATGL        9      OIL&GAS    01-Dec-20   360.4       388.5       1084   ₹30,464       +7.8%       +5.3%     
  IOLCP       25     HEALTH     03-Aug-20   140.5       129.2       2105   ₹-23,757      -8.0%       -3.2%     

  AFTER: Invested ₹9,258,351 | Cash ₹117,597 | Total ₹9,375,948 | Positions 17/20 | Slot ₹468,908

========================================================================
  REBALANCE #38  —  01 Apr 2021
  NAV: ₹12,095,818  |  Slot: ₹604,791  |  Cash: ₹117,597
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SFL         106    CON DUR    01-Feb-21   983.1       994.1       476    ₹5,212        +1.1%     59d   
  PGHL        192    HEALTH     01-Feb-21   6,114.1     5,279.3     76     ₹-63,449      -13.7%    59d   
  IOLCP       186    HEALTH     03-Aug-20   140.5       103.6       2105   ₹-77,588      -26.2%    241d  

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIENSOL  4      ENERGY     4.310    0.28   +435.3%   +129.1%   999.2       605    ₹604,516      +21.3%    
  DEEPAKNTR   10     MFG        2.897    0.07   +338.0%   +77.5%    1,618.8     373    ₹603,810      +7.1%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  DIXON       13     CON DUR    03-Feb-20   946.1       3,580.2     274    ₹721,765      +278.4%     -5.8%     
  ADANIENT    2      METAL      01-Oct-20   307.3       1,104.0     1101   ₹877,230      +259.3%     +16.5%    
  ATGL        1      OIL&GAS    01-Dec-20   360.4       1,058.9     1084   ₹757,222      +193.8%     +31.6%    
  CGPOWER     3      ENERGY     01-Oct-20   24.3        67.9        13959  ₹609,487      +180.0%     +9.6%     
  ALKYLAMINE  16     MFG        03-Aug-20   893.0       2,249.1     331    ₹448,840      +151.8%     +6.3%     
  APLAPOLLO   8      METAL      01-Oct-20   291.4       654.3       1161   ₹421,359      +124.5%     +5.2%     
  VAIBHAVGBL  5      CON DUR    01-Oct-20   344.5       727.2       982    ₹375,793      +111.1%     +4.1%     
  TTML        6      CONSUMP    01-Dec-20   7.0         14.1        55826  ₹393,573      +100.7%     -1.8%     
  AFFLE       9      IT         01-Oct-20   572.0       1,120.5     591    ₹324,128      +95.9%      +3.0%     
  TATAELXSI   12     IT         01-Dec-20   1,527.0     2,599.1     255    ₹273,380      +70.2%      +2.7%     
  AARTIDRUGS  34     HEALTH     03-Aug-20   429.0       708.6       689    ₹192,642      +65.2%      +2.8%     
  TATACHEM    7      MFG        01-Feb-21   453.6       715.4       1033   ₹270,374      +57.7%      +5.5%     
  PERSISTENT  25     IT         01-Feb-21   739.7       941.6       633    ₹127,801      +27.3%      +6.1%     
  LAURUSLABS  27     HEALTH     01-Dec-20   317.1       359.2       1232   ₹51,824       +13.3%      +2.0%     

  AFTER: Invested ₹12,094,053 | Cash ₹330 | Total ₹12,094,383 | Positions 16/20 | Slot ₹604,791

========================================================================
  REBALANCE #39  —  01 Jun 2021
  NAV: ₹14,572,145  |  Slot: ₹728,607  |  Cash: ₹330
========================================================================
  [SECTOR CAP≤4] dropped: KPRMILL

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIENT    —      METAL      01-Oct-20   307.3       1,412.2     1101   ₹1,216,518    +359.6%   243d  
  DIXON       54     CON DUR    03-Feb-20   946.1       4,098.5     274    ₹863,758      +333.2%   484d  
  CGPOWER     —      ENERGY     01-Oct-20   24.3        82.9        13959  ₹819,085      +242.0%   243d  
  TTML        48     CONSUMP    01-Dec-20   7.0         16.4        55826  ₹521,973      +133.6%   182d  
  VAIBHAVGBL  52     CON DUR    01-Oct-20   344.5       755.0       982    ₹403,114      +119.2%   243d  
  AFFLE       103    IT         01-Oct-20   572.0       1,056.1     591    ₹286,074      +84.6%    243d  
  AARTIDRUGS  109    HEALTH     03-Aug-20   429.0       726.3       689    ₹204,836      +69.3%    302d  
  TATACHEM    —      MFG        01-Feb-21   453.6       648.6       1033   ₹201,417      +43.0%    120d  

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  INTELLECT   3      IT         4.661    0.67   +1018.7%  +69.2%    733.7       993    ₹728,592      -0.1%     
  PRINCEPIPE  4      MFG        4.195    0.66   +817.7%   +74.9%    703.3       1036   ₹728,606      +7.7%     
  PRAJIND     5      ENERGY     3.965    0.40   +462.5%   +133.6%   311.3       2340   ₹728,462      +2.1%     
  HIKAL       6      HEALTH     3.432    0.69   +230.7%   +137.6%   375.1       1942   ₹728,472      +6.6%     
  JSWSTEEL    7      METAL      3.386    0.56   +281.0%   +70.2%    657.4       1108   ₹728,430      +0.4%     
  MASTEK      8      IT         3.042    0.21   +634.3%   +61.1%    1,858.7     391    ₹726,754      +5.9%     
  BALAMINES   11     MFG        2.906    0.59   +579.2%   +64.3%    2,686.7     271    ₹728,084      +4.1%     
  LUXIND      13     CON DUR    2.672    0.51   +228.9%   +68.2%    2,984.9     244    ₹728,312      +24.8%    
  CDSL        14     FIN SVC    2.639    0.33   +294.5%   +55.9%    459.4       1586   ₹728,594      +10.6%    

  HOLDS (8)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    01-Dec-20   360.4       1,438.2     1084   ₹1,168,406    +299.1%     +10.6%    
  ALKYLAMINE  12     MFG        03-Aug-20   893.0       3,513.2     331    ₹867,262      +293.4%     +3.0%     
  TATAELXSI   16     IT         01-Dec-20   1,527.0     3,384.3     255    ₹473,597      +121.6%     +1.8%     
  APLAPOLLO   21     METAL      01-Oct-20   291.4       639.9       1161   ₹404,553      +119.6%     +1.6%     
  LAURUSLABS  10     HEALTH     01-Dec-20   317.1       524.5       1232   ₹255,437      +65.4%      +7.1%     
  PERSISTENT  9      IT         01-Feb-21   739.7       1,202.4     633    ₹292,847      +62.5%      +5.6%     
  ADANIENSOL  2      ENERGY     01-Apr-21   999.2       1,499.4     605    ₹302,651      +50.1%      +13.2%    
  DEEPAKNTR   35     MFG        01-Apr-21   1,618.8     1,730.5     373    ₹41,654       +6.9%       -0.6%     

  AFTER: Invested ₹13,841,957 | Cash ₹722,405 | Total ₹14,564,362 | Positions 17/20 | Slot ₹728,607

========================================================================
  REBALANCE #40  —  02 Aug 2021
  NAV: ₹16,260,919  |  Slot: ₹813,046  |  Cash: ₹722,405
========================================================================
  [SECTOR CAP≤4] dropped: ECLERX, FSL, HGS, REDINGTON

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ATGL        102    OIL&GAS    01-Dec-20   360.4       887.1       1084   ₹570,939      +146.2%   244d  
  DEEPAKNTR   86     MFG        01-Apr-21   1,618.8     2,032.0     373    ₹154,131      +25.5%    123d  
  JSWSTEEL    56     METAL      01-Jun-21   657.4       713.8       1108   ₹62,408       +8.6%     62d   
  INTELLECT   57     IT         01-Jun-21   733.7       724.8       993    ₹-8,837       -1.2%     62d   
  ADANIENSOL  149    ENERGY     01-Apr-21   999.2       908.8       605    ₹-54,692      -9.0%     123d  

  ENTRIES (5)
  [52w filter blocked 2: TTML(-26.4%), HFCL(-22.5%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BSE         5      FIN SVC    3.041    0.37   +151.3%   +106.8%   133.3       6100   ₹812,962      +10.0%    
  ICIL        7      CONSUMP    2.801    0.31   +326.2%   +98.9%    261.4       3110   ₹812,993      +21.0%    
  KPRMILL     18     MFG        2.325    0.48   +360.6%   +34.6%    383.4       2120   ₹812,719      +7.0%     
  TATASTEEL   19     METAL      2.303    0.92   +295.7%   +35.4%    121.6       6686   ₹812,996      +8.9%     
  WELSPUNLIV  20     CONSUMP    2.264    0.49   +257.1%   +68.2%    135.5       6000   ₹812,938      +11.0%    

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ALKYLAMINE  30     MFG        03-Aug-20   893.0       4,316.7     331    ₹1,133,227    +383.4%     +9.1%     
  APLAPOLLO   12     METAL      01-Oct-20   291.4       874.6       1161   ₹677,136      +200.1%     +7.8%     
  TATAELXSI   25     IT         01-Dec-20   1,527.0     4,010.4     255    ₹633,258      +162.6%     +0.6%     
  PERSISTENT  16     IT         01-Feb-21   739.7       1,515.7     633    ₹491,155      +104.9%     +6.2%     
  LAURUSLABS  22     HEALTH     01-Dec-20   317.1       644.6       1232   ₹403,478      +103.3%     +1.3%     
  CDSL        6      FIN SVC    01-Jun-21   459.4       665.3       1586   ₹326,541      +44.8%      +8.7%     
  HIKAL       11     HEALTH     01-Jun-21   375.1       539.1       1942   ₹318,474      +43.7%      +5.3%     
  LUXIND      2      CON DUR    01-Jun-21   2,984.9     4,167.5     244    ₹288,560      +39.6%      +7.8%     
  MASTEK      8      IT         01-Jun-21   1,858.7     2,537.1     391    ₹265,236      +36.5%      +6.3%     
  BALAMINES   26     MFG        01-Jun-21   2,686.7     3,261.3     271    ₹155,741      +21.4%      +9.2%     
  PRAJIND     9      ENERGY     01-Jun-21   311.3       357.6       2340   ₹108,333      +14.9%      +2.0%     
  PRINCEPIPE  10     MFG        01-Jun-21   703.3       681.9       1036   ₹-22,125      -3.0%       -0.8%     

  AFTER: Invested ₹15,823,203 | Cash ₹432,890 | Total ₹16,256,093 | Positions 17/20 | Slot ₹813,046

========================================================================
  REBALANCE #41  —  01 Oct 2021
  NAV: ₹16,951,520  |  Slot: ₹847,576  |  Cash: ₹432,890
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ALKYLAMINE  89     MFG        03-Aug-20   893.0       3,765.1     331    ₹950,655      +321.6%   424d  
  APLAPOLLO   69     METAL      01-Oct-20   291.4       837.3       1161   ₹633,762      +187.3%   365d  
  LAURUSLABS  159    HEALTH     01-Dec-20   317.1       609.9       1232   ₹360,644      +92.3%    304d  
  HIKAL       59     HEALTH     01-Jun-21   375.1       566.4       1942   ₹371,493      +51.0%    122d  
  CDSL        29     FIN SVC    01-Jun-21   459.4       618.0       1586   ₹251,553      +34.5%    122d  
  LUXIND      116    CON DUR    01-Jun-21   2,984.9     3,603.0     244    ₹150,816      +20.7%    122d  
  PRAJIND     35     ENERGY     01-Jun-21   311.3       334.8       2340   ₹54,997       +7.5%     122d  
  PRINCEPIPE  124    MFG        01-Jun-21   703.3       685.6       1036   ₹-18,315      -2.5%     122d  
  BSE         36     FIN SVC    02-Aug-21   133.3       129.4       6100   ₹-23,850      -2.9%     60d   
  TATASTEEL   27     METAL      02-Aug-21   121.6       111.9       6686   ₹-64,691      -8.0%     60d   

  ENTRIES (11)
  [52w filter blocked 2: TTML(-31.4%), ADANIENSOL(-20.7%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ATGL        4      OIL&GAS    3.577    0.63   +674.9%   +46.9%    1,421.5     596    ₹847,224      +3.2%     
  IRCTC       5      PSE        3.534    0.67   +181.5%   +86.7%    725.8       1167   ₹846,981      +6.9%     
  SRF         6      MFG        3.230    0.68   +172.9%   +51.1%    2,186.9     387    ₹846,337      +3.6%     
  IEX         7      FIN SVC    3.224    0.36   +230.8%   +63.7%    191.1       4436   ₹847,510      +6.2%     
  BAJAJFINSV  9      FIN SVC    2.863    0.95   +196.1%   +45.4%    1,712.9     494    ₹846,162      -0.1%     
  HATSUN      10     FMCG       2.715    0.32   +153.0%   +60.0%    1,350.7     627    ₹846,877      +6.9%     
  CARBORUNIV  11     MFG        2.618    0.47   +252.3%   +38.0%    858.7       987    ₹847,524      +2.0%     
  BAJAJHLDNG  12     FIN SVC    2.532    0.37   +99.7%    +34.8%    4,433.3     191    ₹846,758      +4.4%     
  PRESTIGE    13     REALTY     2.514    0.92   +97.5%    +69.3%    477.2       1776   ₹847,518      +8.8%     
  MPHASIS     14     IT         2.430    0.38   +143.3%   +42.4%    2,770.0     305    ₹844,860      -1.8%     
  SOLARINDS   15     DEFENCE    2.398    0.54   +112.1%   +36.8%    2,153.2     393    ₹846,217      +10.8%    

  HOLDS (7)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   8      IT         01-Dec-20   1,527.0     5,491.9     255    ₹1,011,047    +259.7%     +6.9%     
  PERSISTENT  22     IT         01-Feb-21   739.7       1,760.2     633    ₹645,932      +137.9%     +2.2%     
  BALAMINES   3      MFG        01-Jun-21   2,686.7     4,471.8     271    ₹483,781      +66.4%      +2.0%     
  MASTEK      21     IT         01-Jun-21   1,858.7     2,935.1     391    ₹420,878      +57.9%      +2.8%     
  WELSPUNLIV  18     CONSUMP    02-Aug-21   135.5       160.4       6000   ₹149,713      +18.4%      +10.5%    
  KPRMILL     19     MFG        02-Aug-21   383.4       417.3       2120   ₹72,052       +8.9%       +0.4%     
  ICIL        20     CONSUMP    02-Aug-21   261.4       269.1       3110   ₹23,976       +2.9%       +2.2%     

  AFTER: Invested ₹16,872,477 | Cash ₹67,984 | Total ₹16,940,460 | Positions 18/20 | Slot ₹847,576

========================================================================
  REBALANCE #42  —  01 Dec 2021
  NAV: ₹16,972,474  |  Slot: ₹848,624  |  Cash: ₹67,984
========================================================================
  [SECTOR CAP≤4] dropped: ELGIEQUIP

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MASTEK      74     IT         01-Jun-21   1,858.7     2,571.4     391    ₹278,661      +38.3%    183d  
  BALAMINES   142    MFG        01-Jun-21   2,686.7     2,901.5     271    ₹58,229       +8.0%     183d  
  WELSPUNLIV  88     CONSUMP    02-Aug-21   135.5       138.1       6000   ₹15,650       +1.9%     121d  
  BAJAJFINSV  75     FIN SVC    01-Oct-21   1,712.9     1,733.3     494    ₹10,106       +1.2%     61d   
  MPHASIS     44     IT         01-Oct-21   2,770.0     2,756.6     305    ₹-4,090       -0.5%     61d   
  HATSUN      52     FMCG       01-Oct-21   1,350.7     1,254.4     627    ₹-60,374      -7.1%     61d   
  PRESTIGE    60     REALTY     01-Oct-21   477.2       439.3       1776   ₹-67,315      -7.9%     61d   
  SRF         82     MFG        01-Oct-21   2,186.9     1,988.5     387    ₹-76,790      -9.1%     61d   
  ICIL        170    CONSUMP    02-Aug-21   261.4       228.2       3110   ₹-103,346     -12.7%    121d  

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TTML        1      CONSUMP    9.918    0.38   +1529.7%  +212.6%   118.2       7182   ₹848,553      +38.2%    
  CENTURYPLY  5      CON DUR    3.015    0.69   +191.3%   +52.8%    578.0       1468   ₹848,482      -6.5%     
  SFL         6      CON DUR    2.875    0.54   +141.5%   +38.6%    1,616.7     524    ₹847,138      +7.0%     
  TCIEXP      7      INFRA      2.686    0.49   +193.4%   +54.3%    2,237.9     379    ₹848,177      +12.8%    
  ADANIENSOL  10     ENERGY     2.507    0.90   +373.4%   +19.4%    1,797.2     472    ₹848,255      -4.2%     
  KEI         11     MFG        2.501    0.80   +199.7%   +46.3%    1,126.2     753    ₹848,065      +9.5%     
  BSE         12     FIN SVC    2.461    0.86   +193.8%   +38.0%    175.3       4840   ₹848,465      +9.5%     
  BEML        13     DEFENCE    2.436    0.82   +178.3%   +41.5%    759.6       1117   ₹848,524      +11.2%    

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   9      IT         01-Dec-20   1,527.0     5,587.5     255    ₹1,035,422    +265.9%     -3.0%     
  PERSISTENT  4      IT         01-Feb-21   739.7       2,032.5     633    ₹818,324      +174.8%     +2.7%     
  SOLARINDS   2      DEFENCE    01-Oct-21   2,153.2     2,813.8     393    ₹259,598      +30.7%      +6.1%     
  KPRMILL     8      MFG        02-Aug-21   383.4       500.8       2120   ₹248,882      +30.6%      +1.9%     
  IEX         3      FIN SVC    01-Oct-21   191.1       224.8       4436   ₹149,733      +17.7%      -4.0%     
  ATGL        18     OIL&GAS    01-Oct-21   1,421.5     1,623.7     596    ₹120,513      +14.2%      +1.9%     
  BAJAJHLDNG  19     FIN SVC    01-Oct-21   4,433.3     4,853.5     191    ₹80,263       +9.5%       +4.8%     
  IRCTC       17     PSE        01-Oct-21   725.8       776.7       1167   ₹59,422       +7.0%       -4.6%     
  CARBORUNIV  41     MFG        01-Oct-21   858.7       875.4       987    ₹16,506       +1.9%       -1.4%     

  AFTER: Invested ₹16,326,897 | Cash ₹637,520 | Total ₹16,964,417 | Positions 17/20 | Slot ₹848,624

========================================================================
  REBALANCE #43  —  01 Feb 2022
  NAV: ₹18,067,628  |  Slot: ₹903,381  |  Cash: ₹637,520
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  IRCTC       33     PSE        01-Oct-21   725.8       819.7       1167   ₹109,601      +12.9%    123d  
  SOLARINDS   95     DEFENCE    01-Oct-21   2,153.2     2,286.3     393    ₹52,310       +6.2%     123d  
  CARBORUNIV  50     MFG        01-Oct-21   858.7       865.3       987    ₹6,505        +0.8%     123d  
  TCIEXP      71     INFRA      01-Dec-21   2,237.9     1,874.1     379    ₹-137,881     -16.3%    62d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SHARDACROP  2      FMCG       3.776    0.52   +109.2%   +88.9%    567.5       1591   ₹902,944      +32.2%    
  ELGIEQUIP   6      MFG        3.268    0.68   +119.4%   +70.6%    336.5       2684   ₹903,045      +2.1%     
  SCHAEFFLER  8      MNC        2.828    0.62   +111.5%   +24.1%    1,790.7     504    ₹902,493      +3.7%     
  UNOMINDA    10     AUTO       2.711    1.07   +123.3%   +43.9%    550.3       1641   ₹903,069      +0.7%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   7      IT         01-Dec-20   1,527.0     7,089.3     255    ₹1,418,393    +364.3%     +10.5%    
  PERSISTENT  11     IT         01-Feb-21   739.7       2,183.0     633    ₹913,604      +195.1%     +3.2%     
  KPRMILL     3      MFG        02-Aug-21   383.4       673.7       2120   ₹615,624      +75.7%      +0.6%     
  ATGL        4      OIL&GAS    01-Oct-21   1,421.5     1,861.9     596    ₹262,481      +31.0%      +3.3%     
  BSE         5      FIN SVC    01-Dec-21   175.3       211.4       4840   ₹174,660      +20.6%      +0.2%     
  TTML        1      CONSUMP    01-Dec-21   118.2       141.8       7182   ₹169,495      +20.0%      -26.2%    
  IEX         30     FIN SVC    01-Oct-21   191.1       216.0       4436   ₹110,524      +13.0%      -5.8%     
  ADANIENSOL  9      ENERGY     01-Dec-21   1,797.2     1,986.9     472    ₹89,562       +10.6%      +1.7%     
  BAJAJHLDNG  32     FIN SVC    01-Oct-21   4,433.3     4,889.1     191    ₹87,055       +10.3%      -0.4%     
  SFL         13     CON DUR    01-Dec-21   1,616.7     1,700.6     524    ₹43,977       +5.2%       +0.7%     
  CENTURYPLY  35     CON DUR    01-Dec-21   578.0       598.2       1468   ₹29,615       +3.5%       -2.7%     
  BEML        45     DEFENCE    01-Dec-21   759.6       736.8       1117   ₹-25,508      -3.0%       -0.3%     
  KEI         34     MFG        01-Dec-21   1,126.2     1,088.9     753    ₹-28,144      -3.3%       -2.9%     

  AFTER: Invested ₹17,622,223 | Cash ₹441,116 | Total ₹18,063,339 | Positions 17/20 | Slot ₹903,381

========================================================================
  REBALANCE #44  —  01 Apr 2022
  NAV: ₹19,575,068  |  Slot: ₹978,753  |  Cash: ₹441,116
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BSE         5      FIN SVC    01-Dec-21   175.3       297.8       4840   ₹592,652      +69.8%    121d  
  KPRMILL     56     MFG        02-Aug-21   383.4       613.5       2120   ₹487,804      +60.0%    242d  
  IEX         94     FIN SVC    01-Oct-21   191.1       214.3       4436   ₹103,013      +12.2%    182d  
  SHARDACROP  —      OTHER      01-Feb-22   567.5       598.8       1591   ₹49,794       +5.5%     59d   
  BEML        76     DEFENCE    01-Dec-21   759.6       731.3       1117   ₹-31,612      -3.7%     121d  
  ELGIEQUIP   103    MFG        01-Feb-22   336.5       290.1       2684   ₹-124,352     -13.8%    59d   
  UNOMINDA    202    AUTO       01-Feb-22   550.3       460.4       1641   ₹-147,477     -16.3%    59d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BCG         1      IT         7.008    0.54   +2544.3%  -7.7%     101.3       9659   ₹978,656      +23.4%    
  RHIM        4      METAL      3.347    0.71   +171.5%   +64.9%    598.5       1635   ₹978,502      +5.8%     
  LINDEINDIA  5      MFG        3.040    0.49   +126.0%   +54.1%    3,817.8     256    ₹977,353      +16.2%    
  POLYPLEX    6      MFG        2.727    0.80   +223.5%   +34.3%    2,280.6     429    ₹978,383      +13.0%    
  POWERINDIA  8      ENERGY     2.486    0.96   +158.1%   +36.8%    3,488.7     280    ₹976,842      +2.6%     
  HAL         9      DEFENCE    2.397    0.95   +59.8%    +28.3%    723.8       1352   ₹978,583      +7.9%     
  SOLARINDS   12     DEFENCE    2.082    0.94   +123.9%   +21.2%    2,851.5     343    ₹978,061      +10.3%    

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   4      IT         01-Dec-20   1,527.0     8,464.4     255    ₹1,769,047    +454.3%     +12.8%    
  PERSISTENT  41     IT         01-Feb-21   739.7       2,292.8     633    ₹983,071      +209.9%     +4.7%     
  ATGL        18     OIL&GAS    01-Oct-21   1,421.5     2,246.2     596    ₹491,516      +58.0%      +16.1%    
  TTML        2      CONSUMP    01-Dec-21   118.2       175.0       7182   ₹408,297      +48.1%      +19.9%    
  ADANIENSOL  12     ENERGY     01-Dec-21   1,797.2     2,421.4     472    ₹294,670      +34.7%      +4.1%     
  CENTURYPLY  20     CON DUR    01-Dec-21   578.0       705.2       1468   ₹186,809      +22.0%      +6.1%     
  BAJAJHLDNG  50     FIN SVC    01-Oct-21   4,433.3     5,052.3     191    ₹118,227      +14.0%      +6.5%     
  KEI         49     MFG        01-Dec-21   1,126.2     1,241.7     753    ₹86,970       +10.3%      +9.3%     
  SFL         39     CON DUR    01-Dec-21   1,616.7     1,760.5     524    ₹75,351       +8.9%       +3.2%     
  SCHAEFFLER  36     MNC        01-Feb-22   1,790.7     1,848.8     504    ₹29,289       +3.2%       +3.8%     

  AFTER: Invested ₹18,984,235 | Cash ₹582,703 | Total ₹19,566,938 | Positions 17/20 | Slot ₹978,753

========================================================================
  REBALANCE #45  —  01 Jun 2022
  NAV: ₹18,074,079  |  Slot: ₹903,704  |  Cash: ₹582,703
========================================================================

  [REGIME OFF] Nifty 500 14,082.9 < EMA200 14,338.7 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   9      IT         01-Dec-20   1,527.0     8,173.1     255    ₹1,694,749    +435.2%     +5.6%     
  PERSISTENT  100    IT         01-Feb-21   739.7       1,815.3     633    ₹680,806      +145.4%     -0.8%     
  ATGL        16     OIL&GAS    01-Oct-21   1,421.5     2,322.5     596    ₹536,984      +63.4%      -2.8%     
  SCHAEFFLER  3      MNC        01-Feb-22   1,790.7     2,324.5     504    ₹269,045      +29.8%      +11.1%    
  HAL         4      DEFENCE    01-Apr-22   723.8       899.0       1352   ₹236,926      +24.2%      +9.7%     
  KEI         24     MFG        01-Dec-21   1,126.2     1,290.7     753    ₹123,869      +14.6%      +7.8%     
  ADANIENSOL  —      ENERGY     01-Dec-21   1,797.2     1,957.9     472    ₹75,874       +8.9%       -13.5%    
  BAJAJHLDNG  93     FIN SVC    01-Oct-21   4,433.3     4,726.9     191    ₹56,079       +6.6%       +0.4%     
  TTML        1      CONSUMP    01-Dec-21   118.2       121.8       7182   ₹26,573       +3.1%       -5.8%     
  POLYPLEX    6      MFG        01-Apr-22   2,280.6     2,351.3     429    ₹30,346       +3.1%       +7.3%     
  POWERINDIA  53     ENERGY     01-Apr-22   3,488.7     3,502.2     280    ₹3,771        +0.4%       +9.4%     
  RHIM        32     METAL      01-Apr-22   598.5       585.3       1635   ₹-21,546      -2.2%       +5.1%     
  CENTURYPLY  135    CON DUR    01-Dec-21   578.0       561.6       1468   ₹-24,071      -2.8% ⚠     +1.6%     
  SFL         —      CON DUR    01-Dec-21   1,616.7     1,491.2     524    ₹-65,736      -7.8%       -5.3%     
  SOLARINDS   34     DEFENCE    01-Apr-22   2,851.5     2,611.8     343    ₹-82,228      -8.4%       -3.9%     
  LINDEINDIA  40     MFG        01-Apr-22   3,817.8     3,049.0     256    ₹-196,812     -20.1%      -1.4%     
  BCG         —      IT         01-Apr-22   101.3       60.5        9659   ₹-394,241     -40.3%      -8.2%     
  ⚠  WAZ < 0 (momentum below universe mean): CENTURYPLY

  AFTER: Invested ₹17,491,376 | Cash ₹582,703 | Total ₹18,074,079 | Positions 17/20 | Slot ₹903,704

========================================================================
  REBALANCE #46  —  01 Aug 2022
  NAV: ₹19,370,848  |  Slot: ₹968,542  |  Cash: ₹582,703
========================================================================

  EXITS (11)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PERSISTENT  196    IT         01-Feb-21   739.7       1,780.3     633    ₹658,656      +140.7%   546d  
  ATGL        1      OIL&GAS    01-Oct-21   1,421.5     3,213.2     596    ₹1,067,869    +126.0%   304d  
  ADANIENSOL  —      ENERGY     01-Dec-21   1,797.2     3,261.8     472    ₹691,291      +81.5%    243d  
  BAJAJHLDNG  84     FIN SVC    01-Oct-21   4,433.3     4,946.0     191    ₹97,930       +11.6%    304d  
  KEI         53     MFG        01-Dec-21   1,126.2     1,249.1     753    ₹92,477       +10.9%    243d  
  CENTURYPLY  127    CON DUR    01-Dec-21   578.0       586.5       1468   ₹12,473       +1.5%     243d  
  POLYPLEX    68     MFG        01-Apr-22   2,280.6     2,299.7     429    ₹8,190        +0.8%     122d  
  SOLARINDS   99     DEFENCE    01-Apr-22   2,851.5     2,684.1     343    ₹-57,411      -5.9%     122d  
  SFL         —      CON DUR    01-Dec-21   1,616.7     1,486.9     524    ₹-67,989      -8.0%     243d  
  RHIM        145    METAL      01-Apr-22   598.5       504.6       1635   ₹-153,557     -15.7%    122d  
  BCG         —      IT         01-Apr-22   101.3       47.6        9659   ₹-519,267     -53.1%    122d  

  ENTRIES (12)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TVSMOTOR    1      AUTO       3.741    0.84   +62.0%    +47.7%    911.5       1062   ₹968,002      +7.4%     
  TIMKEN      2      MNC        3.653    0.71   +86.9%    +57.3%    2,949.9     328    ₹967,577      +8.5%     
  M&M         3      AUTO       3.645    1.01   +71.6%    +39.5%    1,193.6     811    ₹968,010      +8.2%     
  ITC         6      FMCG       2.933    0.76   +54.3%    +21.3%    254.0       3813   ₹968,438      +3.9%     
  VBL         7      FMCG       2.903    0.69   +83.6%    +27.4%    183.1       5289   ₹968,521      +7.8%     
  BLUEDART    8      INFRA      2.845    0.76   +58.4%    +26.6%    8,621.4     112    ₹965,596      +6.6%     
  SIEMENS     9      ENERGY     2.642    0.95   +42.9%    +25.4%    1,598.0     606    ₹968,380      +3.8%     
  FINEORG     10     MFG        2.534    0.98   +87.6%    +25.3%    5,475.9     176    ₹963,765      +7.1%     
  CUMMINSIND  11     INFRA      2.526    0.80   +48.7%    +20.6%    1,157.0     837    ₹968,371      +6.1%     
  AIAENG      12     MFG        2.493    0.53   +25.3%    +30.1%    2,434.5     397    ₹966,498      +5.5%     
  BEL         13     DEFENCE    2.450    1.02   +52.8%    +22.9%    90.6        10694  ₹968,504      +10.0%    
  BDL         14     DEFENCE    2.430    0.93   +110.0%   +19.9%    400.5       2418   ₹968,506      +11.2%    

  HOLDS (6)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   16     IT         01-Dec-20   1,527.0     8,324.1     255    ₹1,733,258    +445.1%     +5.1%     
  SCHAEFFLER  5      MNC        01-Feb-22   1,790.7     2,742.8     504    ₹479,854      +53.2%      +13.4%    
  HAL         6      DEFENCE    01-Apr-22   723.8       963.6       1352   ₹324,140      +33.1%      +8.3%     
  TTML        17     CONSUMP    01-Dec-21   118.2       114.2       7182   ₹-28,369      -3.3%       -0.7%     
  LINDEINDIA  18     MFG        01-Apr-22   3,817.8     3,657.0     256    ₹-41,157      -4.2%       +5.0%     
  POWERINDIA  34     ENERGY     01-Apr-22   3,488.7     3,305.8     280    ₹-51,231      -5.2%       +0.6%     

  AFTER: Invested ₹19,099,873 | Cash ₹257,189 | Total ₹19,357,062 | Positions 18/20 | Slot ₹968,542

========================================================================
  REBALANCE #47  —  03 Oct 2022
  NAV: ₹20,101,026  |  Slot: ₹1,005,051  |  Cash: ₹257,189
========================================================================
  [SECTOR CAP≤4] dropped: SOLARINDS

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TATAELXSI   127    IT         01-Dec-20   1,527.0     7,934.6     255    ₹1,633,924    +419.6%   671d  
  AIAENG      81     MFG        01-Aug-22   2,434.5     2,434.7     397    ₹92           +0.0%     63d   
  POWERINDIA  95     ENERGY     01-Apr-22   3,488.7     3,469.8     280    ₹-5,306       -0.5%     185d  
  SIEMENS     70     ENERGY     01-Aug-22   1,598.0     1,568.0     606    ₹-18,200      -1.9%     63d   
  CUMMINSIND  68     INFRA      01-Aug-22   1,157.0     1,129.1     837    ₹-23,324      -2.4%     63d   
  LINDEINDIA  146    MFG        01-Apr-22   3,817.8     3,315.8     256    ₹-128,498     -13.1%    185d  
  TTML        52     CONSUMP    01-Dec-21   118.2       102.7       7182   ₹-110,962     -13.1%    306d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MAZDOCK     1      DEFENCE    4.734    1.17   +110.5%   +107.6%   247.0       4068   ₹1,004,970    +17.6%    
  TATAINVEST  2      FIN SVC    3.890    1.05   +87.1%    +73.3%    224.3       4479   ₹1,004,844    +3.4%     
  RITES       4      INFRA      3.218    0.51   +32.6%    +47.5%    149.3       6733   ₹1,005,018    +11.1%    
  TIINDIA     5      AUTO       3.132    0.90   +97.9%    +53.1%    2,688.0     373    ₹1,002,620    +3.6%     
  TRITURBINE  7      ENERGY     2.907    0.90   +82.3%    +65.7%    250.8       4007   ₹1,004,968    +12.3%    
  BAJAJHLDNG  12     FIN SVC    2.645    0.87   +38.5%    +44.7%    6,221.9     161    ₹1,001,725    +1.1%     
  ZFCVINDIA   13     AUTO       2.567    0.49   +39.6%    +28.2%    1,643.3     611    ₹1,004,061    +0.5%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  SCHAEFFLER  4      MNC        01-Feb-22   1,790.7     3,084.4     504    ₹652,043      +72.2%      -2.1%     
  HAL         10     DEFENCE    01-Apr-22   723.8       1,092.9     1352   ₹498,979      +51.0%      -3.3%     
  FINEORG     11     MFG        01-Aug-22   5,475.9     6,593.5     176    ₹196,692      +20.4%      -2.5%     
  VBL         13     FMCG       01-Aug-22   183.1       212.1       5289   ₹153,497      +15.8%      +0.2%     
  TVSMOTOR    18     AUTO       01-Aug-22   911.5       979.2       1062   ₹71,922       +7.4%       -2.8%     
  BDL         12     DEFENCE    01-Aug-22   400.5       427.7       2418   ₹65,702       +6.8%       +1.1%     
  ITC         25     FMCG       01-Aug-22   254.0       267.9       3813   ₹53,225       +5.5%       -1.9%     
  BEL         22     DEFENCE    01-Aug-22   90.6        94.6        10694  ₹43,494       +4.5%       -5.2%     
  TIMKEN      20     MNC        01-Aug-22   2,949.9     3,057.3     328    ₹35,202       +3.6%       +2.8%     
  BLUEDART    37     INFRA      01-Aug-22   8,621.4     8,901.2     112    ₹31,339       +3.2%       +3.5%     
  M&M         29     AUTO       01-Aug-22   1,193.6     1,206.7     811    ₹10,636       +1.1%       -1.5%     

  AFTER: Invested ₹19,428,930 | Cash ₹663,750 | Total ₹20,092,681 | Positions 18/20 | Slot ₹1,005,051

========================================================================
  REBALANCE #48  —  01 Dec 2022
  NAV: ₹21,655,306  |  Slot: ₹1,082,765  |  Cash: ₹663,750
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MAZDOCK     1      DEFENCE    03-Oct-22   247.0       443.8       4068   ₹800,465      +79.7%    59d   
  SCHAEFFLER  56     MNC        01-Feb-22   1,790.7     2,684.3     504    ₹450,383      +49.9%    303d  
  FINEORG     78     MFG        01-Aug-22   5,475.9     6,121.5     176    ₹113,623      +11.8%    122d  
  M&M         59     AUTO       01-Aug-22   1,193.6     1,247.3     811    ₹43,561       +4.5%     122d  
  BAJAJHLDNG  62     FIN SVC    03-Oct-22   6,221.9     6,038.0     161    ₹-29,602      -3.0%     59d   
  ZFCVINDIA   138    AUTO       03-Oct-22   1,643.3     1,572.9     611    ₹-43,024      -4.3%     59d   
  BLUEDART    225    INFRA      01-Aug-22   8,621.4     7,461.6     112    ₹-129,902     -13.5%    122d  

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RVNL        1      PSE        5.585    0.95   +128.6%   +129.8%   72.2        14989  ₹1,082,729    +19.4%    
  IRFC        2      FIN SVC    5.031    0.51   +55.2%    +69.4%    32.4        33416  ₹1,082,753    +18.8%    
  UCOBANK     3      PSU BNK    3.686    0.72   +55.0%    +69.2%    19.6        55233  ₹1,082,757    +14.0%    
  GODFRYPHLP  5      FMCG       3.219    0.93   +57.5%    +63.7%    577.8       1874   ₹1,082,724    +5.4%     
  RHIM        6      METAL      2.993    0.66   +138.0%   +35.1%    788.3       1373   ₹1,082,387    +11.6%    
  AMBUJACEM   8      INFRA      2.867    0.92   +59.3%    +41.3%    570.9       1896   ₹1,082,463    +3.4%     
  CUMMINSIND  10     INFRA      2.776    0.82   +67.1%    +20.6%    1,372.2     789    ₹1,082,627    +5.9%     
  HUDCO       12     FIN SVC    2.699    0.96   +44.8%    +40.3%    47.2        22921  ₹1,082,722    +14.1%    

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         6      DEFENCE    01-Apr-22   723.8       1,323.9     1352   ₹811,371      +82.9%      +4.1%     
  VBL         15     FMCG       01-Aug-22   183.1       250.6       5289   ₹356,891      +36.8%      +9.5%     
  TIMKEN      32     MNC        01-Aug-22   2,949.9     3,485.3     328    ₹175,590      +18.1%      +6.7%     
  BDL         17     DEFENCE    01-Aug-22   400.5       471.0       2418   ₹170,315      +17.6%      +1.0%     
  TRITURBINE  21     ENERGY     03-Oct-22   250.8       288.1       4007   ₹149,467      +14.9%      +4.6%     
  TVSMOTOR    44     AUTO       01-Aug-22   911.5       1,032.8     1062   ₹128,855      +13.3%      -2.1%     
  RITES       24     INFRA      03-Oct-22   149.3       167.3       6733   ₹121,418      +12.1%      -1.4%     
  BEL         48     DEFENCE    01-Aug-22   90.6        100.1       10694  ₹102,086      +10.5%      -2.3%     
  ITC         22     FMCG       01-Aug-22   254.0       280.5       3813   ₹100,938      +10.4%      -0.9%     
  TIINDIA     28     AUTO       03-Oct-22   2,688.0     2,806.1     373    ₹44,069       +4.4%       +5.2%     
  TATAINVEST  12     FIN SVC    03-Oct-22   224.3       226.3       4479   ₹8,854        +0.9%       -1.1%     

  AFTER: Invested ₹21,636,596 | Cash ₹8,426 | Total ₹21,645,022 | Positions 19/20 | Slot ₹1,082,765

========================================================================
  REBALANCE #49  —  01 Feb 2023
  NAV: ₹20,356,083  |  Slot: ₹1,017,804  |  Cash: ₹8,426
========================================================================

  [REGIME OFF] Nifty 500 14,847.7 < EMA200 15,015.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         37     DEFENCE    01-Apr-22   723.8       1,136.0     1352   ₹557,339      +57.0%      -5.1%     
  UCOBANK     1      PSU BNK    01-Dec-22   19.6        27.7        55233  ₹445,371      +41.1%      -4.3%     
  VBL         18     FMCG       01-Aug-22   183.1       232.5       5289   ₹261,118      +27.0%      -4.7%     
  ITC         4      FMCG       01-Aug-22   254.0       298.5       3813   ₹169,752      +17.5%      +6.2%     
  BDL         44     DEFENCE    01-Aug-22   400.5       444.1       2418   ₹105,374      +10.9%      -1.9%     
  TVSMOTOR    53     AUTO       01-Aug-22   911.5       1,001.7     1062   ₹95,792       +9.9%       -0.2%     
  TIMKEN      26     MNC        01-Aug-22   2,949.9     3,165.1     328    ₹70,582       +7.3%       +2.4%     
  TRITURBINE  118    ENERGY     03-Oct-22   250.8       262.7       4007   ₹47,790       +4.8%       -1.7%     
  RITES       101    INFRA      03-Oct-22   149.3       152.5       6733   ₹21,756       +2.2%       -0.2%     
  GODFRYPHLP  27     FMCG       01-Dec-22   577.8       578.5       1874   ₹1,430        +0.1%       -5.4%     
  CUMMINSIND  16     INFRA      01-Dec-22   1,372.2     1,361.3     789    ₹-8,546       -0.8%       -0.0%     
  RHIM        21     METAL      01-Dec-22   788.3       777.1       1373   ₹-15,386      -1.4%       -4.5%     
  TIINDIA     81     AUTO       03-Oct-22   2,688.0     2,605.8     373    ₹-30,660      -3.1%       -1.5%     
  RVNL        2      PSE        01-Dec-22   72.2        69.9        14989  ₹-34,624      -3.2%       -1.7%     
  BEL         135    DEFENCE    01-Aug-22   90.6        87.4        10694  ₹-33,601      -3.5%       -7.3%     
  IRFC        3      FIN SVC    01-Dec-22   32.4        29.8        33416  ₹-85,481      -7.9%       -2.7%     
  TATAINVEST  99     FIN SVC    03-Oct-22   224.3       202.3       4479   ₹-98,928      -9.8%       -3.9%     
  HUDCO       30     FIN SVC    01-Dec-22   47.2        41.6        22921  ₹-128,217     -11.8%      -4.9%     
  AMBUJACEM   299    INFRA      01-Dec-22   570.9       328.3       1896   ₹-459,947     -42.5% ⚠    -28.6%    
  ⚠  WAZ < 0 (momentum below universe mean): AMBUJACEM

  AFTER: Invested ₹20,347,658 | Cash ₹8,426 | Total ₹20,356,083 | Positions 19/20 | Slot ₹1,017,804

========================================================================
  REBALANCE #50  —  03 Apr 2023
  NAV: ₹20,896,080  |  Slot: ₹1,044,804  |  Cash: ₹8,426
========================================================================

  [REGIME OFF] Nifty 500 14,602.0 < EMA200 14,892.1 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         13     DEFENCE    01-Apr-22   723.8       1,312.6     1352   ₹796,119      +81.4%      +1.9%     
  VBL         9      FMCG       01-Aug-22   183.1       280.4       5289   ₹514,416      +53.1%      +5.0%     
  UCOBANK     53     PSU BNK    01-Dec-22   19.6        25.1        55233  ₹304,025      +28.1%      +4.1%     
  TRITURBINE  20     ENERGY     03-Oct-22   250.8       319.8       4007   ₹276,317      +27.5%      +3.9%     
  ITC         4      FMCG       01-Aug-22   254.0       318.1       3813   ₹244,423      +25.2%      -0.3%     
  BDL         47     DEFENCE    01-Aug-22   400.5       486.1       2418   ₹206,974      +21.4%      +6.3%     
  TVSMOTOR    24     AUTO       01-Aug-22   911.5       1,072.3     1062   ₹170,746      +17.6%      +2.1%     
  CUMMINSIND  10     INFRA      01-Dec-22   1,372.2     1,543.4     789    ₹135,097      +12.5%      -1.1%     
  RITES       45     INFRA      03-Oct-22   149.3       163.0       6733   ₹92,189       +9.2%       +2.6%     
  BEL         70     DEFENCE    01-Aug-22   90.6        94.1        10694  ₹37,288       +3.9%       +3.4%     
  RVNL        14     PSE        01-Dec-22   72.2        72.5        14989  ₹4,328        +0.4%       +13.3%    
  GODFRYPHLP  74     FMCG       01-Dec-22   577.8       562.1       1874   ₹-29,337      -2.7%       -3.8%     
  TIINDIA     87     AUTO       03-Oct-22   2,688.0     2,553.0     373    ₹-50,339      -5.0%       -0.7%     
  TIMKEN      138    MNC        01-Aug-22   2,949.9     2,733.7     328    ₹-70,925      -7.3%       -1.5%     
  HUDCO       118    FIN SVC    01-Dec-22   47.2        40.5        22921  ₹-154,273     -14.2%      +2.2%     
  IRFC        136    FIN SVC    01-Dec-22   32.4        26.3        33416  ₹-202,621     -18.7%      +2.5%     
  RHIM        256    METAL      01-Dec-22   788.3       619.2       1373   ₹-232,216     -21.5% ⚠    +0.7%     
  TATAINVEST  145    FIN SVC    03-Oct-22   224.3       171.4       4479   ₹-237,189     -23.6% ⚠    -7.1%     
  AMBUJACEM   214    INFRA      01-Dec-22   570.9       368.3       1896   ₹-384,112     -35.5% ⚠    +1.3%     
  ⚠  WAZ < 0 (momentum below universe mean): TATAINVEST, AMBUJACEM, RHIM

  AFTER: Invested ₹20,887,654 | Cash ₹8,426 | Total ₹20,896,080 | Positions 19/20 | Slot ₹1,044,804

========================================================================
  REBALANCE #51  —  01 Jun 2023
  NAV: ₹24,394,640  |  Slot: ₹1,219,732  |  Cash: ₹8,426
========================================================================
  [SECTOR CAP≤4] dropped: CHOLAHLDNG

  EXITS (13)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TRITURBINE  —      ENERGY     03-Oct-22   250.8       384.3       4007   ₹534,737      +53.2%    241d  
  UCOBANK     50     PSU BNK    01-Dec-22   19.6        26.8        55233  ₹400,033      +36.9%    182d  
  BDL         —      DEFENCE    01-Aug-22   400.5       547.7       2418   ₹355,898      +36.7%    304d  
  BEL         47     DEFENCE    01-Aug-22   90.6        109.9       10694  ₹207,003      +21.4%    304d  
  RITES       96     INFRA      03-Oct-22   149.3       173.7       6733   ₹164,630      +16.4%    241d  
  HUDCO       —      FIN SVC    01-Dec-22   47.2        52.7        22921  ₹125,193      +11.6%    182d  
  TIMKEN      100    MNC        01-Aug-22   2,949.9     3,289.3     328    ₹111,298      +11.5%    304d  
  TIINDIA     67     AUTO       03-Oct-22   2,688.0     2,875.8     373    ₹70,041       +7.0%     241d  
  TATAINVEST  120    FIN SVC    03-Oct-22   224.3       215.8       4479   ₹-38,186      -3.8%     241d  
  GODFRYPHLP  201    FMCG       01-Dec-22   577.8       546.0       1874   ₹-59,448      -5.5%     182d  
  IRFC        85     FIN SVC    01-Dec-22   32.4        30.5        33416  ₹-64,902      -6.0%     182d  
  RHIM        —      METAL      01-Dec-22   788.3       645.9       1373   ₹-195,615     -18.1%    182d  
  AMBUJACEM   —      INFRA      01-Dec-22   570.9       421.8       1896   ₹-282,657     -26.1%    182d  

  ENTRIES (12)
  [52w filter blocked 1: GVT&D(-25.1%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ANURAS      2      FIN SVC    3.309    0.29   +60.8%    +72.0%    1,112.9     1096   ₹1,219,718    -3.3%     
  APARINDS    3      ENERGY     3.128    -0.02  +353.8%   +19.9%    2,657.1     459    ₹1,219,603    -1.1%     
  IDFCFIRSTB  5      PVT BNK    2.886    0.44   +111.7%   +33.7%    72.4        16855  ₹1,219,681    +8.0%     
  ENGINERSIN  6      INFRA      2.871    0.32   +94.5%    +52.0%    104.1       11717  ₹1,219,653    +7.8%     
  SONATSOFTW  7      IT         2.771    0.40   +110.1%   +39.8%    473.4       2576   ₹1,219,403    +6.6%     
  NCC         9      INFRA      2.693    0.57   +106.7%   +37.7%    119.2       10230  ₹1,219,729    +6.8%     
  RECLTD      10     FIN SVC    2.659    0.47   +77.0%    +24.4%    120.3       10135  ₹1,219,628    +5.1%     
  KPITTECH    12     IT         2.597    0.43   +140.3%   +32.0%    1,083.7     1125   ₹1,219,171    +13.3%    
  PFC         13     FIN SVC    2.554    0.70   +83.1%    +26.6%    128.6       9485   ₹1,219,708    +7.0%     
  CHOLAFIN    14     FIN SVC    2.534    0.11   +63.8%    +36.8%    1,039.4     1173   ₹1,219,171    +3.0%     
  CERA        15     CON DUR    2.459    0.18   +91.9%    +24.8%    7,452.5     163    ₹1,214,758    +4.6%     
  DLF         17     REALTY     2.430    0.38   +50.4%    +37.9%    467.4       2609   ₹1,219,327    +3.6%     

  HOLDS (6)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         31     DEFENCE    01-Apr-22   723.8       1,490.9     1352   ₹1,037,127    +106.0%     +1.9%     
  VBL         11     FMCG       01-Aug-22   183.1       335.0       5289   ₹803,491      +83.0%      +5.7%     
  RVNL        1      PSE        01-Dec-22   72.2        116.8       14989  ₹667,313      +61.6%      +2.4%     
  ITC         4      FMCG       01-Aug-22   254.0       377.4       3813   ₹470,658      +48.6%      +3.5%     
  TVSMOTOR    24     AUTO       01-Aug-22   911.5       1,257.5     1062   ₹367,423      +38.0%      +1.9%     
  CUMMINSIND  40     INFRA      01-Dec-22   1,372.2     1,683.2     789    ₹245,434      +22.7%      +4.5%     

  AFTER: Invested ₹24,269,896 | Cash ₹107,373 | Total ₹24,377,269 | Positions 18/20 | Slot ₹1,219,732

========================================================================
  REBALANCE #52  —  01 Aug 2023
  NAV: ₹27,960,371  |  Slot: ₹1,398,019  |  Cash: ₹107,373
========================================================================
  [SECTOR CAP≤4] dropped: LTF

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VBL         125    FMCG       01-Aug-22   183.1       317.7       5289   ₹711,586      +73.5%    365d  
  ITC         42     FMCG       01-Aug-22   254.0       399.0       3813   ₹552,808      +57.1%    365d  
  TVSMOTOR    69     AUTO       01-Aug-22   911.5       1,353.9     1062   ₹469,881      +48.5%    365d  
  SONATSOFTW  58     IT         01-Jun-23   473.4       506.3       2576   ₹84,719       +6.9%     61d   
  DLF         116    REALTY     01-Jun-23   467.4       493.3       2609   ₹67,569       +5.5%     61d   
  CERA        57     CON DUR    01-Jun-23   7,452.5     7,559.8     163    ₹17,493       +1.4%     61d   
  KPITTECH    75     IT         01-Jun-23   1,083.7     1,075.9     1125   ₹-8,753       -0.7%     61d   
  ANURAS      275    FIN SVC    01-Jun-23   1,112.9     957.7       1096   ₹-170,058     -13.9%    61d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MAZDOCK     1      DEFENCE    6.061    0.42   +596.7%   +137.6%   905.1       1544   ₹1,397,512    +8.4%     
  POLYCAB     4      MFG        2.916    0.52   +109.5%   +41.5%    4,561.7     306    ₹1,395,877    +7.3%     
  JYOTHYLAB   7      FMCG       2.806    0.47   +85.0%    +59.7%    294.5       4746   ₹1,397,807    +14.9%    
  FACT        8      ENERGY     2.679    1.04   +375.7%   +42.2%    493.8       2831   ₹1,398,006    +5.2%     
  HBLENGINE   9      MFG        2.539    0.42   +119.7%   +76.8%    199.6       7002   ₹1,397,872    +17.9%    
  CHOLAHLDNG  10     FIN SVC    2.489    0.63   +53.2%    +48.9%    963.3       1451   ₹1,397,780    +3.9%     
  ZFCVINDIA   12     AUTO       2.430    0.05   +65.4%    +33.7%    2,291.0     610    ₹1,397,480    +11.3%    

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         22     DEFENCE    01-Apr-22   723.8       1,870.9     1352   ₹1,550,845    +158.5%     +0.8%     
  RVNL        27     PSE        01-Dec-22   72.2        123.3       14989  ₹765,564      +70.7%      -0.1%     
  RECLTD      3      FIN SVC    01-Jun-23   120.3       175.8       10135  ₹562,151      +46.1%      +14.9%    
  PFC         2      FIN SVC    01-Jun-23   128.6       185.9       9485   ₹543,149      +44.5%      +10.0%    
  ENGINERSIN  5      INFRA      01-Jun-23   104.1       147.3       11717  ₹506,061      +41.5%      +13.1%    
  APARINDS    23     ENERGY     01-Jun-23   2,657.1     3,627.4     459    ₹445,386      +36.5%      +2.7%     
  CUMMINSIND  39     INFRA      01-Dec-22   1,372.2     1,865.6     789    ₹389,314      +36.0%      +1.2%     
  NCC         13     INFRA      01-Jun-23   119.2       150.3       10230  ₹317,824      +26.1%      +11.5%    
  IDFCFIRSTB  6      PVT BNK    01-Jun-23   72.4        88.2        16855  ₹267,120      +21.9%      +6.9%     
  CHOLAFIN    38     FIN SVC    01-Jun-23   1,039.4     1,126.2     1173   ₹101,889      +8.4%       -0.7%     

  AFTER: Invested ₹26,912,749 | Cash ₹1,036,007 | Total ₹27,948,755 | Positions 17/20 | Slot ₹1,398,019

========================================================================
  REBALANCE #53  —  03 Oct 2023
  NAV: ₹32,624,121  |  Slot: ₹1,631,206  |  Cash: ₹1,036,007
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HAL         112    DEFENCE    01-Apr-22   723.8       1,901.3     1352   ₹1,591,947    +162.7%   550d  
  RVNL        10     PSE        01-Dec-22   72.2        170.4       14989  ₹1,471,437    +135.9%   306d  
  IDFCFIRSTB  47     PVT BNK    01-Jun-23   72.4        93.8        16855  ₹362,040      +29.7%    124d  
  CHOLAFIN    79     FIN SVC    01-Jun-23   1,039.4     1,249.1     1173   ₹246,038      +20.2%    124d  
  CUMMINSIND  219    INFRA      01-Dec-22   1,372.2     1,624.9     789    ₹199,446      +18.4%    306d  
  ZFCVINDIA   43     AUTO       01-Aug-23   2,291.0     2,566.3     610    ₹167,991      +12.0%    63d   
  FACT        29     ENERGY     01-Aug-23   493.8       531.5       2831   ₹106,561      +7.6%     63d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IRFC        2      FIN SVC    4.507    0.98   +281.0%   +135.9%   73.2        22282  ₹1,631,134    +5.8%     
  GVT&D       5      ENERGY     3.429    0.73   +234.5%   +92.5%    424.5       3842   ₹1,630,988    +9.7%     
  MAPMYINDIA  9      IT         3.138    0.16   +62.8%    +78.5%    2,109.7     773    ₹1,630,772    +17.0%    
  JINDALSAW   11     METAL      2.985    0.96   +338.0%   +35.3%    177.9       9170   ₹1,631,204    +3.9%     
  UCOBANK     12     PSU BNK    2.900    0.73   +285.8%   +59.3%    43.2        37746  ₹1,631,175    +12.1%    
  IOB         13     PSU BNK    2.893    0.76   +183.0%   +94.0%    48.4        33702  ₹1,631,177    +18.3%    
  CENTRALBK   14     PSU BNK    2.845    1.06   +169.9%   +75.6%    50.6        32236  ₹1,631,177    +13.9%    
  MAHABANK    15     PSU BNK    2.817    0.97   +200.3%   +65.3%    45.6        35795  ₹1,631,185    +10.1%    

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RECLTD      1      FIN SVC    01-Jun-23   120.3       260.0       10135  ₹1,415,229    +116.0%     +11.0%    
  APARINDS    7      ENERGY     01-Jun-23   2,657.1     5,430.4     459    ₹1,272,939    +104.4%     +4.6%     
  PFC         4      FIN SVC    01-Jun-23   128.6       225.2       9485   ₹916,611      +75.2%      +7.4%     
  HBLENGINE   11     MFG        01-Aug-23   199.6       279.9       7002   ₹561,761      +40.2%      +8.0%     
  ENGINERSIN  36     INFRA      01-Jun-23   104.1       138.6       11717  ₹404,585      +33.2%      -2.1%     
  NCC         33     INFRA      01-Jun-23   119.2       157.9       10230  ₹395,790      +32.4%      +3.8%     
  JYOTHYLAB   12     FMCG       01-Aug-23   294.5       355.9       4746   ₹291,153      +20.8%      +3.6%     
  MAZDOCK     3      DEFENCE    01-Aug-23   905.1       1,068.2     1544   ₹251,847      +18.0%      +2.9%     
  CHOLAHLDNG  38     FIN SVC    01-Aug-23   963.3       1,132.4     1451   ₹245,285      +17.5%      +1.4%     
  POLYCAB     8      MFG        01-Aug-23   4,561.7     5,293.7     306    ₹224,009      +16.0%      +3.5%     

  AFTER: Invested ₹32,113,189 | Cash ₹495,437 | Total ₹32,608,627 | Positions 18/20 | Slot ₹1,631,206

========================================================================
  REBALANCE #54  —  01 Dec 2023
  NAV: ₹34,817,634  |  Slot: ₹1,740,882  |  Cash: ₹495,437
========================================================================

  EXITS (12)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  RECLTD      2      FIN SVC    01-Jun-23   120.3       336.8       10135  ₹2,194,153    +179.9%   183d  
  PFC         1      FIN SVC    01-Jun-23   128.6       332.4       9485   ₹1,933,018    +158.5%   183d  
  HBLENGINE   16     MFG        01-Aug-23   199.6       382.0       7002   ₹1,277,237    +91.4%    122d  
  ENGINERSIN  131    INFRA      01-Jun-23   104.1       144.4       11717  ₹472,704      +38.8%    183d  
  NCC         81     INFRA      01-Jun-23   119.2       162.7       10230  ₹444,578      +36.4%    183d  
  MAZDOCK     80     DEFENCE    01-Aug-23   905.1       990.7       1544   ₹132,076      +9.5%     122d  
  CHOLAHLDNG  78     FIN SVC    01-Aug-23   963.3       1,027.2     1451   ₹92,726       +6.6%     122d  
  IRFC        27     FIN SVC    03-Oct-23   73.2        72.8        22282  ₹-8,148       -0.5%     59d   
  MAHABANK    127    PSU BNK    03-Oct-23   45.6        40.8        35795  ₹-169,709     -10.4%    59d   
  UCOBANK     109    PSU BNK    03-Oct-23   43.2        36.7        37746  ₹-246,043     -15.1%    59d   
  CENTRALBK   82     PSU BNK    03-Oct-23   50.6        42.7        32236  ₹-254,774     -15.6%    59d   
  IOB         97     PSU BNK    03-Oct-23   48.4        39.6        33702  ₹-296,578     -18.2%    59d   

  ENTRIES (13)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MCX         2      FIN SVC    3.920    0.68   +105.5%   +96.4%    621.8       2799   ₹1,740,510    +10.4%    
  COALINDIA   3      ENERGY     3.740    0.74   +65.4%    +57.7%    301.3       5777   ₹1,740,662    +6.1%     
  TVSMOTOR    4      AUTO       3.183    0.63   +82.3%    +38.3%    1,887.8     922    ₹1,740,574    +9.7%     
  RATNAMANI   5      METAL      3.159    0.50   +97.2%    +48.1%    3,785.4     459    ₹1,737,500    +12.2%    
  TRENT       6      CONSUMP    3.051    0.67   +96.6%    +37.0%    2,800.7     621    ₹1,739,209    +10.1%    
  PCBL        7      MFG        2.957    1.19   +105.6%   +56.4%    253.5       6868   ₹1,740,810    +12.0%    
  ANGELONE    8      FIN SVC    2.945    0.89   +110.5%   +73.1%    294.2       5916   ₹1,740,644    +6.6%     
  BAJAJ-AUTO  9      AUTO       2.937    0.49   +72.1%    +29.6%    5,767.9     301    ₹1,736,137    +5.8%     
  SONATSOFTW  10     IT         2.888    0.68   +161.7%   +35.1%    665.6       2615   ₹1,740,670    +6.4%     
  KALYANKJIL  11     CON DUR    2.885    1.11   +239.4%   +47.0%    333.1       5225   ₹1,740,585    +4.4%     
  NTPC        12     ENERGY     2.705    0.72   +66.7%    +22.8%    253.9       6856   ₹1,740,823    +7.3%     
  ECLERX      13     IT         2.688    0.80   +84.0%    +61.5%    1,315.7     1323   ₹1,740,662    +9.8%     
  BRIGADE     14     REALTY     2.660    0.71   +79.1%    +40.6%    626.3       2779   ₹1,740,528    +14.3%    

  HOLDS (6)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  APARINDS    24     ENERGY     01-Jun-23   2,657.1     5,464.9     459    ₹1,288,767    +105.7%     +1.0%     
  JYOTHYLAB   21     FMCG       01-Aug-23   294.5       428.7       4746   ₹636,595      +45.5%      +4.7%     
  JINDALSAW   4      METAL      03-Oct-23   177.9       229.4       9170   ₹472,379      +29.0%      +2.8%     
  POLYCAB     54     MFG        01-Aug-23   4,561.7     5,157.1     306    ₹182,195      +13.1%      +0.5%     
  MAPMYINDIA  41     IT         03-Oct-23   2,109.7     2,200.0     773    ₹69,791       +4.3%       +2.4%     
  GVT&D       29     ENERGY     03-Oct-23   424.5       416.0       3842   ₹-32,750      -2.0%       +3.0%     

  AFTER: Invested ₹34,142,541 | Cash ₹648,235 | Total ₹34,790,776 | Positions 19/20 | Slot ₹1,740,882

========================================================================
  REBALANCE #55  —  01 Feb 2024
  NAV: ₹38,648,091  |  Slot: ₹1,932,405  |  Cash: ₹648,235
========================================================================
  [SECTOR CAP≤4] dropped: NHPC, SJVN, BHEL

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  APARINDS    54     ENERGY     01-Jun-23   2,657.1     6,226.6     459    ₹1,638,401    +134.3%   245d  
  JYOTHYLAB   38     FMCG       01-Aug-23   294.5       490.6       4746   ₹930,497      +66.6%    184d  
  SONATSOFTW  53     IT         01-Dec-23   665.6       743.7       2615   ₹204,033      +11.7%    62d   
  MCX         41     FIN SVC    01-Dec-23   621.8       686.5       2799   ₹180,893      +10.4%    62d   
  ANGELONE    78     FIN SVC    01-Dec-23   294.2       324.7       5916   ₹180,088      +10.3%    62d   
  ECLERX      85     IT         01-Dec-23   1,315.7     1,380.3     1323   ₹85,481       +4.9%     62d   
  KALYANKJIL  102    CON DUR    01-Dec-23   333.1       332.3       5225   ₹-4,158       -0.2%     62d   
  POLYCAB     252    MFG        01-Aug-23   4,561.7     4,200.9     306    ₹-110,414     -7.9%     184d  
  RATNAMANI   117    METAL      01-Dec-23   3,785.4     3,317.8     459    ₹-214,637     -12.4%    62d   
  MAPMYINDIA  232    IT         03-Oct-23   2,109.7     1,828.7     773    ₹-217,192     -13.3%    121d  

  ENTRIES (10)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IRFC        1      FIN SVC    4.104    0.44   +423.4%   +136.1%   164.1       11773  ₹1,932,279    +18.3%    
  RECLTD      2      FIN SVC    4.006    0.49   +330.7%   +79.5%    445.1       4341   ₹1,932,310    +8.7%     
  PFC         4      FIN SVC    3.865    0.48   +293.7%   +89.2%    405.8       4762   ₹1,932,228    +7.5%     
  HBLENGINE   5      MFG        3.618    0.09   +440.7%   +82.4%    527.6       3662   ₹1,932,189    +10.7%    
  NBCC        6      INFRA      3.414    0.30   +276.7%   +111.1%   92.9        20802  ₹1,932,399    +34.6%    
  NLCINDIA    8      ENERGY     3.159    0.18   +232.9%   +92.2%    250.8       7705   ₹1,932,381    +9.9%     
  HAL         10     DEFENCE    2.929    0.35   +141.5%   +63.4%    2,912.1     663    ₹1,930,741    +2.0%     
  MEDANTA     12     HEALTH     2.880    -0.07  +165.3%   +56.8%    1,198.0     1613   ₹1,932,324    +11.7%    
  RVNL        19     PSE        2.655    0.97   +297.9%   +90.3%    293.9       6574   ₹1,932,311    +16.8%    
  ENGINERSIN  21     INFRA      2.516    0.33   +178.1%   +86.8%    225.8       8558   ₹1,932,221    +8.9%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GVT&D       3      ENERGY     03-Oct-23   424.5       696.8       3842   ₹1,046,276    +64.1%      +12.5%    
  JINDALSAW   29     METAL      03-Oct-23   177.9       249.8       9170   ₹659,632      +40.4%      +3.7%     
  BAJAJ-AUTO  9      AUTO       01-Dec-23   5,767.9     7,303.5     301    ₹462,229      +26.6%      +5.9%     
  PCBL        15     MFG        01-Dec-23   253.5       307.8       6868   ₹372,853      +21.4%      +8.8%     
  BRIGADE     18     REALTY     01-Dec-23   626.3       754.0       2779   ₹354,727      +20.4%      +5.2%     
  NTPC        20     ENERGY     01-Dec-23   253.9       304.0       6856   ₹343,374      +19.7%      +3.3%     
  COALINDIA   28     ENERGY     01-Dec-23   301.3       353.5       5777   ₹301,534      +17.3%      +4.9%     
  TRENT       7      CONSUMP    01-Dec-23   2,800.7     3,095.0     621    ₹182,798      +10.5%      -0.6%     
  TVSMOTOR    31     AUTO       01-Dec-23   1,887.8     1,972.3     922    ₹77,878       +4.5%       +0.1%     

  AFTER: Invested ₹38,563,618 | Cash ₹61,531 | Total ₹38,625,149 | Positions 19/20 | Slot ₹1,932,405

========================================================================
  REBALANCE #56  —  01 Apr 2024
  NAV: ₹38,913,901  |  Slot: ₹1,945,695  |  Cash: ₹61,531
========================================================================

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  JINDALSAW   81     METAL      03-Oct-23   177.9       230.5       9170   ₹482,434      +29.6%    181d  
  NTPC        58     ENERGY     01-Dec-23   253.9       326.4       6856   ₹496,730      +28.5%    122d  
  BRIGADE     99     REALTY     01-Dec-23   626.3       714.4       2779   ₹244,926      +14.1%    122d  
  TVSMOTOR    63     AUTO       01-Dec-23   1,887.8     2,122.8     922    ₹216,690      +12.4%    122d  
  PCBL        92     MFG        01-Dec-23   253.5       266.7       6868   ₹90,765       +5.2%     122d  
  PFC         40     FIN SVC    01-Feb-24   405.8       371.1       4762   ₹-165,174     -8.5%     60d   
  ENGINERSIN  65     INFRA      01-Feb-24   225.8       202.6       8558   ₹-197,966     -10.2%    60d   
  NLCINDIA    130    ENERGY     01-Feb-24   250.8       224.1       7705   ₹-205,992     -10.7%    60d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SCHNEIDER   2      ENERGY     4.326    0.23   +404.4%   +88.0%    778.1       2500   ₹1,945,250    +18.9%    
  ANANDRATHI  3      FIN SVC    4.100    -0.07  +353.1%   +40.8%    894.4       2175   ₹1,945,301    -1.1%     
  ACE         4      MFG        3.558    0.74   +345.1%   +91.1%    1,600.4     1215   ₹1,944,540    +20.2%    
  SUNPHARMA   7      HEALTH     2.861    -0.01  +71.1%    +30.8%    1,598.7     1217   ₹1,945,604    +3.3%     
  CUMMINSIND  8      INFRA      2.851    0.00   +84.2%    +52.5%    2,927.3     664    ₹1,943,748    +6.4%     
  BOSCHLTD    9      AUTO       2.841    0.30   +70.7%    +38.6%    29,732.3    65     ₹1,932,599    +2.7%     
  TORNTPOWER  10     ENERGY     2.738    0.05   +172.7%   +59.9%    1,379.4     1410   ₹1,944,887    +13.6%    

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GVT&D       1      ENERGY     03-Oct-23   424.5       873.3       3842   ₹1,724,067    +105.7%     +1.1%     
  BAJAJ-AUTO  5      AUTO       01-Dec-23   5,767.9     8,626.2     301    ₹860,337      +49.6%      +4.3%     
  TRENT       12     CONSUMP    01-Dec-23   2,800.7     3,877.1     621    ₹668,441      +38.4%      -0.9%     
  COALINDIA   37     ENERGY     01-Dec-23   301.3       388.7       5777   ₹504,610      +29.0%      +1.8%     
  HAL         22     DEFENCE    01-Feb-24   2,912.1     3,330.6     663    ₹277,455      +14.4%      +6.8%     
  MEDANTA     11     HEALTH     01-Feb-24   1,198.0     1,330.8     1613   ₹214,210      +11.1%      +4.8%     
  RECLTD      15     FIN SVC    01-Feb-24   445.1       419.0       4341   ₹-113,384     -5.9%       +2.2%     
  NBCC        17     INFRA      01-Feb-24   92.9        81.8        20802  ₹-229,869     -11.9%      +4.1%     
  RVNL        21     PSE        01-Feb-24   293.9       258.6       6574   ₹-232,487     -12.0%      +4.6%     
  HBLENGINE   33     MFG        01-Feb-24   527.6       456.6       3662   ₹-260,274     -13.5%      -1.8%     
  IRFC        6      FIN SVC    01-Feb-24   164.1       139.9       11773  ₹-285,463     -14.8%      +2.1%     

  AFTER: Invested ₹37,101,118 | Cash ₹1,796,632 | Total ₹38,897,750 | Positions 18/20 | Slot ₹1,945,695

========================================================================
  REBALANCE #57  —  03 Jun 2024
  NAV: ₹45,641,734  |  Slot: ₹2,282,087  |  Cash: ₹1,796,632
========================================================================
  [SECTOR CAP≤4] dropped: HUDCO

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJAJ-AUTO  59     AUTO       01-Dec-23   5,767.9     8,906.0     301    ₹944,559      +54.4%    185d  
  COALINDIA   65     ENERGY     01-Dec-23   301.3       450.5       5777   ₹861,679      +49.5%    185d  
  TORNTPOWER  —      ENERGY     01-Apr-24   1,379.4     1,469.3     1410   ₹126,811      +6.5%     63d   
  NBCC        87     INFRA      01-Feb-24   92.9        98.8        20802  ₹122,779      +6.4%     123d  
  BOSCHLTD    116    AUTO       01-Apr-24   29,732.3    29,444.7    65     ₹-18,696      -1.0%     63d   
  MEDANTA     233    HEALTH     01-Feb-24   1,198.0     1,165.8     1613   ₹-51,839      -2.7%     123d  
  HBLENGINE   57     MFG        01-Feb-24   527.6       501.4       3662   ₹-96,053      -5.0%     123d  
  ACE         76     MFG        01-Apr-24   1,600.4     1,444.7     1215   ₹-189,200     -9.7%     63d   
  SUNPHARMA   201    HEALTH     01-Apr-24   1,598.7     1,425.8     1217   ₹-210,413     -10.8%    63d   

  ENTRIES (8)
  [52w filter blocked 1: JAIBALAJI(-31.4%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COCHINSHIP  1      DEFENCE    4.734    1.05   +738.3%   +134.3%   1,987.1     1148   ₹2,281,197    +20.1%    
  JWL         5      INFRA      3.307    0.61   +457.2%   +72.3%    643.9       3543   ₹2,281,489    +22.8%    
  BEL         6      DEFENCE    3.285    0.75   +198.9%   +56.6%    314.0       7268   ₹2,281,939    +17.3%    
  SIEMENS     7      ENERGY     3.260    0.37   +114.1%   +59.3%    4,254.8     536    ₹2,280,599    +6.5%     
  POWERINDIA  9      ENERGY     2.992    0.51   +177.6%   +88.8%    11,094.8    205    ₹2,274,440    +8.2%     
  DIXON       11     CON DUR    2.746    0.47   +202.0%   +42.3%    9,878.1     231    ₹2,281,834    +10.6%    
  PFC         12     FIN SVC    2.744    1.06   +341.1%   +35.9%    513.1       4447   ₹2,281,934    +16.4%    
  TITAGARH    13     INFRA      2.707    0.75   +341.5%   +51.9%    1,489.9     1531   ₹2,281,059    +20.1%    

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GVT&D       2      ENERGY     03-Oct-23   424.5       1,414.4     3842   ₹3,803,020    +233.2%     +13.3%    
  HAL         4      DEFENCE    01-Feb-24   2,912.1     5,160.9     663    ₹1,490,962    +77.2%      +13.2%    
  TRENT       22     CONSUMP    01-Dec-23   2,800.7     4,652.6     621    ₹1,150,077    +66.1%      +2.2%     
  RVNL        32     PSE        01-Feb-24   293.9       399.6       6574   ₹694,866      +36.0%      +19.5%    
  RECLTD      10     FIN SVC    01-Feb-24   445.1       550.0       4341   ₹455,141      +23.6%      +11.6%    
  CUMMINSIND  21     INFRA      01-Apr-24   2,927.3     3,618.2     664    ₹458,756      +23.6%      +2.9%     
  ANANDRATHI  8      FIN SVC    01-Apr-24   894.4       1,013.3     2175   ₹258,522      +13.3%      +1.3%     
  IRFC        20     FIN SVC    01-Feb-24   164.1       182.5       11773  ₹216,657      +11.2%      +10.4%    
  SCHNEIDER   39     ENERGY     01-Apr-24   778.1       719.7       2500   ₹-146,000     -7.5%       -8.9%     

  AFTER: Invested ₹43,558,628 | Cash ₹2,061,443 | Total ₹45,620,071 | Positions 17/20 | Slot ₹2,282,087

========================================================================
  REBALANCE #58  —  01 Aug 2024
  NAV: ₹49,493,978  |  Slot: ₹2,474,699  |  Cash: ₹2,061,443
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HAL         57     DEFENCE    01-Feb-24   2,912.1     4,710.4     663    ₹1,192,273    +61.8%    182d  
  RECLTD      22     FIN SVC    01-Feb-24   445.1       574.3       4341   ₹560,645      +29.0%    182d  
  COCHINSHIP  1      DEFENCE    03-Jun-24   1,987.1     2,547.1     1148   ₹642,883      +28.2%    59d   
  CUMMINSIND  76     INFRA      01-Apr-24   2,927.3     3,738.1     664    ₹538,328      +27.7%    122d  
  TITAGARH    53     INFRA      03-Jun-24   1,489.9     1,566.2     1531   ₹116,842      +5.1%     59d   
  SCHNEIDER   146    ENERGY     01-Apr-24   778.1       803.7       2500   ₹64,000       +3.3%     122d  
  PFC         63     FIN SVC    03-Jun-24   513.1       504.5       4447   ₹-38,469      -1.7%     59d   
  BEL         26     DEFENCE    03-Jun-24   314.0       306.6       7268   ₹-53,710      -2.4%     59d   
  SIEMENS     69     ENERGY     03-Jun-24   4,254.8     4,111.4     536    ₹-76,877      -3.4%     59d   
  JWL         38     INFRA      03-Jun-24   643.9       596.7       3543   ₹-167,269     -7.3%     59d   

  ENTRIES (10)
  [52w filter blocked 1: JAIBALAJI(-29.3%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  GRSE        3      DEFENCE    3.754    1.17   +291.7%   +137.6%   2,318.3     1067   ₹2,473,617    +0.0%     
  EMAMILTD    6      FMCG       3.059    0.35   +101.3%   +69.3%    792.6       3122   ₹2,474,489    +5.5%     
  MOTHERSON   7      AUTO       2.810    0.53   +104.4%   +49.6%    129.1       19170  ₹2,474,662    +1.1%     
  OFSS        10     IT         2.581    0.22   +186.5%   +48.1%    10,120.6    244    ₹2,469,437    +1.7%     
  KALYANKJIL  11     CON DUR    2.556    0.13   +231.5%   +36.4%    561.8       4405   ₹2,474,699    +5.2%     
  ZYDUSLIFE   12     HEALTH     2.448    0.55   +103.4%   +30.5%    1,227.3     2016   ₹2,474,318    +5.2%     
  ARE&M       13     AUTO       2.381    1.06   +167.6%   +46.4%    1,580.3     1565   ₹2,473,174    +0.7%     
  PERSISTENT  14     IT         2.367    0.30   +91.3%    +42.7%    4,750.7     520    ₹2,470,372    +2.8%     
  KAYNES      15     MFG        2.269    1.10   +140.9%   +64.3%    4,378.2     565    ₹2,473,683    +4.8%     
  TVSMOTOR    16     AUTO       2.221    0.38   +92.9%    +25.4%    2,564.6     964    ₹2,472,240    +5.0%     

  HOLDS (7)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GVT&D       2      ENERGY     03-Oct-23   424.5       1,659.3     3842   ₹4,743,960    +290.9%     +5.7%     
  TRENT       6      CONSUMP    01-Dec-23   2,800.7     5,759.0     621    ₹1,837,105    +105.6%     +5.3%     
  RVNL        5      PSE        01-Feb-24   293.9       588.3       6574   ₹1,934,909    +100.1%     +5.1%     
  DIXON       10     CON DUR    03-Jun-24   9,878.1     11,661.2    231    ₹411,900      +18.1%      -0.2%     
  IRFC        11     FIN SVC    01-Feb-24   164.1       183.3       11773  ₹226,096      +11.7%      -1.7%     
  POWERINDIA  39     ENERGY     03-Jun-24   11,094.8    12,384.2    205    ₹264,321      +11.6%      +3.3%     
  ANANDRATHI  36     FIN SVC    01-Apr-24   894.4       921.5       2175   ₹58,970       +3.0%       -2.8%     

  AFTER: Invested ₹47,944,315 | Cash ₹1,520,298 | Total ₹49,464,613 | Positions 17/20 | Slot ₹2,474,699

========================================================================
  REBALANCE #59  —  01 Oct 2024
  NAV: ₹51,768,384  |  Slot: ₹2,588,419  |  Cash: ₹1,520,298
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DIXON       18     CON DUR    03-Jun-24   9,878.1     14,189.5    231    ₹995,937      +43.6%    120d  
  MOTHERSON   65     AUTO       01-Aug-24   129.1       139.2       19170  ₹193,979      +7.8%     61d   
  ANANDRATHI  44     FIN SVC    01-Apr-24   894.4       961.4       2175   ₹145,769      +7.5%     183d  
  OFSS        39     IT         01-Aug-24   10,120.6    10,613.9    244    ₹120,350      +4.9%     61d   
  IRFC        198    FIN SVC    01-Feb-24   164.1       150.6       11773  ₹-158,714     -8.2%     243d  
  EMAMILTD    186    FMCG       01-Aug-24   792.6       724.8       3122   ₹-211,537     -8.5%     61d   
  ARE&M       177    AUTO       01-Aug-24   1,580.3     1,390.4     1565   ₹-297,154     -12.0%    61d   
  ZYDUSLIFE   115    HEALTH     01-Aug-24   1,227.3     1,068.1     2016   ₹-321,000     -13.0%    61d   
  GRSE        268    DEFENCE    01-Aug-24   2,318.3     1,686.2     1067   ₹-674,415     -27.3%    61d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  PCBL        1      MFG        4.837    0.67   +253.9%   +123.6%   553.3       4677   ₹2,587,878    +10.0%    
  BAJAJ-AUTO  4      AUTO       3.491    0.48   +136.5%   +29.3%    11,692.4    221    ₹2,584,020    +2.8%     
  HSCL        5      MFG        3.455    0.63   +182.0%   +66.4%    665.9       3887   ₹2,588,170    +8.5%     
  COLPAL      6      FMCG       3.229    0.12   +95.6%    +33.1%    3,666.2     706    ₹2,588,328    +3.9%     
  BASF        7      MFG        3.163    1.00   +196.8%   +54.8%    7,884.8     328    ₹2,586,219    +13.8%    
  SUNPHARMA   8      HEALTH     2.969    0.29   +68.0%    +26.4%    1,889.9     1369   ₹2,587,287    +2.9%     
  LUPIN       9      HEALTH     2.908    0.38   +91.7%    +35.0%    2,180.8     1186   ₹2,586,471    -0.1%     
  GODFRYPHLP  10     FMCG       2.870    0.72   +236.7%   +63.1%    2,234.8     1158   ₹2,587,877    -1.9%     

  HOLDS (8)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GVT&D       12     ENERGY     03-Oct-23   424.5       1,682.9     3842   ₹4,834,586    +296.4%     +1.9%     
  TRENT       2      CONSUMP    01-Dec-23   2,800.7     7,597.1     621    ₹2,978,579    +171.3%     +2.9%     
  RVNL        34     PSE        01-Feb-24   293.9       520.3       6574   ₹1,488,390    +77.0%      -2.8%     
  KALYANKJIL  3      CON DUR    01-Aug-24   561.8       748.0       4405   ₹820,071      +33.1%      +6.8%     
  POWERINDIA  28     ENERGY     03-Jun-24   11,094.8    14,274.8    205    ₹651,886      +28.7%      +9.7%     
  KAYNES      27     MFG        01-Aug-24   4,378.2     5,466.4     565    ₹614,833      +24.9%      +3.2%     
  PERSISTENT  35     IT         01-Aug-24   4,750.7     5,435.4     520    ₹356,048      +14.4%      +3.4%     
  TVSMOTOR    23     AUTO       01-Aug-24   2,564.6     2,817.1     964    ₹243,426      +9.8%       +0.7%     

  AFTER: Invested ₹50,152,010 | Cash ₹1,591,799 | Total ₹51,743,809 | Positions 16/20 | Slot ₹2,588,419

========================================================================
  REBALANCE #60  —  02 Dec 2024
  NAV: ₹47,020,147  |  Slot: ₹2,351,007  |  Cash: ₹1,591,799
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  RVNL        117    PSE        01-Feb-24   293.9       433.4       6574   ₹916,969      +47.5%    305d  
  KAYNES      11     MFG        01-Aug-24   4,378.2     6,319.0     565    ₹1,096,580    +44.3%    123d  
  POWERINDIA  33     ENERGY     03-Jun-24   11,094.8    12,260.7    205    ₹239,013      +10.5%    182d  
  TVSMOTOR    181    AUTO       01-Aug-24   2,564.6     2,474.5     964    ₹-86,866      -3.5%     123d  
  LUPIN       86     HEALTH     01-Oct-24   2,180.8     2,056.8     1186   ₹-147,157     -5.7%     62d   
  HSCL        57     MFG        01-Oct-24   665.9       532.9       3887   ₹-516,781     -20.0%    62d   
  PCBL        133    MFG        01-Oct-24   553.3       422.5       4677   ₹-611,857     -23.6%    62d   
  COLPAL      298    FMCG       01-Oct-24   3,666.2     2,792.9     706    ₹-616,512     -23.8%    62d   
  BAJAJ-AUTO  176    AUTO       01-Oct-24   11,692.4    8,781.1     221    ₹-643,399     -24.9%    62d   
  BASF        146    MFG        01-Oct-24   7,884.8     5,636.0     328    ₹-737,610     -28.5%    62d   

  ENTRIES (10)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BSE         2      FIN SVC    3.079    1.09   +113.5%   +61.1%    1,516.5     1550   ₹2,350,576    +0.2%     
  OFSS        3      IT         3.029    0.64   +204.6%   +11.6%    11,378.1    206    ₹2,343,880    +5.8%     
  COFORGE     4      IT         3.009    0.46   +56.8%    +37.7%    1,722.2     1365   ₹2,350,787    +5.8%     
  NETWEB      5      IT         3.003    0.75   +240.2%   +2.5%     2,784.8     844    ₹2,350,393    +0.8%     
  INDHOTEL    6      CONSUMP    2.986    0.96   +90.9%    +23.7%    795.2       2956   ₹2,350,493    +6.1%     
  FORTIS      7      HEALTH     2.943    0.44   +82.2%    +22.7%    676.2       3477   ₹2,350,975    +4.4%     
  RADICO      9      FMCG       2.677    0.52   +70.4%    +24.6%    2,415.9     973    ₹2,350,716    +3.6%     
  SIEMENS     10     ENERGY     2.581    1.16   +111.6%   +9.7%     4,424.0     531    ₹2,349,151    +5.8%     
  VIJAYA      11     HEALTH     2.546    0.26   +85.3%    +26.0%    1,147.7     2048   ₹2,350,568    +5.0%     
  CONCORDBIO  12     HEALTH     2.531    0.50   +72.1%    +31.1%    2,170.4     1083   ₹2,350,553    +9.8%     

  HOLDS (6)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GVT&D       2      ENERGY     03-Oct-23   424.5       1,765.4     3842   ₹5,151,738    +315.9%     -1.6%     
  TRENT       19     CONSUMP    01-Dec-23   2,800.7     6,791.3     621    ₹2,478,211    +142.5%     +0.3%     
  KALYANKJIL  12     CON DUR    01-Aug-24   561.8       720.0       4405   ₹696,871      +28.2%      +3.3%     
  PERSISTENT  20     IT         01-Aug-24   4,750.7     5,875.8     520    ₹585,061      +23.7%      +3.1%     
  SUNPHARMA   51     HEALTH     01-Oct-24   1,889.9     1,780.3     1369   ₹-150,120     -5.8%       +0.8%     
  GODFRYPHLP  52     FMCG       01-Oct-24   2,234.8     1,898.0     1158   ₹-389,984     -15.1%      -5.1%     

  AFTER: Invested ₹45,360,301 | Cash ₹1,631,945 | Total ₹46,992,246 | Positions 16/20 | Slot ₹2,351,007

========================================================================
  REBALANCE #61  —  01 Feb 2025
  NAV: ₹43,090,335  |  Slot: ₹2,154,517  |  Cash: ₹1,631,945
========================================================================
  [SECTOR CAP≤4] dropped: KFINTECH

  [REGIME OFF] Nifty 500 21,580.9 < EMA200 22,008.1 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GVT&D       2      ENERGY     03-Oct-23   424.5       1,767.9     3842   ₹5,161,134    +316.4%     -2.6%     
  TRENT       30     CONSUMP    01-Dec-23   2,800.7     6,176.8     621    ₹2,096,584    +120.5%     +2.9%     
  PERSISTENT  24     IT         01-Aug-24   4,750.7     5,896.4     520    ₹595,779      +24.1%      -2.5%     
  BSE         5      FIN SVC    02-Dec-24   1,516.5     1,795.7     1550   ₹432,739      +18.4%      -1.5%     
  INDHOTEL    7      CONSUMP    02-Dec-24   795.2       795.6       2956   ₹1,174        +0.0%       +1.3%     
  RADICO      42     FMCG       02-Dec-24   2,415.9     2,376.8     973    ₹-38,135      -1.6%       +4.9%     
  CONCORDBIO  21     HEALTH     02-Dec-24   2,170.4     2,086.5     1083   ₹-90,826      -3.9%       -1.4%     
  VIJAYA      15     HEALTH     02-Dec-24   1,147.7     1,087.5     2048   ₹-123,348     -5.2%       +3.5%     
  COFORGE     34     IT         02-Dec-24   1,722.2     1,600.2     1365   ₹-166,449     -7.1%       -7.2%     
  FORTIS      39     HEALTH     02-Dec-24   676.2       626.1       3477   ₹-173,992     -7.4%       -2.9%     
  SUNPHARMA   73     HEALTH     01-Oct-24   1,889.9     1,714.9     1369   ₹-239,532     -9.3%       -1.9%     
  KALYANKJIL  179    CON DUR    01-Aug-24   561.8       504.0       4405   ₹-254,688     -10.3% ⚠    -5.2%     
  SIEMENS     147    ENERGY     02-Dec-24   4,424.0     3,374.2     531    ₹-557,440     -23.7%      -4.9%     
  GODFRYPHLP  56     FMCG       01-Oct-24   2,234.8     1,643.3     1158   ₹-684,919     -26.5%      +7.9%     
  OFSS        158    IT         02-Dec-24   11,378.1    8,249.1     206    ₹-644,560     -27.5%      -11.8%    
  NETWEB      286    IT         02-Dec-24   2,784.8     1,785.2     844    ₹-843,653     -35.9% ⚠    -13.5%    
  ⚠  WAZ < 0 (momentum below universe mean): KALYANKJIL, NETWEB

  AFTER: Invested ₹41,458,390 | Cash ₹1,631,945 | Total ₹43,090,335 | Positions 16/20 | Slot ₹2,154,517

========================================================================
  REBALANCE #62  —  01 Apr 2025
  NAV: ₹40,556,228  |  Slot: ₹2,027,811  |  Cash: ₹1,631,945
========================================================================
  [SECTOR CAP≤4] dropped: MUTHOOTFIN, BAJAJFINSV, CHOLAHLDNG, CHOLAFIN

  [REGIME OFF] Nifty 500 21,070.8 < EMA200 21,620.7 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GVT&D       119    ENERGY     03-Oct-23   424.5       1,478.1     3842   ₹4,048,033    +248.2%     -2.5%     
  TRENT       138    CONSUMP    01-Dec-23   2,800.7     5,565.3     621    ₹1,716,848    +98.7% ⚠    +6.8%     
  BSE         21     FIN SVC    02-Dec-24   1,516.5     1,816.3     1550   ₹464,645      +19.8%      +15.8%    
  PERSISTENT  145    IT         01-Aug-24   4,750.7     5,179.0     520    ₹222,687      +9.0% ⚠     -3.6%     
  GODFRYPHLP  7      FMCG       01-Oct-24   2,234.8     2,343.2     1158   ₹125,593      +4.9%       +17.0%    
  FORTIS      27     HEALTH     02-Dec-24   676.2       687.8       3477   ₹40,633       +1.7%       +7.1%     
  INDHOTEL    72     CONSUMP    02-Dec-24   795.2       799.8       2956   ₹13,791       +0.6%       +2.7%     
  RADICO      87     FMCG       02-Dec-24   2,415.9     2,323.9     973    ₹-89,581      -3.8%       +2.7%     
  COFORGE     127    IT         02-Dec-24   1,722.2     1,541.7     1365   ₹-246,383     -10.5%      +0.1%     
  SUNPHARMA   184    HEALTH     01-Oct-24   1,889.9     1,681.9     1369   ₹-284,831     -11.0% ⚠    -0.7%     
  VIJAYA      73     HEALTH     02-Dec-24   1,147.7     971.3       2048   ₹-361,256     -15.4%      -4.5%     
  KALYANKJIL  270    CON DUR    01-Aug-24   561.8       456.7       4405   ₹-462,876     -18.7% ⚠    -1.1%     
  CONCORDBIO  207    HEALTH     02-Dec-24   2,170.4     1,652.4     1083   ₹-561,044     -23.9% ⚠    -1.6%     
  SIEMENS     218    ENERGY     02-Dec-24   4,424.0     3,070.0     531    ₹-718,992     -30.6% ⚠    +1.4%     
  OFSS        311    IT         02-Dec-24   11,378.1    7,036.0     206    ₹-894,473     -38.2% ⚠    -3.4%     
  NETWEB      293    IT         02-Dec-24   2,784.8     1,508.7     844    ₹-1,077,032   -45.8% ⚠    -1.8%     
  ⚠  WAZ < 0 (momentum below universe mean): TRENT, PERSISTENT, SUNPHARMA, CONCORDBIO, SIEMENS, KALYANKJIL, NETWEB, OFSS

  AFTER: Invested ₹38,924,284 | Cash ₹1,631,945 | Total ₹40,556,228 | Positions 16/20 | Slot ₹2,027,811

========================================================================
  REBALANCE #63  —  02 Jun 2025
  NAV: ₹47,456,673  |  Slot: ₹2,372,834  |  Cash: ₹1,631,945
========================================================================

  EXITS (11)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TRENT       158    CONSUMP    01-Dec-23   2,800.7     5,610.0     621    ₹1,744,580    +100.3%   549d  
  PERSISTENT  147    IT         01-Aug-24   4,750.7     5,484.5     520    ₹381,561      +15.4%    305d  
  COFORGE     —      IT         02-Dec-24   1,722.2     1,705.3     1365   ₹-23,041      -1.0%     182d  
  KALYANKJIL  107    CON DUR    01-Aug-24   561.8       555.0       4405   ₹-29,810      -1.2%     305d  
  INDHOTEL    115    CONSUMP    02-Dec-24   795.2       777.8       2956   ₹-51,203      -2.2%     182d  
  SUNPHARMA   168    HEALTH     01-Oct-24   1,889.9     1,658.3     1369   ₹-317,029     -12.3%    244d  
  VIJAYA      197    HEALTH     02-Dec-24   1,147.7     960.0       2048   ₹-384,557     -16.4%    182d  
  CONCORDBIO  174    HEALTH     02-Dec-24   2,170.4     1,807.9     1083   ₹-392,629     -16.7%    182d  
  SIEMENS     240    ENERGY     02-Dec-24   4,424.0     3,292.0     531    ₹-601,099     -25.6%    182d  
  OFSS        211    IT         02-Dec-24   11,378.1    8,038.0     206    ₹-688,046     -29.4%    182d  
  NETWEB      176    IT         02-Dec-24   2,784.8     1,946.7     844    ₹-707,343     -30.1%    182d  

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SOLARINDS   1      DEFENCE    4.107    0.41   +68.3%    +83.8%    16,284.3    145    ₹2,361,217    +10.5%    
  JSWHL       2      FIN SVC    3.885    0.45   +241.4%   +39.8%    22,430.0    105    ₹2,355,150    -4.3%     
  GRSE        3      DEFENCE    3.624    0.46   +110.1%   +121.7%   2,945.3     805    ₹2,370,977    +17.5%    
  ZENTEC      4      DEFENCE    3.583    0.81   +121.1%   +93.6%    2,114.7     1122   ₹2,372,716    +15.6%    
  COROMANDEL  5      MFG        3.471    0.46   +84.5%    +36.7%    2,265.8     1047   ₹2,372,322    -2.5%     
  KIMS        7      HEALTH     3.166    0.46   +82.4%    +26.7%    664.1       3573   ₹2,372,829    +1.2%     
  DEEPAKFERT  8      MFG        3.159    0.32   +169.1%   +40.1%    1,469.7     1614   ₹2,372,056    +7.7%     
  BHARTIHEXA  10     IT         2.749    0.21   +80.6%    +46.6%    1,840.5     1289   ₹2,372,353    +7.6%     
  ASTERDM     11     HEALTH     2.707    0.07   +49.4%    +36.7%    549.8       4316   ₹2,372,814    +1.0%     
  POWERINDIA  12     ENERGY     2.650    0.64   +78.2%    +65.6%    19,215.4    123    ₹2,363,498    +14.4%    
  CCL         13     FMCG       2.588    0.19   +49.2%    +51.1%    882.3       2689   ₹2,372,527    +13.9%    

  HOLDS (5)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GVT&D       9      ENERGY     03-Oct-23   424.5       2,295.2     3842   ₹7,187,150    +440.7%     +17.9%    
  BSE         6      FIN SVC    02-Dec-24   1,516.5     2,693.3     1550   ₹1,824,040    +77.6%      +12.3%    
  GODFRYPHLP  21     FMCG       01-Oct-24   2,234.8     2,761.7     1158   ₹610,189      +23.6%      +0.2%     
  FORTIS      28     HEALTH     02-Dec-24   676.2       721.5       3477   ₹157,669      +6.7%       +3.6%     
  RADICO      26     FMCG       02-Dec-24   2,415.9     2,545.2     973    ₹125,772      +5.4%       +2.2%     

  AFTER: Invested ₹47,234,411 | Cash ₹191,320 | Total ₹47,425,731 | Positions 16/20 | Slot ₹2,372,834

========================================================================
  REBALANCE #64  —  01 Aug 2025
  NAV: ₹49,548,588  |  Slot: ₹2,477,429  |  Cash: ₹191,320
========================================================================
  [SECTOR CAP≤4] dropped: MCX

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  GODFRYPHLP  49     FMCG       01-Oct-24   2,234.8     2,893.6     1158   ₹762,897      +29.5%    304d  
  DEEPAKFERT  33     MFG        02-Jun-25   1,469.7     1,547.4     1614   ₹125,478      +5.3%     60d   
  BHARTIHEXA  45     IT         02-Jun-25   1,840.5     1,844.6     1289   ₹5,336        +0.2%     60d   
  GRSE        54     DEFENCE    02-Jun-25   2,945.3     2,562.7     805    ₹-308,003     -13.0%    60d   
  SOLARINDS   77     DEFENCE    02-Jun-25   16,284.3    13,807.0    145    ₹-359,202     -15.2%    60d   
  ZENTEC      159    DEFENCE    02-Jun-25   2,114.7     1,531.0     1122   ₹-654,886     -27.6%    60d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ECLERX      2      IT         3.639    0.60   +58.4%    +57.8%    1,899.1     1304   ₹2,476,459    +3.9%     
  ANANDRATHI  3      FIN SVC    3.547    0.18   +39.0%    +53.0%    1,295.5     1912   ₹2,476,984    +3.8%     
  LAURUSLABS  4      HEALTH     3.544    0.82   +85.5%    +38.8%    847.8       2922   ₹2,477,288    +2.8%     
  SCHNEIDER   5      ENERGY     3.387    0.86   +25.8%    +77.5%    1,001.3     2474   ₹2,477,216    +10.6%    
  JKCEMENT    6      MFG        3.360    0.27   +48.6%    +31.2%    6,681.5     370    ₹2,472,155    +3.5%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GVT&D       1      ENERGY     03-Oct-23   424.5       2,857.9     3842   ₹9,348,926    +573.2%     +15.1%    
  BSE         7      FIN SVC    02-Dec-24   1,516.5     2,411.3     1550   ₹1,386,940    +59.0%      -4.0%     
  FORTIS      8      HEALTH     02-Dec-24   676.2       859.2       3477   ₹636,637      +27.1%      +4.7%     
  RADICO      24     FMCG       02-Dec-24   2,415.9     2,838.8     973    ₹411,437      +17.5%      +4.7%     
  KIMS        13     HEALTH     02-Jun-25   664.1       756.5       3573   ₹329,967      +13.9%      +2.5%     
  COROMANDEL  25     MFG        02-Jun-25   2,265.8     2,580.2     1047   ₹329,177      +13.9%      +6.3%     
  ASTERDM     15     HEALTH     02-Jun-25   549.8       591.8       4316   ₹181,468      +7.6%       +0.1%     
  POWERINDIA  18     ENERGY     02-Jun-25   19,215.4    20,544.0    123    ₹163,420      +6.9%       +4.8%     
  CCL         28     FMCG       02-Jun-25   882.3       886.5       2689   ₹11,331       +0.5%       +3.8%     
  JSWHL       31     FIN SVC    02-Jun-25   22,430.0    19,161.0    105    ₹-343,245     -14.6%      -9.9%     

  AFTER: Invested ₹47,728,555 | Cash ₹1,805,333 | Total ₹49,533,888 | Positions 15/20 | Slot ₹2,477,429

========================================================================
  REBALANCE #65  —  01 Oct 2025
  NAV: ₹49,042,026  |  Slot: ₹2,452,101  |  Cash: ₹1,805,333
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BSE         88     FIN SVC    02-Dec-24   1,516.5     2,081.4     1550   ₹875,594      +37.3%    303d  
  ASTERDM     —      HEALTH     02-Jun-25   549.8       628.1       4316   ₹338,060      +14.2%    121d  
  CCL         76     FMCG       02-Jun-25   882.3       851.4       2689   ₹-83,246      -3.5%     121d  
  POWERINDIA  92     ENERGY     02-Jun-25   19,215.4    18,134.0    123    ₹-133,016     -5.6%     121d  
  SCHNEIDER   120    ENERGY     01-Aug-25   1,001.3     828.0       2474   ₹-428,868     -17.3%    61d   
  JSWHL       —      OTHER      02-Jun-25   22,430.0    15,494.0    105    ₹-728,280     -30.9%    121d  

  ENTRIES (6)
  [52w filter blocked 1: FORCEMOT(-23.2%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  EICHERMOT   2      AUTO       3.830    0.56   +42.4%    +24.3%    7,021.5     349    ₹2,450,504    +3.0%     
  MUTHOOTFIN  7      FIN SVC    3.374    0.35   +55.5%    +19.8%    3,118.1     786    ₹2,450,854    +5.7%     
  LTF         8      FIN SVC    3.229    1.01   +40.6%    +25.4%    255.9       9580   ₹2,451,952    +7.6%     
  CHOICEIN    9      FIN SVC    3.228    0.78   +63.6%    +10.6%    766.2       3200   ₹2,451,840    -4.0%     
  SYRMA       10     MFG        3.143    0.95   +78.6%    +33.4%    804.1       3049   ₹2,451,701    +0.1%     
  INDIANB     11     PSU BNK    2.749    0.43   +41.9%    +13.3%    721.7       3397   ₹2,451,534    +4.8%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GVT&D       5      ENERGY     03-Oct-23   424.5       3,073.0     3842   ₹10,175,478   +623.9%     +5.2%     
  FORTIS      4      HEALTH     02-Dec-24   676.2       989.7       3477   ₹1,090,038    +46.4%      +3.4%     
  RADICO      17     FMCG       02-Dec-24   2,415.9     2,914.2     973    ₹484,801      +20.6%      -0.1%     
  ANANDRATHI  2      FIN SVC    01-Aug-25   1,295.5     1,437.9     1912   ₹272,353      +11.0%      -0.7%     
  ECLERX      24     IT         01-Aug-25   1,899.1     1,986.8     1304   ₹114,263      +4.6%       -5.4%     
  KIMS        43     HEALTH     02-Jun-25   664.1       687.9       3573   ₹85,038       +3.6%       -5.6%     
  LAURUSLABS  7      HEALTH     01-Aug-25   847.8       870.4       2922   ₹66,063       +2.7%       -0.8%     
  COROMANDEL  39     MFG        02-Jun-25   2,265.8     2,242.2     1047   ₹-24,779      -1.0%       -0.5%     
  JKCEMENT    27     MFG        01-Aug-25   6,681.5     6,305.0     370    ₹-139,305     -5.6%       -4.6%     

  AFTER: Invested ₹47,813,051 | Cash ₹1,211,510 | Total ₹49,024,561 | Positions 15/20 | Slot ₹2,452,101

========================================================================
  REBALANCE #66  —  01 Dec 2025
  NAV: ₹50,566,957  |  Slot: ₹2,528,348  |  Cash: ₹1,211,510
========================================================================
  [SECTOR CAP≤4] dropped: BAJFINANCE, FORCEMOT, TVSMOTOR

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  GVT&D       79     ENERGY     03-Oct-23   424.5       2,801.3     3842   ₹9,131,607    +559.9%   790d  
  FORTIS      96     HEALTH     02-Dec-24   676.2       904.8       3477   ₹795,188      +33.8%    364d  
  RADICO      53     FMCG       02-Dec-24   2,415.9     3,217.0     973    ₹779,425      +33.2%    364d  
  ECLERX      68     IT         01-Aug-25   1,899.1     2,344.9     1304   ₹581,291      +23.5%    122d  
  ANANDRATHI  60     FIN SVC    01-Aug-25   1,295.5     1,459.6     1912   ₹313,745      +12.7%    122d  
  COROMANDEL  84     MFG        02-Jun-25   2,265.8     2,382.9     1047   ₹122,581      +5.2%     182d  
  KIMS        155    HEALTH     02-Jun-25   664.1       690.0       3573   ₹92,719       +3.9%     182d  
  CHOICEIN    54     FIN SVC    01-Oct-25   766.2       794.9       3200   ₹91,840       +3.7%     61d   
  SYRMA       82     MFG        01-Oct-25   804.1       813.0       3049   ₹27,289       +1.1%     61d   
  JKCEMENT    148    MFG        01-Aug-25   6,681.5     5,783.5     370    ₹-332,260     -13.4%    122d  

  ENTRIES (14)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  CANBK       3      PSU BNK    3.584    0.93   +53.2%    +44.9%    145.7       17353  ₹2,528,226    +3.7%     
  CUB         4      PVT BNK    3.411    0.62   +59.4%    +44.3%    212.2       11914  ₹2,528,300    +7.1%     
  M&MFIN      6      FIN SVC    3.261    0.84   +39.5%    +44.9%    367.9       6872   ₹2,528,209    +9.0%     
  SHRIRAMFIN  7      FIN SVC    3.010    1.07   +41.9%    +47.6%    851.5       2969   ₹2,528,252    +4.5%     
  NAVINFLUOR  9      MFG        2.806    0.42   +67.0%    +22.8%    5,743.1     440    ₹2,526,960    -0.3%     
  BANKINDIA   11     PSU BNK    2.701    0.91   +41.6%    +33.5%    142.6       17732  ₹2,528,271    +1.8%     
  SBIN        12     PSU BNK    2.638    0.64   +18.3%    +21.3%    955.9       2645   ₹2,528,277    +1.1%     
  MARUTI      14     AUTO       2.628    0.50   +48.7%    +8.8%     16,097.0    157    ₹2,527,229    +1.2%     
  HEROMOTOCO  15     AUTO       2.615    0.77   +35.4%    +23.7%    6,175.1     409    ₹2,525,631    +7.1%     
  ASHOKLEY    16     AUTO       2.552    0.77   +41.9%    +27.1%    157.6       16041  ₹2,528,243    +8.3%     
  NYKAA       17     CONSUMP    2.537    0.76   +57.6%    +15.1%    264.9       9544   ₹2,528,206    +0.6%     
  CHENNPETRO  20     OIL&GAS    2.358    0.74   +50.7%    +41.2%    904.4       2795   ₹2,527,757    -5.2%     
  AIAENG      21     MFG        2.332    0.61   +11.0%    +26.8%    3,854.1     656    ₹2,528,290    +5.1%     
  PTCIL       22     MFG        2.278    0.80   +56.1%    +30.3%    18,255.0    138    ₹2,519,190    +4.0%     

  HOLDS (5)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  LAURUSLABS  9      HEALTH     01-Aug-25   847.8       1,028.4     2922   ₹527,627      +21.3%      +4.2%     
  MUTHOOTFIN  2      FIN SVC    01-Oct-25   3,118.1     3,779.1     786    ₹519,554      +21.2%      +6.6%     
  INDIANB     5      PSU BNK    01-Oct-25   721.7       868.8       3397   ₹499,919      +20.4%      +2.6%     
  LTF         1      FIN SVC    01-Oct-25   255.9       306.0       9580   ₹479,523      +19.6%      +4.9%     
  EICHERMOT   11     AUTO       01-Oct-25   7,021.5     7,125.5     349    ₹36,296       +1.5%       +1.6%     

  AFTER: Invested ₹49,726,093 | Cash ₹798,853 | Total ₹50,524,945 | Positions 19/20 | Slot ₹2,528,348

========================================================================
  REBALANCE #67  —  02 Feb 2026
  NAV: ₹49,714,276  |  Slot: ₹2,485,714  |  Cash: ₹798,853
========================================================================

  [REGIME OFF] Nifty 500 22,837.0 < EMA200 23,100.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ASHOKLEY    1      AUTO       01-Dec-25   157.6       191.1       16041  ₹537,823      +21.3%      +3.5%     
  INDIANB     32     PSU BNK    01-Oct-25   721.7       817.4       3397   ₹325,131      +13.3%      -3.0%     
  SHRIRAMFIN  4      FIN SVC    01-Dec-25   851.5       962.1       2969   ₹328,223      +13.0%      -2.6%     
  MUTHOOTFIN  18     FIN SVC    01-Oct-25   3,118.1     3,505.5     786    ₹304,437      +12.4%      -8.4%     
  LAURUSLABS  26     HEALTH     01-Aug-25   847.8       953.0       2922   ₹307,381      +12.4%      -6.7%     
  LTF         11     FIN SVC    01-Oct-25   255.9       274.7       9580   ₹179,881      +7.3%       -5.2%     
  SBIN        10     PSU BNK    01-Dec-25   955.9       1,010.5     2645   ₹144,458      +5.7%       -0.3%     
  AIAENG      24     MFG        01-Dec-25   3,854.1     4,016.2     656    ₹106,338      +4.2%       +2.9%     
  BANKINDIA   31     PSU BNK    01-Dec-25   142.6       146.9       17732  ₹76,068       +3.0%       -3.1%     
  NAVINFLUOR  12     MFG        01-Dec-25   5,743.1     5,893.9     440    ₹66,360       +2.6%       -0.7%     
  CUB         8      PVT BNK    01-Dec-25   212.2       214.1       11914  ₹22,339       +0.9%       +0.6%     
  EICHERMOT   43     AUTO       01-Oct-25   7,021.5     6,985.5     349    ₹-12,564      -0.5%       -2.8%     
  PTCIL       87     MFG        01-Dec-25   18,255.0    17,980.0    138    ₹-37,950      -1.5%       +0.3%     
  CANBK       16     PSU BNK    01-Dec-25   145.7       141.8       17353  ₹-67,699      -2.7%       -3.6%     
  M&MFIN      34     FIN SVC    01-Dec-25   367.9       353.6       6872   ₹-98,270      -3.9%       -2.9%     
  CHENNPETRO  83     OIL&GAS    01-Dec-25   904.4       857.7       2795   ₹-130,610     -5.2%       +1.9%     
  NYKAA       61     CONSUMP    01-Dec-25   264.9       237.6       9544   ₹-260,360     -10.3%      -3.3%     
  MARUTI      157    AUTO       01-Dec-25   16,097.0    14,384.0    157    ₹-268,941     -10.6% ⚠    -7.8%     
  HEROMOTOCO  38     AUTO       01-Dec-25   6,175.1     5,515.5     409    ₹-269,794     -10.7%      -0.1%     
  ⚠  WAZ < 0 (momentum below universe mean): MARUTI

  AFTER: Invested ₹48,915,423 | Cash ₹798,853 | Total ₹49,714,276 | Positions 19/20 | Slot ₹2,485,714

========================================================================
  REBALANCE #68  —  01 Apr 2026
  NAV: ₹46,710,728  |  Slot: ₹2,335,536  |  Cash: ₹798,853
========================================================================

  [REGIME OFF] Nifty 500 20,935.2 < EMA200 22,889.0 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  LAURUSLABS  34     HEALTH     01-Aug-25   847.8       1,037.8     2922   ₹555,067      +22.4%      +3.2%     
  INDIANB     7      PSU BNK    01-Oct-25   721.7       869.5       3397   ₹502,082      +20.5%      -0.2%     
  CHENNPETRO  12     OIL&GAS    01-Dec-25   904.4       1,009.8     2795   ₹294,773      +11.7%      +4.0%     
  SHRIRAMFIN  —      FIN SVC    01-Dec-25   851.5       900.5       2969   ₹145,481      +5.8%       -6.5%     
  NAVINFLUOR  42     MFG        01-Dec-25   5,743.1     6,026.7     440    ₹124,810      +4.9%       -2.7%     
  SBIN        26     PSU BNK    01-Dec-25   955.9       999.8       2645   ₹116,138      +4.6%       -4.7%     
  MUTHOOTFIN  93     FIN SVC    01-Oct-25   3,118.1     3,228.7     786    ₹86,904       +3.5%       -1.7%     
  EICHERMOT   74     AUTO       01-Oct-25   7,021.5     6,825.5     349    ₹-68,404      -2.8%       -3.2%     
  BANKINDIA   55     PSU BNK    01-Dec-25   142.6       137.2       17732  ₹-95,128      -3.8%       -6.2%     
  AIAENG      108    MFG        01-Dec-25   3,854.1     3,681.0     656    ₹-113,554     -4.5%       +1.6%     
  LTF         83     FIN SVC    01-Oct-25   255.9       242.1       9580   ₹-132,184     -5.4%       -6.6%     
  ASHOKLEY    90     AUTO       01-Dec-25   157.6       146.6       16041  ₹-176,488     -7.0%       -14.6%    
  NYKAA       67     CONSUMP    01-Dec-25   264.9       240.0       9544   ₹-237,932     -9.4%       -2.1%     
  CUB         98     PVT BNK    01-Dec-25   212.2       179.8       11914  ₹-386,639     -15.3%      -4.4%     
  CANBK       91     PSU BNK    01-Dec-25   145.7       123.2       17353  ₹-389,733     -15.4%      -6.8%     
  HEROMOTOCO  56     AUTO       01-Dec-25   6,175.1     5,122.0     409    ₹-430,733     -17.1%      -3.5%     
  PTCIL       162    MFG        01-Dec-25   18,255.0    15,030.0    138    ₹-445,050     -17.7% ⚠    -10.5%    
  M&MFIN      —      FIN SVC    01-Dec-25   367.9       289.7       6872   ₹-537,390     -21.3%      -10.3%    
  MARUTI      276    AUTO       01-Dec-25   16,097.0    12,509.0    157    ₹-563,316     -22.3% ⚠    -4.6%     
  ⚠  WAZ < 0 (momentum below universe mean): PTCIL, MARUTI

  AFTER: Invested ₹45,911,875 | Cash ₹798,853 | Total ₹46,710,728 | Positions 19/20 | Slot ₹2,335,536

========================================================================
  REBALANCE #69  —  01 Jun 2026
  NAV: ₹49,873,785  |  Slot: ₹2,493,689  |  Cash: ₹798,853
========================================================================
  [SECTOR CAP≤4] dropped: CEMPRO, GVT&D

  [REGIME OFF] Nifty 500 22,437.9 < EMA200 22,792.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  LAURUSLABS  2      HEALTH     01-Aug-25   847.8       1,388.4     2922   ₹1,579,617    +63.8%      +6.1%     
  NAVINFLUOR  39     MFG        01-Dec-25   5,743.1     6,990.1     440    ₹548,680      +21.7%      -0.6%     
  CHENNPETRO  50     OIL&GAS    01-Dec-25   904.4       1,096.9     2795   ₹538,078      +21.3%      +5.4%     
  AIAENG      40     MFG        01-Dec-25   3,854.1     4,522.7     656    ₹438,602      +17.3%      +9.8%     
  INDIANB     156    PSU BNK    01-Oct-25   721.7       792.9       3397   ₹241,811      +9.9% ⚠     -3.3%     
  SHRIRAMFIN  140    FIN SVC    01-Dec-25   851.5       919.0       2969   ₹200,408      +7.9%       -3.3%     
  LTF         86     FIN SVC    01-Oct-25   255.9       271.2       9580   ₹146,623      +6.0%       -2.9%     
  MUTHOOTFIN  82     FIN SVC    01-Oct-25   3,118.1     3,246.4     786    ₹100,816      +4.1%       -3.3%     
  PTCIL       120    MFG        01-Dec-25   18,255.0    18,489.0    138    ₹32,292       +1.3%       +11.5%    
  EICHERMOT   127    AUTO       01-Oct-25   7,021.5     7,100.5     349    ₹27,571       +1.1%       -0.9%     
  NYKAA       92     CONSUMP    01-Dec-25   264.9       266.7       9544   ₹17,179       +0.7%       -0.3%     
  SBIN        229    PSU BNK    01-Dec-25   955.9       954.1       2645   ₹-4,683       -0.2% ⚠     -2.5%     
  BANKINDIA   204    PSU BNK    01-Dec-25   142.6       136.7       17732  ₹-103,598     -4.1% ⚠     -1.3%     
  ASHOKLEY    245    AUTO       01-Dec-25   157.6       147.3       16041  ₹-165,921     -6.6% ⚠     -5.9%     
  CUB         149    PVT BNK    01-Dec-25   212.2       190.6       11914  ₹-257,789     -10.2% ⚠    -0.2%     
  CANBK       218    PSU BNK    01-Dec-25   145.7       123.9       17353  ₹-378,646     -15.0% ⚠    -2.8%     
  MARUTI      251    AUTO       01-Dec-25   16,097.0    12,946.0    157    ₹-494,707     -19.6% ⚠    -1.8%     
  M&MFIN      233    FIN SVC    01-Dec-25   367.9       295.1       6872   ₹-500,282     -19.8% ⚠    -4.5%     
  HEROMOTOCO  227    AUTO       01-Dec-25   6,175.1     4,819.9     409    ₹-554,292     -21.9% ⚠    -4.1%     
  ⚠  WAZ < 0 (momentum below universe mean): CUB, INDIANB, BANKINDIA, CANBK, HEROMOTOCO, SBIN, M&MFIN, ASHOKLEY, MARUTI

  AFTER: Invested ₹49,074,932 | Cash ₹798,853 | Total ₹49,873,785 | Positions 19/20 | Slot ₹2,493,689

========================================================================
  FINAL SUMMARY
========================================================================
  Period        : 2015-01-01 → 2026-07-01  (11.5 years)
  Starting Cap  : ₹2,000,000
  Final Value   : ₹53,024,734
  Total Return  : +2551.2%  (on total invested)
  CAGR          : +33.0%

  Closed Trades : 414  |  Open: 19
  Win Rate      : 59.4%  (246W / 168L)
  Profit Factor : 3.50
  Avg hold      : 171 days
  Total charges : ₹1,225,870
  Closed net P&L: ₹46,129,261
  Open unreal   : ₹4,562,710

  YEAR-BY-YEAR:
  2015  -  0.1%  
  2016  + 14.9%  ██████████████
  2017  + 75.3%  ████████████████████████████████████████
  2018  +  2.2%  ██
  2019  + 14.4%  ██████████████
  2020  + 66.0%  ████████████████████████████████████████
  2021  +117.2%  ████████████████████████████████████████
  2022  + 27.6%  ███████████████████████████
  2023  + 60.8%  ████████████████████████████████████████
  2024  + 35.0%  ███████████████████████████████████
  2025  +  7.5%  ███████
  2026  -  1.4%  ░

  Rebalance NAV exported → mom500_rebal.csv (69 rows)
