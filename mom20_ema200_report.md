=== Mom20 — 2-Monthly Rebalance, β≤1.2 | Regime ON [EMA200] | Sector≤4 ===
    top_n=20 buffer_in=15 buffer_out=40 beta_cap=1.2
Loading PIT universe...
  388 unique PIT tickers across all periods
Loading EPS data...
  871 stocks with EPS data
  Sector map loaded: 43 PIT dates
Loading cached data from /Users/jay/dev/relative_strength/data/cache/mom15_daily.pkl...
Fetching Nifty 50 (beta)...
  3127 bars
Fetching Nifty 200 (regime filter)...
  3133 bars
  Trading days in backtest: 2840 (2015-01-01 → 2026-07-01)
  Rebalance dates: 69

==============================================================================================
  MOM20 PIT BACKTEST  |  NAV/20 slot  |  Monthly Rebalance  |  Beta≤1.2  |  Regime ON [EMA200]
==============================================================================================

========================================================================
  REBALANCE #01  —  02 Feb 2015
  NAV: ₹2,000,000  |  Slot: ₹100,000  |  Cash: ₹2,000,000
========================================================================

  EXITS (0)
    —

  ENTRIES (20)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BOSCHLTD    1      AUTO       4.014    0.10   +163.6%   +67.5%    22,437.0    4      ₹89,748       +14.3%    
  HONAUT      2      MFG        3.196    -0.19  +184.0%   +40.5%    7,224.6     13     ₹93,919       +3.3%     
  AXISBANK    3      PVT BNK    3.093    0.10   +168.1%   +43.5%    597.6       167    ₹99,792       +11.9%    
  WHIRLPOOL   4      CON DUR    3.061    0.46   +277.4%   +50.5%    693.1       144    ₹99,799       +3.3%     
  BEL         5      DEFENCE    3.058    -0.09  +245.9%   +67.9%    28.5        3505   ₹99,988       +6.7%     
  EICHERMOT   6      AUTO       2.725    0.15   +229.2%   +32.2%    1,530.4     65     ₹99,477       +6.3%     
  SHREECEM    7      INFRA      2.660    0.26   +147.9%   +25.9%    10,437.7    9      ₹93,939       +6.1%     
  IBULHSGFIN  8      FIN SVC    2.652    0.53   +214.3%   +50.8%    382.5       261    ₹99,838       +8.0%     
  ASHOKLEY    9      AUTO       2.626    -0.15  +293.2%   +42.7%    26.9        3716   ₹99,986       +6.9%     
  BAJFINANCE  10     FIN SVC    2.618    -0.05  +167.1%   +48.8%    39.9        2507   ₹99,987       +7.9%     
  BBTC        11     FMCG       2.556    0.14   +322.4%   +84.3%    438.1       228    ₹99,890       +1.1%     
  BHARATFORG  12     DEFENCE    2.554    -0.04  +210.7%   +33.0%    490.3       203    ₹99,535       +6.6%     
  BAJAJFINSV  13     FIN SVC    2.449    0.17   +121.0%   +40.6%    149.9       667    ₹99,993       +11.8%    
  AUROPHARMA  14     HEALTH     2.406    0.01   +209.3%   +32.8%    596.9       167    ₹99,688       +7.4%     
  AMARAJABAT  15     AUTO       2.349    0.10   +169.8%   +37.1%    801.8       124    ₹99,418       +4.9%     
  SRF         16     MFG        2.345    -0.11  +378.9%   +7.4%     181.9       549    ₹99,844       +4.3%     
  WOCKPHARMA  17     HEALTH     2.209    0.26   +196.2%   +63.7%    1,099.6     90     ₹98,968       +9.7%     
  SPARC       18     HEALTH     2.202    0.10   +127.8%   +83.7%    358.5       278    ₹99,652       +30.6%    
  REPCOHOME   19     FIN SVC    2.184    -0.07  +109.9%   +38.5%    609.4       164    ₹99,943       -0.6%     
  ALSTOMT&D   20     ENERGY     2.147    0.34   +124.9%   +41.8%    673.8       148    ₹99,719       +2.6%     

  HOLDS (0)
    —

  AFTER: Invested ₹1,973,127 | Cash ₹24,531 | Total ₹1,997,657 | Positions 20/20 | Slot ₹100,000

========================================================================
  REBALANCE #02  —  01 Apr 2015
  NAV: ₹2,126,929  |  Slot: ₹106,346  |  Cash: ₹24,531
========================================================================
  [SECTOR CAP≤4] dropped: NATCOPHARM, SUNPHARMA

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ALSTOMT&D   51     ENERGY     02-Feb-15   673.8       710.7       148    ₹5,467        +5.5%     58d   
  BAJFINANCE  25     FIN SVC    02-Feb-15   39.9        39.7        2507   ₹-482         -0.5%     58d   
  AUROPHARMA  44     HEALTH     02-Feb-15   596.9       590.1       167    ₹-1,145       -1.1%     58d   
  EICHERMOT   31     AUTO       02-Feb-15   1,530.4     1,485.0     65     ₹-2,954       -3.0%     58d   
  REPCOHOME   83     FIN SVC    02-Feb-15   609.4       580.5       164    ₹-4,746       -4.7%     58d   
  AMARAJABAT  61     AUTO       02-Feb-15   801.8       761.8       124    ₹-4,955       -5.0%     58d   
  BAJAJFINSV  52     FIN SVC    02-Feb-15   149.9       141.8       667    ₹-5,416       -5.4%     58d   
  AXISBANK    38     PVT BNK    02-Feb-15   597.6       551.4       167    ₹-7,703       -7.7%     58d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  LUPIN       2      HEALTH     3.567    0.27   +116.3%   +42.7%    1,917.4     55     ₹105,460      +6.5%     
  SADBHAV     7      INFRA      2.782    0.39   +291.5%   +40.1%    342.4       310    ₹106,148      +1.8%     
  SIEMENS     9      ENERGY     2.614    0.39   +108.8%   +59.3%    737.5       144    ₹106,205      +3.3%     
  NCC         10     INFRA      2.434    0.40   +460.7%   +33.2%    97.5        1090   ₹106,288      +13.5%    
  BRITANNIA   11     FMCG       2.399    0.20   +157.9%   +22.1%    946.0       112    ₹105,946      +1.4%     
  AJANTPHARM  12     HEALTH     2.335    0.35   +208.0%   +30.3%    737.1       144    ₹106,143      +2.9%     
  EMAMILTD    17     FMCG       2.160    0.26   +126.1%   +26.4%    420.8       252    ₹106,047      -1.9%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  WOCKPHARMA  3      HEALTH     02-Feb-15   1,099.6     1,701.9     90     ₹54,206       +54.8%      +3.9%     
  SPARC       1      HEALTH     02-Feb-15   358.5       500.0       278    ₹39,347       +39.5%      +5.2%     
  BHARATFORG  4      DEFENCE    02-Feb-15   490.3       610.1       203    ₹24,305       +24.4%      +4.3%     
  HONAUT      6      MFG        02-Feb-15   7,224.6     8,387.7     13     ₹15,121       +16.1%      -0.9%     
  ASHOKLEY    5      AUTO       02-Feb-15   26.9        30.2        3716   ₹12,110       +12.1%      +4.6%     
  SRF         19     MFG        02-Feb-15   181.9       189.4       549    ₹4,131        +4.1%       +6.7%     
  BOSCHLTD    8      AUTO       02-Feb-15   22,437.0    22,972.6    4      ₹2,142        +2.4%       -1.7%     
  WHIRLPOOL   16     CON DUR    02-Feb-15   693.1       709.4       144    ₹2,351        +2.4%       +0.7%     
  BEL         14     DEFENCE    02-Feb-15   28.5        29.1        3505   ₹2,078        +2.1%       +4.8%     
  SHREECEM    18     INFRA      02-Feb-15   10,437.7    10,419.0    9      ₹-169         -0.2%       -0.1%     
  BBTC        20     FMCG       02-Feb-15   438.1       435.4       228    ₹-619         -0.6%       -1.0%     
  IBULHSGFIN  22     FIN SVC    02-Feb-15   382.5       368.0       261    ₹-3,800       -3.8%       +0.6%     

  AFTER: Invested ₹2,068,551 | Cash ₹57,497 | Total ₹2,126,048 | Positions 19/20 | Slot ₹106,346

========================================================================
  REBALANCE #03  —  01 Jun 2015
  NAV: ₹2,082,031  |  Slot: ₹104,102  |  Cash: ₹57,497
========================================================================
  [SECTOR CAP≤4] dropped: APLLTD, AUROPHARMA, ZYDUSLIFE

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  WOCKPHARMA  70     HEALTH     02-Feb-15   1,099.6     1,341.2     90     ₹21,738       +22.0%    119d  
  BEL         37     DEFENCE    02-Feb-15   28.5        30.1        3505   ₹5,413        +5.4%     119d  
  IBULHSGFIN  60     FIN SVC    02-Feb-15   382.5       397.3       261    ₹3,849        +3.9%     119d  
  HONAUT      97     MFG        02-Feb-15   7,224.6     7,173.1     13     ₹-669         -0.7%     119d  
  SIEMENS     63     ENERGY     01-Apr-15   737.5       713.3       144    ₹-3,493       -3.3%     61d   
  BOSCHLTD    58     AUTO       02-Feb-15   22,437.0    20,961.6    4      ₹-5,902       -6.6%     119d  
  SADBHAV     98     INFRA      01-Apr-15   342.4       280.9       310    ₹-19,071      -18.0%    61d   
  NCC         72     INFRA      01-Apr-15   97.5        72.8        1090   ₹-26,968      -25.4%    61d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  PAGEIND     1      MFG        4.085    0.16   +185.4%   +45.2%    15,087.9    6      ₹90,527       +14.7%    
  NATCOPHARM  4      HEALTH     3.504    0.69   +212.4%   +63.3%    407.7       255    ₹103,958      -0.5%     
  EICHERMOT   5      AUTO       3.072    0.15   +196.3%   +20.8%    1,740.7     59     ₹102,699      +5.2%     
  MARICO      6      FMCG       2.961    -0.01  +101.2%   +25.6%    190.3       546    ₹103,922      +8.3%     
  INDUSTOWER  12     INFRA      2.608    -0.02  +107.5%   +26.6%    327.7       317    ₹103,867      +9.6%     
  UPL         15     MFG        2.395    0.22   +79.6%    +32.0%    325.1       320    ₹104,041      +4.3%     
  DCBBANK     17     PVT BNK    2.272    0.11   +99.8%    +22.6%    124.9       833    ₹104,051      +3.6%     
  BAJFINANCE  18     FIN SVC    2.181    0.17   +145.2%   +7.3%     42.0        2479   ₹104,094      -0.1%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  AJANTPHARM  2      HEALTH     01-Apr-15   737.1       976.8       144    ₹34,516       +32.5%      +15.1%    
  SRF         14     MFG        02-Feb-15   181.9       216.0       549    ₹18,716       +18.7%      +12.8%    
  BRITANNIA   3      FMCG       01-Apr-15   946.0       1,115.5     112    ₹18,992       +17.9%      +6.8%     
  BBTC        10     FMCG       02-Feb-15   438.1       500.5       228    ₹14,223       +14.2%      +10.3%    
  BHARATFORG  28     DEFENCE    02-Feb-15   490.3       560.0       203    ₹14,146       +14.2%      -2.3%     
  EMAMILTD    8      FMCG       01-Apr-15   420.8       479.2       252    ₹14,704       +13.9%      +9.5%     
  SPARC       36     HEALTH     02-Feb-15   358.5       398.2       278    ₹11,059       +11.1%      -2.5%     
  ASHOKLEY    26     AUTO       02-Feb-15   26.9        29.3        3716   ₹8,931        +8.9%       +2.1%     
  WHIRLPOOL   13     CON DUR    02-Feb-15   693.1       729.2       144    ₹5,201        +5.2%       +2.7%     
  SHREECEM    29     INFRA      02-Feb-15   10,437.7    10,938.8    9      ₹4,510        +4.8%       +2.0%     
  LUPIN       16     HEALTH     01-Apr-15   1,917.4     1,686.5     55     ₹-12,704      -12.0%      +1.8%     

  AFTER: Invested ₹2,065,695 | Cash ₹15,366 | Total ₹2,081,061 | Positions 19/20 | Slot ₹104,102

========================================================================
  REBALANCE #04  —  03 Aug 2015
  NAV: ₹2,204,672  |  Slot: ₹110,234  |  Cash: ₹15,366
========================================================================

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SPARC       64     HEALTH     02-Feb-15   358.5       398.4       278    ₹11,114       +11.2%    182d  
  BHARATFORG  96     DEFENCE    02-Feb-15   490.3       539.6       203    ₹10,008       +10.1%    182d  
  SHREECEM    51     INFRA      02-Feb-15   10,437.7    10,979.2    9      ₹4,873        +5.2%     182d  
  NATCOPHARM  50     HEALTH     01-Jun-15   407.7       425.0       255    ₹4,427        +4.3%     63d   
  DCBBANK     55     PVT BNK    01-Jun-15   124.9       128.0       833    ₹2,585        +2.5%     63d   
  UPL         60     MFG        01-Jun-15   325.1       327.1       320    ₹618          +0.6%     63d   
  LUPIN       91     HEALTH     01-Apr-15   1,917.4     1,572.2     55     ₹-18,989      -18.0%    124d  
  PAGEIND     81     MFG        01-Jun-15   15,087.9    12,198.5    6      ₹-17,336      -19.2%    63d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RAJESHEXPO  1      CON DUR    4.593    0.46   +239.0%   +148.2%   561.3       196    ₹110,011      +23.9%    
  HINDPETRO   4      OIL&GAS    3.048    0.44   +142.5%   +53.0%    81.6        1350   ₹110,154      +3.8%     
  APLLTD      6      HEALTH     2.856    0.19   +112.2%   +57.6%    657.5       167    ₹109,810      +0.6%     
  MARUTI      7      AUTO       2.703    0.31   +75.0%    +21.8%    4,018.0     27     ₹108,485      +5.9%     
  IBULHSGFIN  10     FIN SVC    2.521    0.26   +106.7%   +32.1%    498.5       221    ₹110,175      +9.1%     
  BAJAJFINSV  11     FIN SVC    2.481    0.19   +100.6%   +32.5%    189.4       581    ₹110,048      +9.3%     
  ABBOTINDIA  14     HEALTH     2.262    0.16   +114.7%   +16.7%    4,000.7     27     ₹108,018      +5.7%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BRITANNIA   2      FMCG       01-Apr-15   946.0       1,370.3     112    ₹47,522       +44.9%      +7.7%     
  SRF         9      MFG        02-Feb-15   181.9       262.4       549    ₹44,239       +44.3%      +5.9%     
  EMAMILTD    3      FMCG       01-Apr-15   420.8       562.9       252    ₹35,810       +33.8%      +9.9%     
  BBTC        8      FMCG       02-Feb-15   438.1       585.5       228    ₹33,601       +33.6%      +11.6%    
  BAJFINANCE  5      FIN SVC    01-Jun-15   42.0        54.7        2479   ₹31,421       +30.2%      +7.3%     
  ASHOKLEY    12     AUTO       02-Feb-15   26.9        34.8        3716   ₹29,508       +29.5%      +7.1%     
  AJANTPHARM  23     HEALTH     01-Apr-15   737.1       938.5       144    ₹28,998       +27.3%      -1.9%     
  WHIRLPOOL   39     CON DUR    02-Feb-15   693.1       738.6       144    ₹6,555        +6.6%       +2.4%     
  INDUSTOWER  29     INFRA      01-Jun-15   327.7       334.8       317    ₹2,268        +2.2%       +7.3%     
  EICHERMOT   13     AUTO       01-Jun-15   1,740.7     1,769.0     59     ₹1,669        +1.6%       -4.1%     
  MARICO      30     FMCG       01-Jun-15   190.3       184.9       546    ₹-2,988       -2.9%       -0.3%     

  AFTER: Invested ₹2,157,545 | Cash ₹46,216 | Total ₹2,203,762 | Positions 18/20 | Slot ₹110,234

========================================================================
  REBALANCE #05  —  01 Oct 2015
  NAV: ₹2,101,820  |  Slot: ₹105,091  |  Cash: ₹46,216
========================================================================
  [SECTOR CAP≤4] dropped: TORNTPHARM, ZYDUSLIFE, DRREDDY

  [REGIME OFF] Nifty 200 4,179.2 < EMA200 4,258.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BRITANNIA   3      FMCG       01-Apr-15   946.0       1,364.1     112    ₹46,835       +44.2%      +4.4%     
  ASHOKLEY    4      AUTO       02-Feb-15   26.9        38.7        3716   ₹43,684       +43.7%      +5.4%     
  ABBOTINDIA  1      HEALTH     03-Aug-15   4,000.7     5,121.9     27     ₹30,274       +28.0%      +5.2%     
  AJANTPHARM  23     HEALTH     01-Apr-15   737.1       913.9       144    ₹25,462       +24.0%      +3.4%     
  SRF         66     MFG        02-Feb-15   181.9       215.0       549    ₹18,179       +18.2%      -1.2%     
  EMAMILTD    35     FMCG       01-Apr-15   420.8       496.1       252    ₹18,982       +17.9%      +0.6%     
  BAJFINANCE  25     FIN SVC    01-Jun-15   42.0        49.1        2479   ₹17,713       +17.0%      +1.3%     
  BBTC        49     FMCG       02-Feb-15   438.1       468.5       228    ₹6,939        +6.9%       +2.3%     
  IBULHSGFIN  5      FIN SVC    03-Aug-15   498.5       524.9       221    ₹5,833        +5.3%       +7.0%     
  MARUTI      6      AUTO       03-Aug-15   4,018.0     4,182.0     27     ₹4,429        +4.1%       +3.1%     
  RAJESHEXPO  2      CON DUR    03-Aug-15   561.3       548.4       196    ₹-2,516       -2.3%       +12.6%    
  EICHERMOT   65     AUTO       01-Jun-15   1,740.7     1,695.1     59     ₹-2,688       -2.6%       +0.2%     
  WHIRLPOOL   55     CON DUR    02-Feb-15   693.1       664.0       144    ₹-4,190       -4.2%       +4.3%     
  APLLTD      24     HEALTH     03-Aug-15   657.5       626.0       167    ₹-5,262       -4.8%       +2.5%     
  BAJAJFINSV  19     FIN SVC    03-Aug-15   189.4       175.2       581    ₹-8,229       -7.5%       -1.1%     
  MARICO      69     FMCG       01-Jun-15   190.3       170.9       546    ₹-10,596      -10.2%      -0.5%     
  HINDPETRO   31     OIL&GAS    03-Aug-15   81.6        71.8        1350   ₹-13,162      -11.9%      -1.0%     
  INDUSTOWER  102    INFRA      01-Jun-15   327.7       280.3       317    ₹-15,024      -14.5% ⚠    +1.8%     
  ⚠  WAZ < 0 (momentum below universe mean): INDUSTOWER

  AFTER: Invested ₹2,055,604 | Cash ₹46,216 | Total ₹2,101,820 | Positions 18/20 | Slot ₹105,091

========================================================================
  REBALANCE #06  —  01 Dec 2015
  NAV: ₹2,102,354  |  Slot: ₹105,118  |  Cash: ₹46,216
========================================================================

  [REGIME OFF] Nifty 200 4,198.9 < EMA200 4,245.7 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ASHOKLEY    17     AUTO       02-Feb-15   26.9        38.3        3716   ₹42,388       +42.4%      +0.3%     
  BRITANNIA   14     FMCG       01-Apr-15   946.0       1,298.0     112    ₹39,430       +37.2%      -0.8%     
  SRF         37     MFG        02-Feb-15   181.9       240.6       549    ₹32,266       +32.3%      -0.3%     
  BAJFINANCE  7      FIN SVC    01-Jun-15   42.0        53.9        2479   ₹29,548       +28.4%      +3.7%     
  RAJESHEXPO  1      CON DUR    03-Aug-15   561.3       698.5       196    ₹26,888       +24.4%      +4.8%     
  ABBOTINDIA  8      HEALTH     03-Aug-15   4,000.7     4,604.4     27     ₹16,302       +15.1%      -1.7%     
  AJANTPHARM  83     HEALTH     01-Apr-15   737.1       814.9       144    ₹11,207       +10.6%      -3.0%     
  BBTC        87     FMCG       02-Feb-15   438.1       475.3       228    ₹8,489        +8.5%       -0.8%     
  BAJAJFINSV  5      FIN SVC    03-Aug-15   189.4       198.0       581    ₹4,974        +4.5%       +2.7%     
  MARUTI      16     AUTO       03-Aug-15   4,018.0     4,159.9     27     ₹3,834        +3.5%       -0.5%     
  IBULHSGFIN  25     FIN SVC    03-Aug-15   498.5       480.1       221    ₹-4,081       -3.7%       +4.9%     
  HINDPETRO   22     OIL&GAS    03-Aug-15   81.6        78.4        1350   ₹-4,318       -3.9%       +6.0%     
  APLLTD      27     HEALTH     03-Aug-15   657.5       629.3       167    ₹-4,711       -4.3%       +3.8%     
  MARICO      38     FMCG       01-Jun-15   190.3       182.2       546    ₹-4,422       -4.3%       +2.7%     
  EMAMILTD    145    FMCG       01-Apr-15   420.8       400.9       252    ₹-5,021       -4.7% ⚠     -4.8%     
  WHIRLPOOL   112    CON DUR    02-Feb-15   693.1       656.8       144    ₹-5,215       -5.2% ⚠     +0.7%     
  EICHERMOT   137    AUTO       01-Jun-15   1,740.7     1,505.2     59     ₹-13,894      -13.5% ⚠    -2.6%     
  INDUSTOWER  86     INFRA      01-Jun-15   327.7       275.7       317    ₹-16,466      -15.9%      +0.8%     
  ⚠  WAZ < 0 (momentum below universe mean): WHIRLPOOL, EICHERMOT, EMAMILTD

  AFTER: Invested ₹2,056,138 | Cash ₹46,216 | Total ₹2,102,354 | Positions 18/20 | Slot ₹105,118

========================================================================
  REBALANCE #07  —  01 Feb 2016
  NAV: ₹2,046,378  |  Slot: ₹102,319  |  Cash: ₹46,216
========================================================================

  [REGIME OFF] Nifty 200 3,969.2 < EMA200 4,175.0 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ASHOKLEY    18     AUTO       02-Feb-15   26.9        37.5        3716   ₹39,492       +39.5%      +3.6%     
  BAJFINANCE  3      FIN SVC    01-Jun-15   42.0        57.8        2479   ₹39,072       +37.5%      +2.0%     
  BRITANNIA   36     FMCG       01-Apr-15   946.0       1,223.3     112    ₹31,066       +29.3%      +1.0%     
  RAJESHEXPO  1      CON DUR    03-Aug-15   561.3       720.9       196    ₹31,278       +28.4%      +2.5%     
  SRF         64     MFG        02-Feb-15   181.9       221.6       549    ₹21,788       +21.8%      -1.6%     
  ABBOTINDIA  40     HEALTH     03-Aug-15   4,000.7     4,529.7     27     ₹14,283       +13.2%      -3.2%     
  AJANTPHARM  65     HEALTH     01-Apr-15   737.1       786.7       144    ₹7,135        +6.7%       +4.6%     
  EMAMILTD    44     FMCG       01-Apr-15   420.8       437.8       252    ₹4,268        +4.0%       +3.4%     
  MARICO      6      FMCG       01-Jun-15   190.3       194.1       546    ₹2,036        +2.0%       +2.2%     
  BAJAJFINSV  24     FIN SVC    03-Aug-15   189.4       185.3       581    ₹-2,397       -2.2%       -1.9%     
  IBULHSGFIN  26     FIN SVC    03-Aug-15   498.5       480.1       221    ₹-4,082       -3.7%       +1.6%     
  BBTC        110    FMCG       02-Feb-15   438.1       406.7       228    ₹-7,168       -7.2% ⚠     -2.2%     
  WHIRLPOOL   63     CON DUR    02-Feb-15   693.1       636.4       144    ₹-8,162       -8.2%       +4.5%     
  HINDPETRO   13     OIL&GAS    03-Aug-15   81.6        74.6        1350   ₹-9,402       -8.5%       -2.4%     
  EICHERMOT   50     AUTO       01-Jun-15   1,740.7     1,582.7     59     ₹-9,320       -9.1%       +3.8%     
  MARUTI      97     AUTO       03-Aug-15   4,018.0     3,604.9     27     ₹-11,153      -10.3% ⚠    -6.0%     
  APLLTD      38     HEALTH     03-Aug-15   657.5       560.0       167    ₹-16,286      -14.8%      -1.1%     
  INDUSTOWER  69     INFRA      01-Jun-15   327.7       260.7       317    ₹-21,228      -20.4%      -3.0%     
  ⚠  WAZ < 0 (momentum below universe mean): MARUTI, BBTC

  AFTER: Invested ₹2,000,161 | Cash ₹46,216 | Total ₹2,046,378 | Positions 18/20 | Slot ₹102,319

========================================================================
  REBALANCE #08  —  01 Apr 2016
  NAV: ₹2,066,671  |  Slot: ₹103,334  |  Cash: ₹46,216
========================================================================

  [REGIME OFF] Nifty 200 4,043.2 < EMA200 4,072.6 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ASHOKLEY    5      AUTO       02-Feb-15   26.9        45.2        3716   ₹68,074       +68.1%      +9.3%     
  BAJFINANCE  3      FIN SVC    01-Jun-15   42.0        66.3        2479   ₹60,381       +58.0%      +4.6%     
  SRF         27     MFG        02-Feb-15   181.9       244.6       549    ₹34,420       +34.5%      +4.6%     
  BRITANNIA   61     FMCG       01-Apr-15   946.0       1,170.3     112    ₹25,124       +23.7%      -1.3%     
  AJANTPHARM  29     HEALTH     01-Apr-15   737.1       847.5       144    ₹15,902       +15.0%      +1.7%     
  MARICO      16     FMCG       01-Jun-15   190.3       210.1       546    ₹10,776       +10.4%      +0.4%     
  RAJESHEXPO  4      CON DUR    03-Aug-15   561.3       615.3       196    ₹10,585       +9.6%       -1.8%     
  ABBOTINDIA  115    HEALTH     03-Aug-15   4,000.7     4,152.0     27     ₹4,086        +3.8% ⚠     -2.0%     
  EICHERMOT   13     AUTO       01-Jun-15   1,740.7     1,794.6     59     ₹3,181        +3.1%       +2.5%     
  WHIRLPOOL   39     CON DUR    02-Feb-15   693.1       681.6       144    ₹-1,642       -1.6%       +5.2%     
  EMAMILTD    106    FMCG       01-Apr-15   420.8       391.6       252    ₹-7,353       -6.9% ⚠     -2.4%     
  HINDPETRO   34     OIL&GAS    03-Aug-15   81.6        75.4        1350   ₹-8,415       -7.6%       +5.7%     
  BAJAJFINSV  52     FIN SVC    03-Aug-15   189.4       173.9       581    ₹-9,038       -8.2%       +3.5%     
  IBULHSGFIN  65     FIN SVC    03-Aug-15   498.5       443.6       221    ₹-12,136      -11.0%      +1.7%     
  MARUTI      146    AUTO       03-Aug-15   4,018.0     3,399.4     27     ₹-16,701      -15.4% ⚠    +1.5%     
  BBTC        145    FMCG       02-Feb-15   438.1       365.3       228    ₹-16,602      -16.6% ⚠    +2.2%     
  INDUSTOWER  91     INFRA      01-Jun-15   327.7       271.9       317    ₹-17,662      -17.0% ⚠    +2.0%     
  APLLTD      75     HEALTH     03-Aug-15   657.5       529.0       167    ₹-21,466      -19.5%      -3.0%     
  ⚠  WAZ < 0 (momentum below universe mean): INDUSTOWER, EMAMILTD, ABBOTINDIA, BBTC, MARUTI

  AFTER: Invested ₹2,020,455 | Cash ₹46,216 | Total ₹2,066,671 | Positions 18/20 | Slot ₹103,334

========================================================================
  REBALANCE #09  —  01 Jun 2016
  NAV: ₹2,138,588  |  Slot: ₹106,929  |  Cash: ₹46,216
========================================================================
  [SECTOR CAP≤4] dropped: BERGEPAINT

  EXITS (11)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BRITANNIA   72     FMCG       01-Apr-15   946.0       1,207.9     112    ₹29,334       +27.7%    427d  
  AJANTPHARM  48     HEALTH     01-Apr-15   737.1       930.8       144    ₹27,887       +26.3%    427d  
  MARICO      57     FMCG       01-Jun-15   190.3       218.4       546    ₹15,302       +14.7%    366d  
  EMAMILTD    —      FMCG       01-Apr-15   420.8       437.7       252    ₹4,257        +4.0%     427d  
  ABBOTINDIA  66     HEALTH     03-Aug-15   4,000.7     4,141.9     27     ₹3,814        +3.5%     303d  
  EICHERMOT   88     AUTO       01-Jun-15   1,740.7     1,744.2     59     ₹207          +0.2%     366d  
  IBULHSGFIN  44     FIN SVC    03-Aug-15   498.5       498.5       221    ₹-13          -0.0%     303d  
  BAJAJFINSV  43     FIN SVC    03-Aug-15   189.4       179.0       581    ₹-6,043       -5.5%     303d  
  BBTC        78     FMCG       02-Feb-15   438.1       370.2       228    ₹-15,494      -15.5%    485d  
  INDUSTOWER  85     INFRA      01-Jun-15   327.7       271.5       317    ₹-17,808      -17.1%    366d  
  APLLTD      101    HEALTH     03-Aug-15   657.5       485.3       167    ₹-28,762      -26.2%    303d  

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BIOCON      1      HEALTH     3.831    0.19   +60.9%    +57.9%    117.0       913    ₹106,861      +10.2%    
  VGUARD      2      CON DUR    3.607    0.26   +46.9%    +66.5%    93.1        1148   ₹106,900      +13.7%    
  RAMCOCEM    3      MFG        2.950    0.28   +62.3%    +35.7%    473.9       225    ₹106,618      +0.7%     
  INDUSINDBK  5      PVT BNK    2.645    0.41   +29.8%    +36.6%    1,042.1     102    ₹106,290      +2.9%     
  HDFC        7      PVT BNK    2.391    0.28   +15.7%    +24.7%    265.8       402    ₹106,836      +1.9%     
  HDFCBANK    8      PVT BNK    2.391    0.28   +15.7%    +24.7%    265.8       402    ₹106,836      +1.9%     
  SHRIRAMFIN  9      FIN SVC    2.237    0.68   +41.9%    +46.0%    197.6       541    ₹106,910      +3.9%     
  MUTHOOTFIN  10     FIN SVC    2.142    0.24   +35.5%    +40.6%    205.5       520    ₹106,844      +12.8%    
  WELSPUNIND  11     CONSUMP    2.085    0.54   +94.1%    +26.3%    97.9        1092   ₹106,857      +6.1%     
  PIDILITIND  12     MFG        2.056    0.41   +31.7%    +20.3%    342.9       311    ₹106,632      +9.8%     
  HAVELLS     13     CON DUR    2.052    0.46   +36.4%    +31.7%    331.1       322    ₹106,605      -0.7%     

  HOLDS (7)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  4      FIN SVC    01-Jun-15   42.0        73.7        2479   ₹78,674       +75.6%      +1.7%     
  ASHOKLEY    15     AUTO       02-Feb-15   26.9        42.8        3716   ₹59,233       +59.2%      +0.5%     
  SRF         33     MFG        02-Feb-15   181.9       246.2       549    ₹35,307       +35.4%      +0.5%     
  WHIRLPOOL   38     CON DUR    02-Feb-15   693.1       748.2       144    ₹7,945        +8.0%       +1.8%     
  HINDPETRO   6      OIL&GAS    03-Aug-15   81.6        86.0        1350   ₹5,932        +5.4%       +5.4%     
  RAJESHEXPO  19     CON DUR    03-Aug-15   561.3       558.5       196    ₹-538         -0.5%       -0.4%     
  MARUTI      27     AUTO       03-Aug-15   4,018.0     3,803.1     27     ₹-5,802       -5.3%       +4.7%     

  AFTER: Invested ₹2,087,314 | Cash ₹49,881 | Total ₹2,137,194 | Positions 18/20 | Slot ₹106,929

========================================================================
  REBALANCE #10  —  01 Aug 2016
  NAV: ₹2,442,544  |  Slot: ₹122,127  |  Cash: ₹49,881
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SRF         47     MFG        02-Feb-15   181.9       289.2       549    ₹58,919       +59.0%    546d  
  ASHOKLEY    94     AUTO       02-Feb-15   26.9        38.0        3716   ₹41,076       +41.1%    546d  
  WHIRLPOOL   37     CON DUR    02-Feb-15   693.1       833.3       144    ₹20,198       +20.2%    546d  
  HDFCBANK    27     PVT BNK    01-Jun-16   265.8       283.3       402    ₹7,032        +6.6%     61d   
  WELSPUNIND  43     CONSUMP    01-Jun-16   97.9        98.3        1092   ₹505          +0.5%     61d   
  RAJESHEXPO  108    CON DUR    03-Aug-15   561.3       436.5       196    ₹-24,454      -22.2%    364d  

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BAJAJFINSV  4      FIN SVC    3.084    0.50   +68.9%    +45.2%    281.5       433    ₹121,911      +12.3%    
  ASIANPAINT  7      CONSUMP    2.368    0.39   +34.0%    +29.2%    1,030.0     118    ₹121,540      +6.5%     
  SHREECEM    10     INFRA      2.284    0.74   +51.9%    +31.1%    15,652.0    7      ₹109,564      +3.3%     
  BERGEPAINT  11     CON DUR    2.124    0.44   +53.4%    +24.9%    186.1       656    ₹122,081      +3.1%     
  SUNTV       12     MEDIA      2.115    0.71   +88.3%    +29.8%    367.7       332    ₹122,085      +14.6%    
  PETRONET    13     OIL&GAS    2.071    0.48   +61.4%    +14.6%    106.3       1148   ₹122,055      +5.9%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  1      FIN SVC    01-Jun-15   42.0        108.4       2479   ₹164,601      +158.1%     +22.0%    
  HINDPETRO   6      OIL&GAS    03-Aug-15   81.6        121.9       1350   ₹54,469       +49.4%      +12.3%    
  MUTHOOTFIN  3      FIN SVC    01-Jun-16   205.5       298.8       520    ₹48,548       +45.4%      +20.6%    
  VGUARD      2      CON DUR    01-Jun-16   93.1        113.9       1148   ₹23,804       +22.3%      +11.9%    
  BIOCON      5      HEALTH     01-Jun-16   117.0       135.3       913    ₹16,682       +15.6%      +7.7%     
  RAMCOCEM    8      MFG        01-Jun-16   473.9       544.0       225    ₹15,777       +14.8%      +0.2%     
  HAVELLS     18     CON DUR    01-Jun-16   331.1       378.3       322    ₹15,205       +14.3%      +8.5%     
  SHRIRAMFIN  19     FIN SVC    01-Jun-16   197.6       221.8       541    ₹13,088       +12.2%      +5.8%     
  MARUTI      22     AUTO       03-Aug-15   4,018.0     4,443.0     27     ₹11,477       +10.6%      +8.6%     
  INDUSINDBK  21     PVT BNK    01-Jun-16   1,042.1     1,138.1     102    ₹9,801        +9.2%       +5.1%     
  HDFC        26     PVT BNK    01-Jun-16   265.8       283.3       402    ₹7,032        +6.6%       +1.9%     
  PIDILITIND  9      MFG        01-Jun-16   342.9       350.4       311    ₹2,340        +2.2%       +1.2%     

  AFTER: Invested ₹2,385,288 | Cash ₹56,402 | Total ₹2,441,690 | Positions 18/20 | Slot ₹122,127

========================================================================
  REBALANCE #11  —  03 Oct 2016
  NAV: ₹2,592,809  |  Slot: ₹129,640  |  Cash: ₹56,402
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  INDUSINDBK  41     PVT BNK    01-Jun-16   1,042.1     1,168.8     102    ₹12,929       +12.2%    124d  
  HDFC        37     PVT BNK    01-Jun-16   265.8       293.5       402    ₹11,147       +10.4%    124d  
  SHREECEM    32     INFRA      01-Aug-16   15,652.0    17,211.1    7      ₹10,914       +10.0%    63d   
  SHRIRAMFIN  69     FIN SVC    01-Jun-16   197.6       208.2       541    ₹5,751        +5.4%     124d  
  PIDILITIND  61     MFG        01-Jun-16   342.9       346.3       311    ₹1,057        +1.0%     124d  

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HINDZINC    2      METAL      2.887    0.36   +120.3%   +40.9%    113.8       1139   ₹129,594      +11.7%    
  MRF         5      MFG        2.569    0.19   +25.7%    +57.3%    51,226.4    2      ₹102,453      +16.3%    
  IOC         6      OIL&GAS    2.485    0.08   +60.9%    +38.9%    53.5        2422   ₹129,629      +4.8%     
  CHOLAFIN    8      FIN SVC    2.432    0.05   +96.7%    +24.8%    228.4       567    ₹129,490      +5.2%     
  SRF         9      MFG        2.293    0.24   +68.7%    +45.9%    366.2       354    ₹129,629      +10.8%    

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  3      FIN SVC    01-Jun-15   42.0        104.9       2479   ₹156,002      +149.9%     +0.1%     
  HINDPETRO   15     OIL&GAS    03-Aug-15   81.6        124.8       1350   ₹58,299       +52.9%      +4.3%     
  MUTHOOTFIN  16     FIN SVC    01-Jun-16   205.5       295.2       520    ₹46,646       +43.7%      -1.6%     
  VGUARD      4      CON DUR    01-Jun-16   93.1        127.4       1148   ₹39,310       +36.8%      +0.4%     
  BIOCON      1      HEALTH     01-Jun-16   117.0       157.9       913    ₹37,313       +34.9%      +3.1%     
  MARUTI      19     AUTO       03-Aug-15   4,018.0     5,223.0     27     ₹32,536       +30.0%      +4.8%     
  RAMCOCEM    14     MFG        01-Jun-16   473.9       596.8       225    ₹27,669       +26.0%      +4.3%     
  HAVELLS     21     CON DUR    01-Jun-16   331.1       403.7       322    ₹23,374       +21.9%      +4.1%     
  PETRONET    10     OIL&GAS    01-Aug-16   106.3       121.2       1148   ₹17,031       +14.0%      +3.5%     
  BAJAJFINSV  7      FIN SVC    01-Aug-16   281.5       318.0       433    ₹15,769       +12.9%      +3.5%     
  BERGEPAINT  13     CON DUR    01-Aug-16   186.1       209.6       656    ₹15,400       +12.6%      +1.5%     
  SUNTV       22     MEDIA      01-Aug-16   367.7       412.8       332    ₹14,977       +12.3%      +8.0%     
  ASIANPAINT  24     CONSUMP    01-Aug-16   1,030.0     1,096.3     118    ₹7,818        +6.4%       +1.9%     

  AFTER: Invested ₹2,579,173 | Cash ₹12,900 | Total ₹2,592,072 | Positions 18/20 | Slot ₹129,640

========================================================================
  REBALANCE #12  —  01 Dec 2016
  NAV: ₹2,370,344  |  Slot: ₹118,517  |  Cash: ₹12,900
========================================================================

  [REGIME OFF] Nifty 200 4,386.1 < EMA200 4,400.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  30     FIN SVC    01-Jun-15   42.0        88.3        2479   ₹114,715      +110.2%     -0.3%     
  HINDPETRO   8      OIL&GAS    03-Aug-15   81.6        129.0       1350   ₹63,946       +58.1%      -1.9%     
  BIOCON      4      HEALTH     01-Jun-16   117.0       151.6       913    ₹31,510       +29.5%      +3.4%     
  VGUARD      12     CON DUR    01-Jun-16   93.1        116.3       1148   ₹26,586       +24.9%      -6.0%     
  MUTHOOTFIN  24     FIN SVC    01-Jun-16   205.5       253.8       520    ₹25,137       +23.5%      -3.8%     
  PETRONET    7      OIL&GAS    01-Aug-16   106.3       130.9       1148   ₹28,239       +23.1%      +2.5%     
  RAMCOCEM    10     MFG        01-Jun-16   473.9       581.0       225    ₹24,101       +22.6%      +2.6%     
  MARUTI      27     AUTO       03-Aug-15   4,018.0     4,828.7     27     ₹21,889       +20.2%      +1.7%     
  HINDZINC    1      METAL      03-Oct-16   113.8       123.8       1139   ₹11,469       +8.9%       +4.4%     
  BAJAJFINSV  18     FIN SVC    01-Aug-16   281.5       292.6       433    ₹4,764        +3.9%       -2.1%     
  BERGEPAINT  32     CON DUR    01-Aug-16   186.1       186.0       656    ₹-78          -0.1%       +4.3%     
  IOC         15     OIL&GAS    03-Oct-16   53.5        52.8        2422   ₹-1,718       -1.3%       -1.5%     
  SUNTV       38     MEDIA      01-Aug-16   367.7       358.9       332    ₹-2,941       -2.4%       -3.5%     
  HAVELLS     83     CON DUR    01-Jun-16   331.1       313.5       322    ₹-5,669       -5.3% ⚠     -4.2%     
  MRF         5      MFG        03-Oct-16   51,226.4    48,256.4    2      ₹-5,940       -5.8%       -0.3%     
  ASIANPAINT  84     CONSUMP    01-Aug-16   1,030.0     865.1       118    ₹-19,454      -16.0% ⚠    -3.7%     
  SRF         41     MFG        03-Oct-16   366.2       303.7       354    ₹-22,120      -17.1%      -0.2%     
  CHOLAFIN    42     FIN SVC    03-Oct-16   228.4       186.0       567    ₹-24,022      -18.6%      -5.0%     
  ⚠  WAZ < 0 (momentum below universe mean): HAVELLS, ASIANPAINT

  AFTER: Invested ₹2,357,444 | Cash ₹12,900 | Total ₹2,370,344 | Positions 18/20 | Slot ₹118,517

========================================================================
  REBALANCE #13  —  01 Feb 2017
  NAV: ₹2,678,638  |  Slot: ₹133,932  |  Cash: ₹12,900
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MUTHOOTFIN  35     FIN SVC    01-Jun-16   205.5       272.1       520    ₹34,668       +32.4%    245d  
  HAVELLS     30     CON DUR    01-Jun-16   331.1       399.7       322    ₹22,108       +20.7%    245d  
  SUNTV       32     MEDIA      01-Aug-16   367.7       427.7       332    ₹19,918       +16.3%    184d  
  BERGEPAINT  97     CON DUR    01-Aug-16   186.1       169.1       656    ₹-11,179      -9.2%     184d  
  SRF         37     MFG        03-Oct-16   366.2       326.3       354    ₹-14,132      -10.9%    121d  
  ASIANPAINT  78     CONSUMP    01-Aug-16   1,030.0     912.9       118    ₹-13,818      -11.4%    184d  
  CHOLAFIN    50     FIN SVC    03-Oct-16   228.4       198.7       567    ₹-16,802      -13.0%    121d  

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  VAKRANGEE   4      IT         2.905    -0.03  +83.6%    +24.3%    137.7       972    ₹133,804      +4.5%     
  POWERGRID   5      ENERGY     2.888    0.34   +56.8%    +17.5%    73.8        1815   ₹133,885      +3.7%     
  IGL         8      OIL&GAS    2.598    0.44   +70.6%    +12.3%    84.2        1591   ₹133,914      +3.1%     
  BEL         9      DEFENCE    2.355    0.63   +33.2%    +20.5%    40.0        3348   ₹133,893      +4.1%     
  RECLTD      10     FIN SVC    2.354    0.84   +77.4%    +15.8%    54.0        2481   ₹133,884      +6.0%     
  PFC         12     FIN SVC    2.197    0.70   +76.0%    +14.8%    62.2        2152   ₹133,906      +2.1%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  21     FIN SVC    01-Jun-15   42.0        102.9       2479   ₹151,084      +145.1%     +12.1%    
  HINDPETRO   2      OIL&GAS    03-Aug-15   81.6        155.5       1350   ₹99,721       +90.5%      +7.6%     
  VGUARD      11     CON DUR    01-Jun-16   93.1        141.1       1148   ₹55,117       +51.6%      +12.8%    
  RAMCOCEM    7      MFG        01-Jun-16   473.9       693.0       225    ₹49,316       +46.3%      +13.1%    
  BIOCON      6      HEALTH     01-Jun-16   117.0       166.6       913    ₹45,261       +42.4%      +2.4%     
  MARUTI      19     AUTO       03-Aug-15   4,018.0     5,681.0     27     ₹44,903       +41.4%      +7.8%     
  PETRONET    23     OIL&GAS    01-Aug-16   106.3       132.7       1148   ₹30,308       +24.8%      +3.6%     
  IOC         3      OIL&GAS    03-Oct-16   53.5        66.6        2422   ₹31,607       +24.4%      +5.1%     
  HINDZINC    1      METAL      03-Oct-16   113.8       138.1       1139   ₹27,730       +21.4%      +6.3%     
  BAJAJFINSV  24     FIN SVC    01-Aug-16   281.5       331.3       433    ₹21,533       +17.7%      +7.6%     
  MRF         25     MFG        03-Oct-16   51,226.4    51,910.7    2      ₹1,369        +1.3%       +0.7%     

  AFTER: Invested ₹2,609,989 | Cash ₹67,695 | Total ₹2,677,684 | Positions 17/20 | Slot ₹133,932

========================================================================
  REBALANCE #14  —  03 Apr 2017
  NAV: ₹2,890,954  |  Slot: ₹144,548  |  Cash: ₹67,695
========================================================================
  [SECTOR CAP≤4] dropped: DHFL

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJFINANCE  30     FIN SVC    01-Jun-15   42.0        114.1       2479   ₹178,752      +171.7%   672d  
  MARUTI      37     AUTO       03-Aug-15   4,018.0     5,582.5     27     ₹42,242       +38.9%    609d  
  RAMCOCEM    39     MFG        01-Jun-16   473.9       654.3       225    ₹40,598       +38.1%    306d  
  PETRONET    44     OIL&GAS    01-Aug-16   106.3       142.1       1148   ₹41,023       +33.6%    245d  
  PFC         33     FIN SVC    01-Feb-17   62.2        70.1        2152   ₹17,020       +12.7%    61d   
  BEL         45     DEFENCE    01-Feb-17   40.0        41.1        3348   ₹3,655        +2.7%     61d   
  POWERGRID   46     ENERGY     01-Feb-17   73.8        70.7        1815   ₹-5,609       -4.2%     61d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  EDELWEISS   4      FIN SVC    2.685    1.18   +176.5%   +67.6%    73.6        1963   ₹144,506      +12.3%    
  IBULHSGFIN  5      FIN SVC    2.523    0.30   +63.1%    +54.9%    707.9       204    ₹144,405      +6.5%     
  DALMIABHA   6      MFG        2.502    0.29   +147.5%   +45.8%    1,969.1     73     ₹143,743      +2.4%     
  BBTC        7      FMCG       2.372    1.19   +122.7%   +65.1%    828.5       174    ₹144,163      +6.8%     
  SUNTV       8      MEDIA      2.335    1.02   +117.6%   +63.7%    626.4       230    ₹144,079      +5.2%     
  GUJGASLTD   10     OIL&GAS    2.305    0.42   +44.0%    +44.8%    142.3       1015   ₹144,481      +4.2%     
  NATCOPHARM  12     HEALTH     2.250    0.28   +85.9%    +50.4%    803.3       179    ₹143,785      +8.1%     
  KARURVYSYA  13     PVT BNK    2.245    0.36   +36.7%    +42.4%    73.3        1971   ₹144,538      +10.0%    

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDPETRO   15     OIL&GAS    03-Aug-15   81.6        158.1       1350   ₹103,272      +93.8%      +0.7%     
  VGUARD      1      CON DUR    01-Jun-16   93.1        168.4       1148   ₹86,453       +80.9%      +2.5%     
  BIOCON      9      HEALTH     01-Jun-16   117.0       185.3       913    ₹62,310       +58.3%      +0.9%     
  BAJAJFINSV  3      FIN SVC    01-Aug-16   281.5       407.9       433    ₹54,712       +44.9%      +2.7%     
  IOC         19     OIL&GAS    03-Oct-16   53.5        70.8        2422   ₹41,884       +32.3%      +2.1%     
  RECLTD      2      FIN SVC    01-Feb-17   54.0        68.6        2481   ₹36,290       +27.1%      +6.9%     
  HINDZINC    21     METAL      03-Oct-16   113.8       143.4       1139   ₹33,765       +26.1%      +2.0%     
  MRF         32     MFG        03-Oct-16   51,226.4    60,214.9    2      ₹17,977       +17.5%      +6.5%     
  VAKRANGEE   27     IT         01-Feb-17   137.7       147.3       972    ₹9,397        +7.0%       +2.1%     
  IGL         28     OIL&GAS    01-Feb-17   84.2        88.9        1591   ₹7,476        +5.6%       -0.4%     

  AFTER: Invested ₹2,816,339 | Cash ₹73,245 | Total ₹2,889,584 | Positions 18/20 | Slot ₹144,548

========================================================================
  REBALANCE #15  —  01 Jun 2017
  NAV: ₹3,022,627  |  Slot: ₹151,131  |  Cash: ₹73,245
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BIOCON      102    HEALTH     01-Jun-16   117.0       156.7       913    ₹36,236       +33.9%    365d  
  RECLTD      3      FIN SVC    01-Feb-17   54.0        72.3        2481   ₹45,376       +33.9%    120d  
  MRF         —      MFG        03-Oct-16   51,226.4    67,346.4    2      ₹32,240       +31.5%    241d  
  EDELWEISS   20     FIN SVC    03-Apr-17   73.6        80.6        1963   ₹13,777       +9.5%     59d   
  BBTC        12     FMCG       03-Apr-17   828.5       871.8       174    ₹7,531        +5.2%     59d   
  HINDZINC    97     METAL      03-Oct-16   113.8       117.4       1139   ₹4,129        +3.2%     241d  
  SUNTV       36     MEDIA      03-Apr-17   626.4       633.1       230    ₹1,531        +1.1%     59d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIENSOL  1      ENERGY     4.624    1.04   +324.2%   +105.6%   127.1       1189   ₹151,062      +22.7%    
  HDFC        4      PVT BNK    2.393    0.56   +39.9%    +17.2%    371.3       406    ₹150,761      +2.9%     
  HDFCBANK    5      PVT BNK    2.393    0.56   +39.9%    +17.2%    371.3       406    ₹150,761      +2.9%     
  TVSMOTOR    6      AUTO       2.315    0.80   +81.3%    +24.9%    508.7       297    ₹151,077      +2.1%     
  MARUTI      7      AUTO       2.289    1.00   +77.2%    +20.7%    6,570.4     23     ₹151,120      +3.7%     
  HINDUNILVR  8      FMCG       2.255    0.50   +32.9%    +26.5%    940.4       160    ₹150,472      +7.3%     
  DHFL        9      FIN SVC    2.186    0.63   +114.3%   +28.8%    408.8       369    ₹150,841      +0.4%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDPETRO   35     OIL&GAS    03-Aug-15   81.6        165.0       1350   ₹112,573      +102.2%     +1.0%     
  VGUARD      23     CON DUR    01-Jun-16   93.1        176.8       1148   ₹96,024       +89.8%      -3.8%     
  BAJAJFINSV  21     FIN SVC    01-Aug-16   281.5       418.6       433    ₹59,338       +48.7%      -0.1%     
  IOC         16     OIL&GAS    03-Oct-16   53.5        76.9        2422   ₹56,599       +43.7%      -3.3%     
  VAKRANGEE   2      IT         01-Feb-17   137.7       174.1       972    ₹35,436       +26.5%      +9.6%     
  DALMIABHA   5      MFG        03-Apr-17   1,969.1     2,419.4     73     ₹32,871       +22.9%      +1.8%     
  IBULHSGFIN  14     FIN SVC    03-Apr-17   707.9       817.2       204    ₹22,314       +15.5%      +6.4%     
  IGL         26     OIL&GAS    01-Feb-17   84.2        93.2        1591   ₹14,339       +10.7%      +3.7%     
  NATCOPHARM  17     HEALTH     03-Apr-17   803.3       877.3       179    ₹13,243       +9.2%       +3.1%     
  KARURVYSYA  44     PVT BNK    03-Apr-17   73.3        74.6        1971   ₹2,544        +1.8%       +0.9%     
  GUJGASLTD   25     OIL&GAS    03-Apr-17   142.3       142.8       1015   ₹477          +0.3%       -1.6%     

  AFTER: Invested ₹2,959,116 | Cash ₹62,257 | Total ₹3,021,373 | Positions 18/20 | Slot ₹151,131

========================================================================
  REBALANCE #16  —  01 Aug 2017
  NAV: ₹3,250,300  |  Slot: ₹162,515  |  Cash: ₹62,257
========================================================================
  [SECTOR CAP≤4] dropped: KOTAKBANK

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HINDPETRO   40     OIL&GAS    03-Aug-15   81.6        177.2       1350   ₹129,106      +117.2%   729d  
  VGUARD      49     CON DUR    01-Jun-16   93.1        174.9       1148   ₹93,921       +87.9%    426d  
  BAJAJFINSV  19     FIN SVC    01-Aug-16   281.5       504.6       433    ₹96,565       +79.2%    365d  
  IOC         91     OIL&GAS    03-Oct-16   53.5        68.8        2422   ₹37,025       +28.6%    302d  
  GUJGASLTD   84     OIL&GAS    03-Apr-17   142.3       142.6       1015   ₹296          +0.2%     120d  
  ADANIENSOL  2      ENERGY     01-Jun-17   127.1       125.8       1189   ₹-1,486       -1.0%     61d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RAJESHEXPO  6      CON DUR    2.610    0.79   +64.8%    +16.7%    714.7       227    ₹162,232      +4.4%     
  RELIANCE    9      OIL&GAS    2.205    0.71   +58.0%    +18.0%    352.5       460    ₹162,173      +3.9%     
  INDUSINDBK  10     PVT BNK    2.022    1.14   +44.6%    +16.2%    1,585.1     102    ₹161,684      +5.7%     
  PGHH        11     FMCG       1.963    0.36   +33.7%    +15.5%    7,228.0     22     ₹159,016      +1.0%     
  GODREJIND   13     CONSUMP    1.848    0.95   +49.3%    +18.0%    649.5       250    ₹162,369      -1.3%     
  UPL         18     MFG        1.635    1.13   +55.5%    +10.9%    548.9       296    ₹162,488      +3.3%     
  HONAUT      19     MFG        1.633    0.51   +24.8%    +12.5%    12,237.3    13     ₹159,085      +2.6%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VAKRANGEE   1      IT         01-Feb-17   137.7       199.9       972    ₹60,474       +45.2%      +1.3%     
  DALMIABHA   28     MFG        03-Apr-17   1,969.1     2,588.4     73     ₹45,209       +31.5%      -1.1%     
  IGL         10     OIL&GAS    01-Feb-17   84.2        104.2       1591   ₹31,904       +23.8%      +4.8%     
  IBULHSGFIN  22     FIN SVC    03-Apr-17   707.9       876.7       204    ₹34,433       +23.8%      +6.3%     
  KARURVYSYA  26     PVT BNK    03-Apr-17   73.3        90.0        1971   ₹32,915       +22.8%      +1.0%     
  DHFL        25     FIN SVC    01-Jun-17   408.8       459.5       369    ₹18,726       +12.4%      +2.8%     
  NATCOPHARM  33     HEALTH     03-Apr-17   803.3       899.3       179    ₹17,192       +12.0%      -1.8%     
  TVSMOTOR    5      AUTO       01-Jun-17   508.7       568.4       297    ₹17,733       +11.7%      +4.4%     
  HDFC        3      PVT BNK    01-Jun-17   371.3       412.5       406    ₹16,713       +11.1%      +4.2%     
  HDFCBANK    4      PVT BNK    01-Jun-17   371.3       412.5       406    ₹16,713       +11.1%      +4.2%     
  MARUTI      9      AUTO       01-Jun-17   6,570.4     7,222.7     23     ₹15,003       +9.9%       +4.1%     
  HINDUNILVR  7      FMCG       01-Jun-17   940.4       1,017.0     160    ₹12,246       +8.1%       +2.8%     

  AFTER: Invested ₹3,197,527 | Cash ₹51,433 | Total ₹3,248,960 | Positions 19/20 | Slot ₹162,515

========================================================================
  REBALANCE #17  —  03 Oct 2017
  NAV: ₹3,390,582  |  Slot: ₹169,529  |  Cash: ₹51,433
========================================================================
  [SECTOR CAP≤4] dropped: GUJGASLTD

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MARUTI      29     AUTO       01-Jun-17   6,570.4     7,324.9     23     ₹17,352       +11.5%    124d  
  NATCOPHARM  105    HEALTH     03-Apr-17   803.3       738.5       179    ₹-11,598      -8.1%     183d  
  GODREJIND   74     CONSUMP    01-Aug-17   649.5       588.7       250    ₹-15,195      -9.4%     63d   
  UPL         91     MFG        01-Aug-17   548.9       488.7       296    ₹-17,828      -11.0%    63d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  GAIL        9      OIL&GAS    2.200    0.84   +58.0%    +19.5%    76.7        2209   ₹169,458      +8.2%     
  MGL         10     OIL&GAS    2.044    0.75   +77.3%    +10.1%    877.9       193    ₹169,443      -0.5%     
  BRITANNIA   12     FMCG       1.983    0.88   +27.7%    +17.8%    1,920.1     88     ₹168,966      +0.9%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VAKRANGEE   6      IT         01-Feb-17   137.7       222.4       972    ₹82,414       +61.6%      +0.6%     
  IGL         1      OIL&GAS    01-Feb-17   84.2        129.7       1591   ₹72,418       +54.1%      +4.3%     
  DALMIABHA   42     MFG        03-Apr-17   1,969.1     2,715.0     73     ₹54,452       +37.9%      +1.4%     
  DHFL        15     FIN SVC    01-Jun-17   408.8       543.3       369    ₹49,639       +32.9%      +0.8%     
  HONAUT      5      MFG        01-Aug-17   12,237.3    15,425.8    13     ₹41,450       +26.1%      +3.6%     
  IBULHSGFIN  25     FIN SVC    03-Apr-17   707.9       890.5       204    ₹37,259       +25.8%      -0.5%     
  KARURVYSYA  46     PVT BNK    03-Apr-17   73.3        91.7        1971   ₹36,265       +25.1%      -3.0%     
  TVSMOTOR    9      AUTO       01-Jun-17   508.7       622.8       297    ₹33,895       +22.4%      +2.2%     
  RAJESHEXPO  3      CON DUR    01-Aug-17   714.7       815.3       227    ₹22,847       +14.1%      +6.7%     
  HDFC        10     PVT BNK    01-Jun-17   371.3       415.2       406    ₹17,799       +11.8%      +0.2%     
  HDFCBANK    11     PVT BNK    01-Jun-17   371.3       415.2       406    ₹17,799       +11.8%      +0.2%     
  HINDUNILVR  27     FMCG       01-Jun-17   940.4       1,027.7     160    ₹13,965       +9.3%       -2.7%     
  PGHH        37     FMCG       01-Aug-17   7,228.0     7,500.9     22     ₹6,004        +3.8%       +0.8%     
  INDUSINDBK  22     PVT BNK    01-Aug-17   1,585.1     1,611.1     102    ₹2,645        +1.6%       -0.2%     
  RELIANCE    18     OIL&GAS    01-Aug-17   352.5       351.0       460    ₹-698         -0.4%       -1.7%     

  AFTER: Invested ₹3,254,523 | Cash ₹135,456 | Total ₹3,389,979 | Positions 18/20 | Slot ₹169,529

========================================================================
  REBALANCE #18  —  01 Dec 2017
  NAV: ₹3,623,890  |  Slot: ₹181,195  |  Cash: ₹135,456
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  IBULHSGFIN  72     FIN SVC    03-Apr-17   707.9       831.7       204    ₹25,253       +17.5%    242d  
  KARURVYSYA  96     PVT BNK    03-Apr-17   73.3        80.2        1971   ₹13,633       +9.4%     242d  
  MGL         51     OIL&GAS    03-Oct-17   877.9       888.1       193    ₹1,956        +1.2%     59d   
  INDUSINDBK  48     PVT BNK    01-Aug-17   1,585.1     1,580.8     102    ₹-446         -0.3%     122d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BBTC        2      FMCG       3.353    0.99   +197.5%   +48.2%    1,478.3     122    ₹180,358      -0.9%     
  FRETAIL     3      CONSUMP    2.538    1.11   +348.9%   +1.6%     544.8       332    ₹180,857      +2.1%     
  BALKRISIND  7      MFG        2.244    0.53   +130.1%   +28.2%    978.3       185    ₹180,990      +4.4%     
  MARUTI      10     AUTO       2.111    1.09   +70.9%    +10.2%    7,994.1     22     ₹175,870      +2.5%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VAKRANGEE   1      IT         01-Feb-17   137.7       320.5       972    ₹177,707      +132.8%     +7.4%     
  IGL         6      OIL&GAS    01-Feb-17   84.2        139.5       1591   ₹88,073       +65.8%      +1.1%     
  DALMIABHA   26     MFG        03-Apr-17   1,969.1     3,120.1     73     ₹84,028       +58.5%      +2.4%     
  DHFL        10     FIN SVC    01-Jun-17   408.8       595.4       369    ₹68,860       +45.7%      -2.9%     
  TVSMOTOR    4      AUTO       01-Jun-17   508.7       692.6       297    ₹54,615       +36.2%      +0.9%     
  HONAUT      5      MFG        01-Aug-17   12,237.3    16,639.1    13     ₹57,223       +36.0%      +1.7%     
  PGHH        9      FMCG       01-Aug-17   7,228.0     8,492.5     22     ₹27,819       +17.5%      +4.5%     
  HINDUNILVR  37     FMCG       01-Jun-17   940.4       1,090.3     160    ₹23,984       +15.9%      -1.0%     
  HDFC        15     PVT BNK    01-Jun-17   371.3       424.2       406    ₹21,465       +14.2%      +0.4%     
  HDFCBANK    16     PVT BNK    01-Jun-17   371.3       424.2       406    ₹21,465       +14.2%      +0.4%     
  RELIANCE    20     OIL&GAS    01-Aug-17   352.5       400.2       460    ₹21,905       +13.5%      -1.3%     
  BRITANNIA   13     FMCG       03-Oct-17   1,920.1     2,126.2     88     ₹18,138       +10.7%      +1.0%     
  RAJESHEXPO  32     CON DUR    01-Aug-17   714.7       751.9       227    ₹8,445        +5.2%       -1.5%     
  GAIL        28     OIL&GAS    03-Oct-17   76.7        80.3        2209   ₹7,940        +4.7%       -0.8%     

  AFTER: Invested ₹3,546,044 | Cash ₹76,994 | Total ₹3,623,038 | Positions 18/20 | Slot ₹181,195

========================================================================
  REBALANCE #19  —  01 Feb 2018
  NAV: ₹3,608,188  |  Slot: ₹180,409  |  Cash: ₹76,994
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VAKRANGEE   41     IT         01-Feb-17   137.7       262.4       972    ₹121,243      +90.6%    365d  
  IGL         56     OIL&GAS    01-Feb-17   84.2        133.1       1591   ₹77,865       +58.1%    365d  
  DALMIABHA   44     MFG        03-Apr-17   1,969.1     2,972.0     73     ₹73,213       +50.9%    304d  
  DHFL        57     FIN SVC    01-Jun-17   408.8       564.2       369    ₹57,349       +38.0%    245d  
  TVSMOTOR    40     AUTO       01-Jun-17   508.7       641.5       297    ₹39,443       +26.1%    245d  
  GAIL        54     OIL&GAS    03-Oct-17   76.7        87.1        2209   ₹22,852       +13.5%    121d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BIOCON      1      HEALTH     2.835    0.41   +82.6%    +62.8%    305.0       591    ₹180,231      +6.0%     
  PCJEWELLER  2      CON DUR    2.775    0.86   +149.8%   +39.3%    48.1        3748   ₹180,398      -8.4%     
  LT          7      INFRA      2.543    0.15   +55.4%    +20.2%    1,278.9     141    ₹180,320      +6.1%     
  MPHASIS     8      IT         2.476    -0.04  +61.5%    +26.3%    722.3       249    ₹179,851      +9.6%     
  TITAN       9      CON DUR    2.415    0.28   +124.0%   +26.8%    807.9       223    ₹180,165      -5.6%     
  IBULHSGFIN  12     FIN SVC    2.130    0.42   +87.9%    +12.5%    1,010.8     178    ₹179,919      +5.3%     
  ABBOTINDIA  14     HEALTH     2.048    0.04   +26.5%    +28.4%    5,020.8     35     ₹175,728      +1.3%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HONAUT      29     MFG        01-Aug-17   12,237.3    16,698.3    13     ₹57,993       +36.5%      -4.5%     
  HINDUNILVR  10     FMCG       01-Jun-17   940.4       1,195.6     160    ₹40,830       +27.1%      +0.3%     
  HDFC        4      PVT BNK    01-Jun-17   371.3       457.0       406    ₹34,786       +23.1%      +2.7%     
  HDFCBANK    5      PVT BNK    01-Jun-17   371.3       457.0       406    ₹34,786       +23.1%      +2.7%     
  RELIANCE    27     OIL&GAS    01-Aug-17   352.5       415.0       460    ₹28,731       +17.7%      -0.3%     
  PGHH        26     FMCG       01-Aug-17   7,228.0     8,267.8     22     ₹22,876       +14.4%      -0.6%     
  RAJESHEXPO  13     CON DUR    01-Aug-17   714.7       814.1       227    ₹22,565       +13.9%      -0.1%     
  MARUTI      6      AUTO       01-Dec-17   7,994.1     8,730.3     22     ₹16,197       +9.2%       -0.1%     
  BRITANNIA   33     FMCG       03-Oct-17   1,920.1     2,097.2     88     ₹15,590       +9.2%       +0.8%     
  BALKRISIND  3      MFG        01-Dec-17   978.3       1,047.3     185    ₹12,752       +7.0%       -1.5%     
  FRETAIL     11     CONSUMP    01-Dec-17   544.8       542.5       332    ₹-730         -0.4%       -1.3%     
  BBTC        37     FMCG       01-Dec-17   1,478.3     1,383.9     122    ₹-11,524      -6.4%       -8.5%     

  AFTER: Invested ₹3,513,005 | Cash ₹93,692 | Total ₹3,606,696 | Positions 19/20 | Slot ₹180,409

========================================================================
  REBALANCE #20  —  02 Apr 2018
  NAV: ₹3,461,733  |  Slot: ₹173,087  |  Cash: ₹93,692
========================================================================
  [SECTOR CAP≤4] dropped: GODREJCP

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HONAUT      46     MFG        01-Aug-17   12,237.3    16,897.3    13     ₹60,580       +38.1%    244d  
  RELIANCE    36     OIL&GAS    01-Aug-17   352.5       392.6       460    ₹18,436       +11.4%    244d  
  RAJESHEXPO  63     CON DUR    01-Aug-17   714.7       733.3       227    ₹4,223        +2.6%     244d  
  BALKRISIND  39     MFG        01-Dec-17   978.3       1,002.1     185    ₹4,396        +2.4%     122d  
  ABBOTINDIA  48     HEALTH     01-Feb-18   5,020.8     4,963.2     35     ₹-2,015       -1.1%     60d   
  BBTC        88     FMCG       01-Dec-17   1,478.3     1,206.8     122    ₹-33,127      -18.4%    122d  
  PCJEWELLER  89     CON DUR    01-Feb-18   48.1        31.1        3748   ₹-63,733      -35.3%    60d   

  ENTRIES (6)
  [52w filter blocked 2: STRTECH(-21.2%), ADANIENSOL(-23.8%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  DBL         1      INFRA      3.529    1.01   +219.8%   +10.8%    1,099.5     157    ₹172,626      +12.7%    
  ASHOKLEY    2      AUTO       3.217    0.44   +72.1%    +24.9%    62.3        2778   ₹173,068      +3.0%     
  CHOLAFIN    7      FIN SVC    2.479    0.43   +49.0%    +14.9%    287.4       602    ₹173,007      +2.6%     
  VBL         10     FMCG       2.353    0.51   +70.4%    +8.5%     37.0        4672   ₹173,068      +2.3%     
  INDUSINDBK  11     PVT BNK    2.348    0.31   +30.4%    +9.3%     1,717.3     100    ₹171,731      +3.7%     
  TCS         16     IT         2.140    0.23   +22.2%    +11.3%    1,178.8     146    ₹172,102      +0.4%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  14     FMCG       01-Jun-17   940.4       1,178.2     160    ₹38,047       +25.3%      +2.2%     
  HDFC        8      PVT BNK    01-Jun-17   371.3       443.3       406    ₹29,200       +19.4%      +2.9%     
  HDFCBANK    9      PVT BNK    01-Jun-17   371.3       443.3       406    ₹29,200       +19.4%      +2.9%     
  BRITANNIA   3      FMCG       03-Oct-17   1,920.1     2,259.4     88     ₹29,861       +17.7%      +4.5%     
  PGHH        18     FMCG       01-Aug-17   7,228.0     8,386.1     22     ₹25,479       +16.0%      +0.5%     
  TITAN       4      CON DUR    01-Feb-18   807.9       917.9       223    ₹24,521       +13.6%      +6.9%     
  MARUTI      30     AUTO       01-Dec-17   7,994.1     8,364.8     22     ₹8,155        +4.6%       +2.1%     
  FRETAIL     13     CONSUMP    01-Dec-17   544.8       549.3       332    ₹1,511        +0.8%       +2.6%     
  BIOCON      23     HEALTH     01-Feb-18   305.0       294.8       591    ₹-5,997       -3.3%       +0.9%     
  MPHASIS     6      IT         01-Feb-18   722.3       695.3       249    ₹-6,716       -3.7%       +0.0%     
  LT          19     INFRA      01-Feb-18   1,278.9     1,173.7     141    ₹-14,822      -8.2%       +2.5%     
  IBULHSGFIN  28     FIN SVC    01-Feb-18   1,010.8     913.5       178    ₹-17,307      -9.6%       +1.1%     

  AFTER: Invested ₹3,213,921 | Cash ₹246,583 | Total ₹3,460,503 | Positions 18/20 | Slot ₹173,087

========================================================================
  REBALANCE #21  —  01 Jun 2018
  NAV: ₹3,674,777  |  Slot: ₹183,739  |  Cash: ₹246,583
========================================================================
  [SECTOR CAP≤4] dropped: DABUR, COLPAL

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PGHH        44     FMCG       01-Aug-17   7,228.0     8,342.6     22     ₹24,523       +15.4%    304d  
  CHOLAFIN    28     FIN SVC    02-Apr-18   287.4       308.5       602    ₹12,692       +7.3%     60d   
  BIOCON      —      HEALTH     01-Feb-18   305.0       319.7       591    ₹8,704        +4.8%     120d  
  MARUTI      49     AUTO       01-Dec-17   7,994.1     8,179.7     22     ₹4,083        +2.3%     182d  
  ASHOKLEY    27     AUTO       02-Apr-18   62.3        63.5        2778   ₹3,214        +1.9%     60d   
  LT          48     INFRA      01-Feb-18   1,278.9     1,205.8     141    ₹-10,307      -5.7%     120d  
  IBULHSGFIN  62     FIN SVC    01-Feb-18   1,010.8     918.6       178    ₹-16,413      -9.1%     120d  
  DBL         42     INFRA      02-Apr-18   1,099.5     847.7       157    ₹-39,529      -22.9%    60d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    1      CONSUMP    3.392    0.45   +166.5%   +23.0%    245.3       749    ₹183,726      -1.3%     
  COFORGE     4      IT         2.740    0.72   +120.6%   +32.7%    201.6       911    ₹183,614      +3.3%     
  PIDILITIND  5      MFG        2.662    0.47   +49.2%    +24.2%    538.3       341    ₹183,556      +0.5%     
  DMART       6      FMCG       2.562    0.47   +114.2%   +13.6%    1,531.1     120    ₹183,732      +3.3%     
  TECHM       7      IT         2.562    0.03   +89.2%    +14.4%    516.6       355    ₹183,397      +2.4%     
  KOTAKBANK   10     PVT BNK    2.516    0.66   +36.2%    +20.9%    262.2       700    ₹183,560      +3.5%     
  M&M         11     AUTO       2.434    0.70   +33.8%    +23.8%    829.0       221    ₹183,216      +4.7%     
  BAJFINANCE  15     FIN SVC    2.377    0.65   +59.3%    +26.7%    201.1       913    ₹183,594      +2.1%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  3      FMCG       01-Jun-17   940.4       1,385.4     160    ₹71,187       +47.3%      +2.3%     
  BRITANNIA   2      FMCG       03-Oct-17   1,920.1     2,571.5     88     ₹57,323       +33.9%      +2.9%     
  HDFC        13     PVT BNK    01-Jun-17   371.3       487.5       406    ₹47,174       +31.3%      +4.9%     
  HDFCBANK    14     PVT BNK    01-Jun-17   371.3       487.5       406    ₹47,174       +31.3%      +4.9%     
  VBL         17     FMCG       02-Apr-18   37.0        44.6        4672   ₹35,140       +20.3%      +7.1%     
  MPHASIS     8      IT         01-Feb-18   722.3       868.5       249    ₹36,393       +20.2%      +0.6%     
  TCS         19     IT         02-Apr-18   1,178.8     1,415.4     146    ₹34,548       +20.1%      +0.3%     
  TITAN       18     CON DUR    01-Feb-18   807.9       875.0       223    ₹14,960       +8.3%       -2.9%     
  FRETAIL     25     CONSUMP    01-Dec-17   544.8       586.9       332    ₹13,994       +7.7%       +1.4%     
  INDUSINDBK  22     PVT BNK    02-Apr-18   1,717.3     1,822.8     100    ₹10,544       +6.1%       +0.9%     

  AFTER: Invested ₹3,515,566 | Cash ₹157,467 | Total ₹3,673,033 | Positions 18/20 | Slot ₹183,739

========================================================================
  REBALANCE #22  —  01 Aug 2018
  NAV: ₹3,913,433  |  Slot: ₹195,672  |  Cash: ₹157,467
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VBL         38     FMCG       02-Apr-18   37.0        43.0        4672   ₹27,958       +16.2%    121d  
  INDUSINDBK  34     PVT BNK    02-Apr-18   1,717.3     1,919.4     100    ₹20,213       +11.8%    121d  
  TITAN       36     CON DUR    01-Feb-18   807.9       895.8       223    ₹19,592       +10.9%    181d  
  KOTAKBANK   28     PVT BNK    01-Jun-18   262.2       261.3       700    ₹-645         -0.4%     61d   
  FRETAIL     61     CONSUMP    01-Dec-17   544.8       540.8       332    ₹-1,311       -0.7%     243d  

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ABBOTINDIA  4      HEALTH     2.803    0.29   +80.6%    +23.2%    6,983.6     28     ₹195,542      +4.1%     
  RELIANCE    6      OIL&GAS    2.762    0.79   +50.6%    +25.8%    527.8       370    ₹195,278      +8.4%     
  PAGEIND     9      MFG        2.515    0.68   +82.6%    +26.9%    27,338.4    7      ₹191,369      +4.9%     
  BAJAJFINSV  10     FIN SVC    2.484    0.82   +41.9%    +30.7%    696.2       281    ₹195,641      +5.8%     
  BATAINDIA   11     CON DUR    2.434    0.68   +64.4%    +21.6%    878.9       222    ₹195,125      +7.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  3      FMCG       01-Jun-17   940.4       1,523.6     160    ₹93,304       +62.0%      +3.2%     
  BRITANNIA   1      FMCG       03-Oct-17   1,920.1     2,882.3     88     ₹84,679       +50.1%      +2.0%     
  MPHASIS     5      IT         01-Feb-18   722.3       1,006.5     249    ₹70,765       +39.3%      +6.6%     
  TCS         8      IT         02-Apr-18   1,178.8     1,618.1     146    ₹64,134       +37.3%      +1.6%     
  HDFC        20     PVT BNK    01-Jun-17   371.3       498.6       406    ₹51,685       +34.3%      +0.0%     
  HDFCBANK    21     PVT BNK    01-Jun-17   371.3       498.6       406    ₹51,685       +34.3%      +0.0%     
  BAJFINANCE  2      FIN SVC    01-Jun-18   201.1       264.0       913    ₹57,452       +31.3%      +5.8%     
  COFORGE     7      IT         01-Jun-18   201.6       227.3       911    ₹23,499       +12.8%      +6.2%     
  JUBLFOOD    11     CONSUMP    01-Jun-18   245.3       273.8       749    ₹21,325       +11.6%      -1.1%     
  DMART       15     FMCG       01-Jun-18   1,531.1     1,669.7     120    ₹16,632       +9.1%       +6.2%     
  M&M         22     AUTO       01-Jun-18   829.0       861.9       221    ₹7,255        +4.0%       +1.6%     
  PIDILITIND  23     MFG        01-Jun-18   538.3       543.6       341    ₹1,820        +1.0%       +3.8%     
  TECHM       17     IT         01-Jun-18   516.6       513.2       355    ₹-1,204       -0.7%       +5.1%     

  AFTER: Invested ₹3,773,733 | Cash ₹138,545 | Total ₹3,912,277 | Positions 18/20 | Slot ₹195,672

========================================================================
  REBALANCE #23  —  01 Oct 2018
  NAV: ₹3,749,758  |  Slot: ₹187,488  |  Cash: ₹138,545
========================================================================
  [SECTOR CAP≤4] dropped: INFY, WIPRO, HCLTECH

  [REGIME OFF] Nifty 200 5,772.3 < EMA200 5,787.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TCS         1      IT         02-Apr-18   1,178.8     1,846.5     146    ₹97,486       +56.6%      +6.3%     
  HINDUNILVR  23     FMCG       01-Jun-17   940.4       1,442.8     160    ₹80,372       +53.4%      -0.1%     
  BRITANNIA   29     FMCG       03-Oct-17   1,920.1     2,595.3     88     ₹59,419       +35.2%      -2.7%     
  MPHASIS     8      IT         01-Feb-18   722.3       968.9       249    ₹61,406       +34.1%      -3.5%     
  HDFC        63     PVT BNK    01-Jun-17   371.3       470.2       406    ₹40,126       +26.6% ⚠    +1.2%     
  HDFCBANK    64     PVT BNK    01-Jun-17   371.3       470.2       406    ₹40,126       +26.6% ⚠    +1.2%     
  PAGEIND     5      MFG        01-Aug-18   27,338.4    30,462.0    7      ₹21,865       +11.4%      +0.7%     
  TECHM       6      IT         01-Jun-18   516.6       572.3       355    ₹19,781       +10.8%      +1.7%     
  COFORGE     7      IT         01-Jun-18   201.6       214.9       911    ₹12,167       +6.6%       -6.0%     
  BAJFINANCE  50     FIN SVC    01-Jun-18   201.1       214.1       913    ₹11,922       +6.5%       -10.5%    
  RELIANCE    3      OIL&GAS    01-Aug-18   527.8       545.2       370    ₹6,445        +3.3%       -0.5%     
  BATAINDIA   14     CON DUR    01-Aug-18   878.9       903.9       222    ₹5,546        +2.8%       -2.9%     
  ABBOTINDIA  10     HEALTH     01-Aug-18   6,983.6     6,960.2     28     ₹-655         -0.3%       -5.4%     
  JUBLFOOD    22     CONSUMP    01-Jun-18   245.3       243.8       749    ₹-1,104       -0.6%       -7.2%     
  M&M         36     AUTO       01-Jun-18   829.0       785.7       221    ₹-9,575       -5.2%       -7.5%     
  PIDILITIND  27     MFG        01-Jun-18   538.3       503.3       341    ₹-11,916      -6.5%       -5.4%     
  DMART       51     FMCG       01-Jun-18   1,531.1     1,348.8     120    ₹-21,882      -11.9%      -9.0%     
  BAJAJFINSV  45     FIN SVC    01-Aug-18   696.2       585.8       281    ₹-31,019      -15.9%      -7.3%     
  ⚠  WAZ < 0 (momentum below universe mean): HDFC, HDFCBANK

  AFTER: Invested ₹3,611,214 | Cash ₹138,545 | Total ₹3,749,758 | Positions 18/20 | Slot ₹187,488

========================================================================
  REBALANCE #24  —  03 Dec 2018
  NAV: ₹3,743,306  |  Slot: ₹187,165  |  Cash: ₹138,545
========================================================================
  [SECTOR CAP≤4] dropped: INFY

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BRITANNIA   33     FMCG       03-Oct-17   1,920.1     2,748.7     88     ₹72,920       +43.2%    426d  
  BAJFINANCE  37     FIN SVC    01-Jun-18   201.1       243.3       913    ₹38,494       +21.0%    185d  
  MPHASIS     42     IT         01-Feb-18   722.3       834.0       249    ₹27,824       +15.5%    305d  
  RELIANCE    45     OIL&GAS    01-Aug-18   527.8       511.9       370    ₹-5,880       -3.0%     124d  
  PAGEIND     73     MFG        01-Aug-18   27,338.4    24,355.1    7      ₹-20,883      -10.9%    124d  
  BAJAJFINSV  56     FIN SVC    01-Aug-18   696.2       597.5       281    ₹-27,732      -14.2%    124d  
  M&M         96     AUTO       01-Jun-18   829.0       705.5       221    ₹-27,305      -14.9%    185d  

  ENTRIES (8)
  [52w filter blocked 1: JUBILANT(-22.6%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COLPAL      1      FMCG       2.894    0.33   +24.6%    +11.1%    1,072.8     174    ₹186,675      +8.5%     
  AUROPHARMA  4      HEALTH     2.435    1.00   +16.3%    +17.5%    771.7       242    ₹186,744      +2.1%     
  WIPRO       5      IT         2.378    0.33   +12.2%    +11.0%    112.7       1660   ₹187,076      +2.4%     
  ICICIGI     6      FIN SVC    2.362    0.51   +23.3%    +13.0%    818.5       228    ₹186,625      +4.7%     
  HONAUT      7      MFG        2.234    0.79   +33.7%    +4.4%     22,459.3    8      ₹179,675      +7.7%     
  LT          9      INFRA      2.187    1.03   +18.9%    +5.3%     1,270.2     147    ₹186,718      +2.9%     
  VBL         10     FMCG       2.178    0.29   +57.6%    -3.8%     45.3        4128   ₹187,155      +0.7%     
  TITAN       18     CON DUR    1.956    0.85   +16.7%    +5.4%     914.2       204    ₹186,502      +4.0%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  2      FMCG       01-Jun-17   940.4       1,613.0     160    ₹107,602      +71.5%      +7.1%     
  TCS         8      IT         02-Apr-18   1,178.8     1,626.3     146    ₹65,338       +38.0%      +3.4%     
  HDFC        16     PVT BNK    01-Jun-17   371.3       488.1       406    ₹47,427       +31.5%      +4.0%     
  HDFCBANK    17     PVT BNK    01-Jun-17   371.3       488.1       406    ₹47,427       +31.5%      +4.0%     
  BATAINDIA   14     CON DUR    01-Aug-18   878.9       993.5       222    ₹25,429       +13.0%      +6.7%     
  JUBLFOOD    28     CONSUMP    01-Jun-18   245.3       261.4       749    ₹12,025       +6.5%       +10.2%    
  TECHM       12     IT         01-Jun-18   516.6       537.0       355    ₹7,249        +4.0%       +1.4%     
  PIDILITIND  11     MFG        01-Jun-18   538.3       556.8       341    ₹6,316        +3.4%       +3.7%     
  ABBOTINDIA  3      HEALTH     01-Aug-18   6,983.6     7,082.9     28     ₹2,780        +1.4%       +4.3%     
  COFORGE     24     IT         01-Jun-18   201.6       203.2       911    ₹1,465        +0.8%       -1.8%     
  DMART       32     FMCG       01-Jun-18   1,531.1     1,477.4     120    ₹-6,438       -3.5%       +4.7%     

  AFTER: Invested ₹3,736,581 | Cash ₹4,960 | Total ₹3,741,540 | Positions 19/20 | Slot ₹187,165

========================================================================
  REBALANCE #25  —  01 Feb 2019
  NAV: ₹3,837,565  |  Slot: ₹191,878  |  Cash: ₹4,960
========================================================================
  [SECTOR CAP≤4] dropped: MARICO

  [REGIME OFF] Nifty 200 5,701.6 < EMA200 5,703.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  5      FMCG       01-Jun-17   940.4       1,589.7     160    ₹103,884      +69.0%      +1.8%     
  TCS         20     IT         02-Apr-18   1,178.8     1,668.9     146    ₹71,554       +41.6%      +4.9%     
  HDFC        22     PVT BNK    01-Jun-17   371.3       482.9       406    ₹45,284       +30.0%      -0.3%     
  HDFCBANK    23     PVT BNK    01-Jun-17   371.3       482.9       406    ₹45,284       +30.0%      -0.3%     
  BATAINDIA   3      CON DUR    01-Aug-18   878.9       1,068.3     222    ₹42,046       +21.5%      +1.3%     
  COFORGE     17     IT         01-Jun-18   201.6       243.0       911    ₹37,715       +20.5%      +5.6%     
  WIPRO       12     IT         03-Dec-18   112.7       127.9       1660   ₹25,275       +13.5%      +6.9%     
  TECHM       30     IT         01-Jun-18   516.6       562.6       355    ₹16,317       +8.9%       +3.9%     
  JUBLFOOD    14     CONSUMP    01-Jun-18   245.3       266.9       749    ₹16,146       +8.8%       +10.6%    
  VBL         32     FMCG       03-Dec-18   45.3        48.3        4128   ₹12,302       +6.6%       +4.4%     
  TITAN       21     CON DUR    03-Dec-18   914.2       968.2       204    ₹11,016       +5.9%       +2.8%     
  ABBOTINDIA  8      HEALTH     01-Aug-18   6,983.6     7,377.5     28     ₹11,027       +5.6%       +1.0%     
  ICICIGI     41     FIN SVC    03-Dec-18   818.5       839.3       228    ₹4,740        +2.5%       +2.5%     
  COLPAL      9      FMCG       03-Dec-18   1,072.8     1,090.0     174    ₹2,977        +1.6%       -0.1%     
  PIDILITIND  11     MFG        01-Jun-18   538.3       542.8       341    ₹1,536        +0.8%       -0.2%     
  AUROPHARMA  38     HEALTH     03-Dec-18   771.7       764.9       242    ₹-1,636       -0.9%       +3.3%     
  HONAUT      26     MFG        03-Dec-18   22,459.3    21,361.3    8      ₹-8,784       -4.9%       -0.3%     
  DMART       31     FMCG       01-Jun-18   1,531.1     1,442.9     120    ₹-10,578      -5.8%       +1.1%     
  LT          63     INFRA      03-Dec-18   1,270.2     1,178.6     147    ₹-13,459      -7.2% ⚠     -0.3%     
  ⚠  WAZ < 0 (momentum below universe mean): LT

  AFTER: Invested ₹3,832,606 | Cash ₹4,960 | Total ₹3,837,565 | Positions 19/20 | Slot ₹191,878

========================================================================
  REBALANCE #26  —  01 Apr 2019
  NAV: ₹4,015,501  |  Slot: ₹200,775  |  Cash: ₹4,960
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HINDUNILVR  57     FMCG       01-Jun-17   940.4       1,493.2     160    ₹88,433       +58.8%    669d  
  TECHM       27     IT         01-Jun-18   516.6       591.9       355    ₹26,723       +14.6%    304d  
  WIPRO       35     IT         03-Dec-18   112.7       120.1       1660   ₹12,311       +6.6%     119d  
  LT          73     INFRA      03-Dec-18   1,270.2     1,256.6     147    ₹-2,001       -1.1%     119d  
  COLPAL      55     FMCG       03-Dec-18   1,072.8     1,060.2     174    ₹-2,192       -1.2%     119d  
  DMART       83     FMCG       01-Jun-18   1,531.1     1,492.9     120    ₹-4,578       -2.5%     304d  
  ABBOTINDIA  42     HEALTH     01-Aug-18   6,983.6     6,646.9     28     ₹-9,429       -4.8%     243d  

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RELIANCE    2      OIL&GAS    3.044    0.35   +56.2%    +24.3%    616.1       325    ₹200,228      +4.9%     
  BAJFINANCE  4      FIN SVC    2.548    0.10   +76.4%    +16.0%    291.0       689    ₹200,523      +5.5%     
  RBLBANK     6      PVT BNK    2.475    0.45   +47.7%    +17.8%    659.7       304    ₹200,543      +6.6%     
  MUTHOOTFIN  7      FIN SVC    2.471    0.08   +55.7%    +24.5%    537.4       373    ₹200,444      +5.2%     
  INFY        8      IT         2.362    0.08   +34.7%    +15.6%    618.5       324    ₹200,381      +3.3%     
  HAVELLS     11     CON DUR    2.300    0.36   +56.7%    +13.3%    736.8       272    ₹200,414      +4.1%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   1      CON DUR    01-Aug-18   878.9       1,295.6     222    ₹92,509       +47.4%      +3.3%     
  HDFC        9      PVT BNK    01-Jun-17   371.3       534.0       406    ₹66,042       +43.8%      +3.5%     
  HDFCBANK    10     PVT BNK    01-Jun-17   371.3       534.0       406    ₹66,042       +43.8%      +3.5%     
  TCS         15     IT         02-Apr-18   1,178.8     1,670.3     146    ₹71,758       +41.7%      +1.5%     
  COFORGE     23     IT         01-Jun-18   201.6       245.6       911    ₹40,147       +21.9%      +1.1%     
  TITAN       3      CON DUR    03-Dec-18   914.2       1,094.0     204    ₹36,673       +19.7%      +2.9%     
  ICICIGI     16     FIN SVC    03-Dec-18   818.5       968.3       228    ₹34,137       +18.3%      +4.1%     
  JUBLFOOD    21     CONSUMP    01-Jun-18   245.3       287.0       749    ₹31,211       +17.0%      +5.1%     
  VBL         5      FMCG       03-Dec-18   45.3        52.0        4128   ₹27,427       +14.7%      +9.5%     
  PIDILITIND  13     MFG        01-Jun-18   538.3       605.6       341    ₹22,941       +12.5%      +6.2%     
  AUROPHARMA  24     HEALTH     03-Dec-18   771.7       761.0       242    ₹-2,593       -1.4%       +3.5%     
  HONAUT      19     MFG        03-Dec-18   22,459.3    21,836.7    8      ₹-4,981       -2.8%       +0.4%     

  AFTER: Invested ₹3,830,195 | Cash ₹183,878 | Total ₹4,014,073 | Positions 18/20 | Slot ₹200,775

========================================================================
  REBALANCE #27  —  03 Jun 2019
  NAV: ₹4,180,262  |  Slot: ₹209,013  |  Cash: ₹183,878
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COFORGE     68     IT         01-Jun-18   201.6       241.5       911    ₹36,389       +19.8%    367d  
  PIDILITIND  —      MFG        01-Jun-18   538.3       626.4       341    ₹30,038       +16.4%    367d  
  JUBLFOOD    74     CONSUMP    01-Jun-18   245.3       263.8       749    ₹13,872       +7.6%     367d  
  INFY        —      IT         01-Apr-19   618.5       609.9       324    ₹-2,773       -1.4%     63d   
  AUROPHARMA  —      HEALTH     03-Dec-18   771.7       630.8       242    ₹-34,085      -18.3%    182d  

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  NAUKRI      2      IT         2.909    0.05   +97.5%    +39.0%    451.4       463    ₹208,992      +15.9%    
  GUJGASLTD   4      OIL&GAS    2.694    0.57   +10.2%    +55.1%    177.2       1179   ₹208,894      +9.8%     
  BAJAJFINSV  6      FIN SVC    2.509    0.25   +39.8%    +31.8%    831.5       251    ₹208,703      +4.8%     
  AXISBANK    9      PVT BNK    2.362    0.37   +55.3%    +15.5%    808.3       258    ₹208,533      +4.0%     
  PIIND       10     MFG        2.361    0.29   +35.4%    +23.7%    1,116.0     187    ₹208,691      +4.8%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TCS         25     IT         02-Apr-18   1,178.8     1,843.5     146    ₹97,043       +56.4%      +5.4%     
  HDFC        7      PVT BNK    01-Jun-17   371.3       567.6       406    ₹79,669       +52.8%      +3.3%     
  HDFCBANK    8      PVT BNK    01-Jun-17   371.3       567.6       406    ₹79,669       +52.8%      +3.3%     
  BATAINDIA   12     CON DUR    01-Aug-18   878.9       1,272.7     222    ₹87,419       +44.8%      +0.7%     
  ICICIGI     1      FIN SVC    03-Dec-18   818.5       1,164.0     228    ₹78,755       +42.2%      +8.1%     
  TITAN       13     CON DUR    03-Dec-18   914.2       1,236.5     204    ₹65,736       +35.2%      +5.1%     
  VBL         22     FMCG       03-Dec-18   45.3        54.8        4128   ₹39,059       +20.9%      +3.0%     
  HONAUT      3      MFG        03-Dec-18   22,459.3    26,282.6    8      ₹30,586       +17.0%      +7.2%     
  BAJFINANCE  5      FIN SVC    01-Apr-19   291.0       340.4       689    ₹34,046       +17.0%      +6.1%     
  MUTHOOTFIN  14     FIN SVC    01-Apr-19   537.4       573.4       373    ₹13,444       +6.7%       +4.1%     
  RBLBANK     19     PVT BNK    01-Apr-19   659.7       675.0       304    ₹4,661        +2.3%       +2.7%     
  HAVELLS     28     CON DUR    01-Apr-19   736.8       731.4       272    ₹-1,473       -0.7%       +3.5%     
  RELIANCE    23     OIL&GAS    01-Apr-19   616.1       602.1       325    ₹-4,553       -2.3%       +2.7%     

  AFTER: Invested ₹4,058,734 | Cash ₹120,289 | Total ₹4,179,022 | Positions 18/20 | Slot ₹209,013

========================================================================
  REBALANCE #28  —  01 Aug 2019
  NAV: ₹3,762,951  |  Slot: ₹188,148  |  Cash: ₹120,289
========================================================================
  [SECTOR CAP≤4] dropped: HDFCLIFE

  [REGIME OFF] Nifty 200 5,661.5 < EMA200 5,891.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TCS         30     IT         02-Apr-18   1,178.8     1,811.0     146    ₹92,305       +53.6%      +1.5%     
  HDFC        67     PVT BNK    01-Jun-17   371.3       517.5       406    ₹59,364       +39.4% ⚠    -4.1%     
  HDFCBANK    68     PVT BNK    01-Jun-17   371.3       517.5       406    ₹59,364       +39.4% ⚠    -4.1%     
  BATAINDIA   13     CON DUR    01-Aug-18   878.9       1,221.7     222    ₹76,081       +39.0%      -2.1%     
  ICICIGI     8      FIN SVC    03-Dec-18   818.5       1,101.5     228    ₹64,517       +34.6%      +2.9%     
  VBL         16     FMCG       03-Dec-18   45.3        54.2        4128   ₹36,641       +19.6%      -0.8%     
  TITAN       47     CON DUR    03-Dec-18   914.2       1,037.1     204    ₹25,069       +13.4%      -5.1%     
  BAJFINANCE  23     FIN SVC    01-Apr-19   291.0       313.0       689    ₹15,119       +7.5%       -3.3%     
  MUTHOOTFIN  9      FIN SVC    01-Apr-19   537.4       544.8       373    ₹2,756        +1.4%       +0.0%     
  HONAUT      34     MFG        03-Dec-18   22,459.3    22,664.6    8      ₹1,642        +0.9%       -0.7%     
  PIIND       4      MFG        03-Jun-19   1,116.0     1,093.1     187    ₹-4,288       -2.1%       -0.1%     
  NAUKRI      1      IT         03-Jun-19   451.4       433.1       463    ₹-8,466       -4.1%       +0.1%     
  GUJGASLTD   22     OIL&GAS    03-Jun-19   177.2       161.8       1179   ₹-18,111      -8.7%       +2.8%     
  BAJAJFINSV  55     FIN SVC    03-Jun-19   831.5       705.6       251    ₹-31,595      -15.1%      -4.9%     
  RELIANCE    80     OIL&GAS    01-Apr-19   616.1       522.4       325    ₹-30,440      -15.2% ⚠    -4.7%     
  HAVELLS     70     CON DUR    01-Apr-19   736.8       619.1       272    ₹-32,009      -16.0% ⚠    -7.5%     
  AXISBANK    50     PVT BNK    03-Jun-19   808.3       666.5       258    ₹-36,572      -17.5%      -8.5%     
  RBLBANK     130    PVT BNK    01-Apr-19   659.7       385.4       304    ₹-83,389      -41.6% ⚠    -21.7%    
  ⚠  WAZ < 0 (momentum below universe mean): HDFC, HDFCBANK, HAVELLS, RELIANCE, RBLBANK

  AFTER: Invested ₹3,642,662 | Cash ₹120,289 | Total ₹3,762,951 | Positions 18/20 | Slot ₹188,148

========================================================================
  REBALANCE #29  —  01 Oct 2019
  NAV: ₹4,112,517  |  Slot: ₹205,626  |  Cash: ₹120,289
========================================================================
  [SECTOR CAP≤4] dropped: PGHH

  EXITS (11)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HDFCBANK    25     PVT BNK    01-Jun-17   371.3       581.8       406    ₹85,447       +56.7%    852d  
  TCS         76     IT         02-Apr-18   1,178.8     1,711.2     146    ₹77,738       +45.2%    547d  
  ICICIGI     26     FIN SVC    03-Dec-18   818.5       1,151.0     228    ₹75,804       +40.6%    302d  
  VBL         57     FMCG       03-Dec-18   45.3        52.7        4128   ₹30,220       +16.1%    302d  
  BAJAJFINSV  36     FIN SVC    03-Jun-19   831.5       842.7       251    ₹2,825        +1.4%     120d  
  RELIANCE    46     OIL&GAS    01-Apr-19   616.1       581.1       325    ₹-11,382      -5.7%     183d  
  GUJGASLTD   42     OIL&GAS    03-Jun-19   177.2       163.7       1179   ₹-15,903      -7.6%     120d  
  HAVELLS     71     CON DUR    01-Apr-19   736.8       678.2       272    ₹-15,942      -8.0%     183d  
  NAUKRI      52     IT         03-Jun-19   451.4       403.6       463    ₹-22,146      -10.6%    120d  
  AXISBANK    83     PVT BNK    03-Jun-19   808.3       676.3       258    ₹-34,041      -16.3%    120d  
  RBLBANK     132    PVT BNK    01-Apr-19   659.7       291.7       304    ₹-111,859     -55.8%    183d  

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COLPAL      1      FMCG       3.341    0.51   +36.9%    +32.6%    1,276.3     161    ₹205,491      +7.4%     
  HDFCAMC     2      FIN SVC    3.315    0.63   +74.0%    +33.9%    1,208.6     170    ₹205,461      +1.9%     
  BERGEPAINT  4      CON DUR    3.162    0.69   +43.0%    +37.3%    350.7       586    ₹205,539      +8.3%     
  ASIANPAINT  5      CONSUMP    2.991    0.56   +36.0%    +30.0%    1,662.1     123    ₹204,441      +6.0%     
  HDFCLIFE    6      FIN SVC    2.903    0.44   +45.5%    +28.9%    587.1       350    ₹205,481      +7.6%     
  ABBOTINDIA  7      HEALTH     2.643    0.47   +29.6%    +23.1%    9,965.1     20     ₹199,302      +6.8%     
  NESTLEIND   8      FMCG       2.616    0.37   +38.1%    +16.9%    636.8       322    ₹205,058      +3.8%     
  LALPATHLAB  10     HEALTH     2.413    0.56   +38.6%    +25.3%    634.6       324    ₹205,613      +1.8%     
  DMART       11     FMCG       2.354    0.70   +24.3%    +35.6%    1,896.0     108    ₹204,768      +9.4%     
  HINDUNILVR  12     FMCG       2.224    0.42   +25.6%    +11.3%    1,771.2     116    ₹205,458      +3.2%     
  WHIRLPOOL   13     CON DUR    2.186    0.54   +22.7%    +23.4%    1,921.6     107    ₹205,616      +10.5%    

  HOLDS (7)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   3      CON DUR    01-Aug-18   878.9       1,630.0     222    ₹166,737      +85.5%      +6.5%     
  HDFC        24     PVT BNK    01-Jun-17   371.3       581.8       406    ₹85,447       +56.7%      +5.4%     
  TITAN       21     CON DUR    03-Dec-18   914.2       1,256.2     204    ₹69,753       +37.4%      +6.3%     
  BAJFINANCE  19     FIN SVC    01-Apr-19   291.0       387.8       689    ₹66,672       +33.2%      +8.0%     
  HONAUT      16     MFG        03-Dec-18   22,459.3    27,751.2    8      ₹42,335       +23.6%      +4.4%     
  MUTHOOTFIN  22     FIN SVC    01-Apr-19   537.4       600.6       373    ₹23,566       +11.8%      +5.1%     
  PIIND       9      MFG        03-Jun-19   1,116.0     1,235.2     187    ₹22,282       +10.7%      +0.5%     

  AFTER: Invested ₹4,050,743 | Cash ₹59,100 | Total ₹4,109,843 | Positions 18/20 | Slot ₹205,626

========================================================================
  REBALANCE #30  —  02 Dec 2019
  NAV: ₹4,271,975  |  Slot: ₹213,599  |  Cash: ₹59,100
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJFINSV

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TITAN       68     CON DUR    03-Dec-18   914.2       1,131.8     204    ₹44,380       +23.8%    364d  
  HONAUT      37     MFG        03-Dec-18   22,459.3    26,924.4    8      ₹35,721       +19.9%    364d  
  MUTHOOTFIN  48     FIN SVC    01-Apr-19   537.4       604.4       373    ₹24,998       +12.5%    245d  
  HINDUNILVR  34     FMCG       01-Oct-19   1,771.2     1,827.7     116    ₹6,557        +3.2%     62d   
  ASIANPAINT  36     CONSUMP    01-Oct-19   1,662.1     1,638.9     123    ₹-2,852       -1.4%     62d   
  HDFCLIFE    47     FIN SVC    01-Oct-19   587.1       558.8       350    ₹-9,893       -4.8%     62d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IGL         7      OIL&GAS    2.491    0.12   +56.4%    +26.9%    185.5       1151   ₹213,486      +2.4%     
  MANAPPURAM  8      FIN SVC    2.365    0.17   +87.0%    +28.7%    139.6       1529   ₹213,471      -1.9%     
  NAM-INDIA   9      FIN SVC    2.341    0.28   +116.1%   +37.1%    290.5       735    ₹213,497      -0.4%     
  GUJGASLTD   11     OIL&GAS    2.246    0.08   +74.3%    +21.1%    206.7       1033   ₹213,471      +8.8%     
  RELIANCE    12     OIL&GAS    2.215    0.16   +41.6%    +24.4%    706.5       302    ₹213,351      +4.8%     
  SIEMENS     13     ENERGY     2.192    0.19   +52.8%    +23.9%    841.1       253    ₹212,789      -3.9%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   21     CON DUR    01-Aug-18   878.9       1,526.7     222    ₹143,800      +73.7%      -2.5%     
  HDFC        26     PVT BNK    01-Jun-17   371.3       589.7       406    ₹88,653       +58.8%      +0.2%     
  PIIND       3      MFG        03-Jun-19   1,116.0     1,478.3     187    ₹67,752       +32.5%      +5.3%     
  BAJFINANCE  20     FIN SVC    01-Apr-19   291.0       383.7       689    ₹63,864       +31.8%      -3.4%     
  HDFCAMC     1      FIN SVC    01-Oct-19   1,208.6     1,498.3     170    ₹49,250       +24.0%      -0.9%     
  LALPATHLAB  4      HEALTH     01-Oct-19   634.6       760.0       324    ₹40,615       +19.8%      +2.5%     
  ABBOTINDIA  2      HEALTH     01-Oct-19   9,965.1     11,580.2    20     ₹32,301       +16.2%      +2.8%     
  BERGEPAINT  6      CON DUR    01-Oct-19   350.7       400.4       586    ₹29,092       +14.2%      +1.3%     
  WHIRLPOOL   5      CON DUR    01-Oct-19   1,921.6     2,121.8     107    ₹21,420       +10.4%      -0.8%     
  NESTLEIND   14     FMCG       01-Oct-19   636.8       677.6       322    ₹13,131       +6.4%       +0.9%     
  COLPAL      16     FMCG       01-Oct-19   1,276.3     1,248.1     161    ₹-4,539       -2.2%       -3.8%     
  DMART       30     FMCG       01-Oct-19   1,896.0     1,846.7     108    ₹-5,324       -2.6%       -0.9%     

  AFTER: Invested ₹4,212,029 | Cash ₹58,426 | Total ₹4,270,455 | Positions 18/20 | Slot ₹213,599

========================================================================
  REBALANCE #31  —  03 Feb 2020
  NAV: ₹4,559,177  |  Slot: ₹227,959  |  Cash: ₹58,426
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HDFC        69     PVT BNK    01-Jun-17   371.3       555.7       406    ₹74,855       +49.7%    977d  
  BAJFINANCE  25     FIN SVC    01-Apr-19   291.0       423.4       689    ₹91,230       +45.5%    308d  
  SIEMENS     68     ENERGY     02-Dec-19   841.1       821.8       253    ₹-4,870       -2.3%     63d   
  COLPAL      112    FMCG       01-Oct-19   1,276.3     1,161.5     161    ₹-18,494      -9.0%     125d  
  RELIANCE    75     OIL&GAS    02-Dec-19   706.5       617.0       302    ₹-27,030      -12.7%    63d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COROMANDEL  2      MFG        3.274    0.04   +45.3%    +35.1%    585.9       389    ₹227,930      +6.6%     
  SRF         4      MFG        3.093    -0.04  +87.1%    +29.3%    741.4       307    ₹227,612      +4.2%     
  JUBLFOOD    5      CONSUMP    2.586    0.34   +66.3%    +25.1%    385.8       590    ₹227,622      +9.9%     
  COFORGE     8      IT         2.407    0.21   +49.5%    +18.8%    349.4       652    ₹227,810      +2.6%     
  IPCALAB     9      HEALTH     2.406    0.07   +53.4%    +18.6%    583.3       390    ₹227,479      -1.9%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   13     CON DUR    01-Aug-18   878.9       1,736.3     222    ₹190,331      +97.5%      +2.9%     
  PIIND       7      MFG        03-Jun-19   1,116.0     1,520.1     187    ₹75,560       +36.2%      +3.5%     
  BERGEPAINT  6      CON DUR    01-Oct-19   350.7       461.5       586    ₹64,893       +31.6%      +4.1%     
  LALPATHLAB  10     HEALTH     01-Oct-19   634.6       830.1       324    ₹63,343       +30.8%      +4.0%     
  GUJGASLTD   1      OIL&GAS    02-Dec-19   206.7       269.8       1033   ₹65,214       +30.5%      +2.3%     
  WHIRLPOOL   20     CON DUR    01-Oct-19   1,921.6     2,361.3     107    ₹47,045       +22.9%      -0.1%     
  IGL         3      OIL&GAS    02-Dec-19   185.5       227.3       1151   ₹48,091       +22.5%      +6.7%     
  NESTLEIND   14     FMCG       01-Oct-19   636.8       761.7       322    ₹40,208       +19.6%      +6.2%     
  ABBOTINDIA  17     HEALTH     01-Oct-19   9,965.1     11,694.8    20     ₹34,595       +17.4%      +0.5%     
  MANAPPURAM  19     FIN SVC    02-Dec-19   139.6       163.8       1529   ₹36,927       +17.3%      +1.0%     
  DMART       24     FMCG       01-Oct-19   1,896.0     2,131.6     108    ₹25,450       +12.4%      +8.4%     
  HDFCAMC     11     FIN SVC    01-Oct-19   1,208.6     1,354.2     170    ₹24,745       +12.0%      -3.1%     
  NAM-INDIA   23     FIN SVC    02-Dec-19   290.5       285.9       735    ₹-3,358       -1.6%       -1.8%     

  AFTER: Invested ₹4,540,599 | Cash ₹17,226 | Total ₹4,557,825 | Positions 18/20 | Slot ₹227,959

========================================================================
  REBALANCE #32  —  01 Apr 2020
  NAV: ₹3,660,983  |  Slot: ₹183,049  |  Cash: ₹17,226
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, DIVISLAB

  [REGIME OFF] Nifty 200 4,275.7 < EMA200 5,815.6 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  1      HEALTH     01-Oct-19   9,965.1     14,325.7    20     ₹87,211       +43.8%      +6.4%     
  BATAINDIA   64     CON DUR    01-Aug-18   878.9       1,116.5     222    ₹52,727       +27.0% ⚠    -9.2%     
  IPCALAB     2      HEALTH     03-Feb-20   583.3       686.1       390    ₹40,115       +17.6%      +4.0%     
  NESTLEIND   3      FMCG       01-Oct-19   636.8       731.5       322    ₹30,481       +14.9%      +3.6%     
  BERGEPAINT  7      CON DUR    01-Oct-19   350.7       392.5       586    ₹24,481       +11.9%      +0.5%     
  DMART       6      FMCG       01-Oct-19   1,896.0     2,082.7     108    ₹20,164       +9.8%       +1.1%     
  PIIND       21     MFG        03-Jun-19   1,116.0     1,175.0     187    ₹11,036       +5.3%       -3.7%     
  LALPATHLAB  12     HEALTH     01-Oct-19   634.6       664.5       324    ₹9,694        +4.7%       -5.6%     
  GUJGASLTD   9      OIL&GAS    02-Dec-19   206.7       216.2       1033   ₹9,851        +4.6%       -6.3%     
  IGL         13     OIL&GAS    02-Dec-19   185.5       174.2       1151   ₹-13,037      -6.1%       +2.5%     
  WHIRLPOOL   30     CON DUR    01-Oct-19   1,921.6     1,742.9     107    ₹-19,122      -9.3%       -8.2%     
  COROMANDEL  11     MFG        03-Feb-20   585.9       497.7       389    ₹-34,335      -15.1%      -3.2%     
  HDFCAMC     20     FIN SVC    01-Oct-19   1,208.6     962.5       170    ₹-41,839      -20.4%      -9.0%     
  NAM-INDIA   18     FIN SVC    02-Dec-19   290.5       215.1       735    ₹-55,365      -25.9%      -6.1%     
  JUBLFOOD    27     CONSUMP    03-Feb-20   385.8       273.9       590    ₹-66,047      -29.0%      -5.2%     
  SRF         28     MFG        03-Feb-20   741.4       520.3       307    ₹-67,869      -29.8%      -14.5%    
  COFORGE     34     IT         03-Feb-20   349.4       216.9       652    ₹-86,409      -37.9%      -10.1%    
  MANAPPURAM  61     FIN SVC    02-Dec-19   139.6       83.7        1529   ₹-85,535      -40.1%      -16.9%    
  ⚠  WAZ < 0 (momentum below universe mean): BATAINDIA

  AFTER: Invested ₹3,643,757 | Cash ₹17,226 | Total ₹3,660,983 | Positions 18/20 | Slot ₹183,049

========================================================================
  REBALANCE #33  —  01 Jun 2020
  NAV: ₹4,247,264  |  Slot: ₹212,363  |  Cash: ₹17,226
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, DIVISLAB, SYNGENE, TORNTPHARM, AUROPHARMA, SUNPHARMA

  [REGIME OFF] Nifty 200 5,087.4 < EMA200 5,490.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  1      HEALTH     01-Oct-19   9,965.1     15,437.7    20     ₹109,452      +54.9%      -0.9%     
  BATAINDIA   —      CON DUR    01-Aug-18   878.9       1,302.1     222    ₹93,946       +48.1%      +5.7%     
  PIIND       16     MFG        03-Jun-19   1,116.0     1,552.3     187    ₹81,588       +39.1%      +4.1%     
  IPCALAB     7      HEALTH     03-Feb-20   583.3       749.6       390    ₹64,849       +28.5%      -2.1%     
  NESTLEIND   5      FMCG       01-Oct-19   636.8       802.9       322    ₹53,485       +26.1%      +2.0%     
  DMART       9      FMCG       01-Oct-19   1,896.0     2,305.5     108    ₹44,226       +21.6%      +0.1%     
  LALPATHLAB  23     HEALTH     01-Oct-19   634.6       728.6       324    ₹30,456       +14.8%      -1.4%     
  BERGEPAINT  21     CON DUR    01-Oct-19   350.7       397.9       586    ₹27,621       +13.4%      +4.6%     
  GUJGASLTD   26     OIL&GAS    02-Dec-19   206.7       233.9       1033   ₹28,174       +13.2%      +0.9%     
  IGL         12     OIL&GAS    02-Dec-19   185.5       209.8       1151   ₹28,023       +13.1%      +2.0%     
  WHIRLPOOL   24     CON DUR    01-Oct-19   1,921.6     1,998.5     107    ₹8,223        +4.0%       +6.4%     
  COROMANDEL  6      MFG        03-Feb-20   585.9       605.6       389    ₹7,656        +3.4%       +4.7%     
  SRF         35     MFG        03-Feb-20   741.4       726.6       307    ₹-4,560       -2.0%       +6.3%     
  HDFCAMC     25     FIN SVC    01-Oct-19   1,208.6     1,168.9     170    ₹-6,756       -3.3%       +5.5%     
  JUBLFOOD    —      CONSUMP    03-Feb-20   385.8       331.8       590    ₹-31,836      -14.0%      +4.1%     
  MANAPPURAM  54     FIN SVC    02-Dec-19   139.6       117.8       1529   ₹-33,415      -15.7%      +9.3%     
  COFORGE     46     IT         03-Feb-20   349.4       272.4       652    ₹-50,189      -22.0% ⚠    +4.4%     
  NAM-INDIA   —      FIN SVC    02-Dec-19   290.5       224.5       735    ₹-48,459      -22.7%      +10.0%    
  ⚠  WAZ < 0 (momentum below universe mean): COFORGE

  AFTER: Invested ₹4,230,038 | Cash ₹17,226 | Total ₹4,247,264 | Positions 18/20 | Slot ₹212,363

========================================================================
  REBALANCE #34  —  03 Aug 2020
  NAV: ₹4,534,129  |  Slot: ₹226,706  |  Cash: ₹17,226
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, TORNTPHARM, DIVISLAB, INFY, AJANTPHARM

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BATAINDIA   —      CON DUR    01-Aug-18   878.9       1,180.7     222    ₹66,979       +34.3%    733d  
  NESTLEIND   51     FMCG       01-Oct-19   636.8       775.0       322    ₹44,496       +21.7%    307d  
  DMART       54     FMCG       01-Oct-19   1,896.0     2,093.1     108    ₹21,292       +10.4%    307d  
  WHIRLPOOL   46     CON DUR    01-Oct-19   1,921.6     2,080.9     107    ₹17,040       +8.3%     307d  
  MANAPPURAM  37     FIN SVC    02-Dec-19   139.6       142.6       1529   ₹4,596        +2.2%     245d  
  SRF         48     MFG        03-Feb-20   741.4       751.3       307    ₹3,036        +1.3%     182d  
  IGL         88     OIL&GAS    02-Dec-19   185.5       175.4       1151   ₹-11,574      -5.4%     245d  
  HDFCAMC     77     FIN SVC    01-Oct-19   1,208.6     1,080.2     170    ₹-21,835      -10.6%    307d  
  JUBLFOOD    —      CONSUMP    03-Feb-20   385.8       344.3       590    ₹-24,511      -10.8%    182d  
  NAM-INDIA   —      FIN SVC    02-Dec-19   290.5       220.6       735    ₹-51,365      -24.1%    245d  

  ENTRIES (9)
  [52w filter blocked 1: ENDURANCE(-22.9%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JUBILANT    1      CONSUMP    3.755    0.79   +100.3%   +127.5%   618.2       366    ₹226,254      +16.9%    
  SYNGENE     3      HEALTH     2.811    0.44   +48.5%    +51.9%    473.6       478    ₹226,365      +7.9%     
  MUTHOOTFIN  4      FIN SVC    2.672    1.12   +115.6%   +64.0%    1,172.9     193    ₹226,371      +4.0%     
  MPHASIS     5      IT         2.547    0.54   +30.2%    +66.8%    1,024.9     221    ₹226,512      +9.8%     
  BALKRISIND  6      MFG        2.349    0.94   +81.9%    +47.5%    1,249.0     181    ₹226,060      +3.8%     
  WIPRO       10     IT         1.986    0.61   +6.9%     +52.9%    129.8       1746   ₹226,579      +7.9%     
  HCLTECH     11     IT         1.960    0.71   +41.1%    +36.1%    559.9       404    ₹226,188      +8.0%     
  BRITANNIA   19     FMCG       1.591    0.74   +40.7%    +26.8%    3,411.0     66     ₹225,128      +0.6%     
  ATGL        21     OIL&GAS    1.506    0.73   -3.8%     +60.8%    156.2       1451   ₹226,706      +3.5%     

  HOLDS (8)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  PIIND       15     MFG        03-Jun-19   1,116.0     1,809.4     187    ₹129,665      +62.1%      +6.6%     
  IPCALAB     7      HEALTH     03-Feb-20   583.3       927.0       390    ₹134,038      +58.9%      +7.8%     
  ABBOTINDIA  35     HEALTH     01-Oct-19   9,965.1     14,688.6    20     ₹94,469       +47.4%      +4.1%     
  GUJGASLTD   9      OIL&GAS    02-Dec-19   206.7       291.7       1033   ₹87,871       +41.2%      +3.6%     
  LALPATHLAB  14     HEALTH     01-Oct-19   634.6       892.7       324    ₹83,608       +40.7%      +0.5%     
  COROMANDEL  2      MFG        03-Feb-20   585.9       737.5       389    ₹58,957       +25.9%      +1.3%     
  BERGEPAINT  20     CON DUR    01-Oct-19   350.7       426.3       586    ₹44,298       +21.6%      +1.5%     
  COFORGE     16     IT         03-Feb-20   349.4       362.1       652    ₹8,289        +3.6%       +12.5%    

  AFTER: Invested ₹4,393,195 | Cash ₹138,516 | Total ₹4,531,711 | Positions 17/20 | Slot ₹226,706

========================================================================
  REBALANCE #35  —  01 Oct 2020
  NAV: ₹4,896,162  |  Slot: ₹244,808  |  Cash: ₹138,516
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, APOLLOHOSP, CIPLA, SANOFI, TORNTPHARM

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LALPATHLAB  28     HEALTH     01-Oct-19   634.6       891.4       324    ₹83,205       +40.5%    366d  
  COFORGE     11     IT         03-Feb-20   349.4       438.6       652    ₹58,131       +25.5%    241d  
  BRITANNIA   37     FMCG       03-Aug-20   3,411.0     3,515.2     66     ₹6,873        +3.1%     59d   
  JUBILANT    —      OTHER      03-Aug-20   618.2       527.1       366    ₹-33,339      -14.7%    59d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIENT    2      METAL      3.375    1.08   +103.2%   +87.7%    307.3       796    ₹244,584      +8.0%     
  DIVISLAB    3      HEALTH     3.215    0.55   +84.3%    +41.8%    2,973.1     82     ₹243,791      -1.9%     
  NAVINFLUOR  5      MFG        2.919    0.71   +180.4%   +24.5%    2,095.0     116    ₹243,020      +3.6%     
  INFY        15     IT         1.843    0.86   +33.5%    +33.2%    867.6       282    ₹244,653      +3.0%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  IPCALAB     2      HEALTH     03-Feb-20   583.3       1,106.3     390    ₹203,983      +89.7%      +7.1%     
  PIIND       19     MFG        03-Jun-19   1,116.0     1,913.5     187    ₹149,125      +71.5%      +0.2%     
  ABBOTINDIA  23     HEALTH     01-Oct-19   9,965.1     15,256.9    20     ₹105,835      +53.1%      +0.2%     
  GUJGASLTD   26     OIL&GAS    02-Dec-19   206.7       297.7       1033   ₹94,077       +44.1%      +1.9%     
  BERGEPAINT  21     CON DUR    01-Oct-19   350.7       479.7       586    ₹75,582       +36.8%      +2.9%     
  COROMANDEL  14     MFG        03-Feb-20   585.9       749.7       389    ₹63,684       +27.9%      +0.9%     
  ATGL        27     OIL&GAS    03-Aug-20   156.2       193.0       1451   ₹53,381       +23.5%      +2.2%     
  MPHASIS     8      IT         03-Aug-20   1,024.9     1,215.0     221    ₹42,013       +18.5%      +4.9%     
  SYNGENE     7      HEALTH     03-Aug-20   473.6       554.2       478    ₹38,548       +17.0%      +3.6%     
  HCLTECH     6      IT         03-Aug-20   559.9       646.5       404    ₹35,007       +15.5%      +3.5%     
  BALKRISIND  17     MFG        03-Aug-20   1,249.0     1,397.6     181    ₹26,900       +11.9%      +6.1%     
  WIPRO       10     IT         03-Aug-20   129.8       144.3       1746   ₹25,350       +11.2%      +3.1%     
  MUTHOOTFIN  32     FIN SVC    03-Aug-20   1,172.9     1,055.2     193    ₹-22,709      -10.0%      +3.8%     

  AFTER: Invested ₹4,734,017 | Cash ₹160,985 | Total ₹4,895,003 | Positions 17/20 | Slot ₹244,808

========================================================================
  REBALANCE #36  —  01 Dec 2020
  NAV: ₹5,566,305  |  Slot: ₹278,315  |  Cash: ₹160,985
========================================================================
  [SECTOR CAP≤4] dropped: MRF, TATACHEM, LALPATHLAB, SRF

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ABBOTINDIA  87     HEALTH     01-Oct-19   9,965.1     14,338.5    20     ₹87,467       +43.9%    427d  
  MUTHOOTFIN  51     FIN SVC    03-Aug-20   1,172.9     1,055.1     193    ₹-22,727      -10.0%    120d  

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  APOLLOHOSP  3      HEALTH     2.631    -0.09  +71.1%    +47.7%    2,431.2     114    ₹277,158      +8.8%     
  ADANIENSOL  16     ENERGY     1.772    -0.17  +35.4%    +43.8%    379.0       734    ₹278,149      +8.4%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       360.4       1451   ₹296,167      +130.6%     +22.2%    
  PIIND       13     MFG        03-Jun-19   1,116.0     2,294.6     187    ₹220,401      +105.6%     +1.8%     
  IPCALAB     11     HEALTH     03-Feb-20   583.3       1,120.5     390    ₹209,502      +92.1%      +5.0%     
  GUJGASLTD   24     OIL&GAS    02-Dec-19   206.7       326.4       1033   ₹123,709      +58.0%      +5.8%     
  BERGEPAINT  23     CON DUR    01-Oct-19   350.7       532.5       586    ₹106,501      +51.8%      +2.4%     
  ADANIENT    4      METAL      01-Oct-20   307.3       420.7       796    ₹90,285       +36.9%      +10.8%    
  COROMANDEL  17     MFG        03-Feb-20   585.9       778.3       389    ₹74,811       +32.8%      +6.0%     
  BALKRISIND  12     MFG        03-Aug-20   1,249.0     1,584.9     181    ₹60,803       +26.9%      +5.6%     
  NAVINFLUOR  2      MFG        01-Oct-20   2,095.0     2,638.5     116    ₹63,048       +25.9%      +6.2%     
  WIPRO       9      IT         03-Aug-20   129.8       162.6       1746   ₹57,379       +25.3%      +1.5%     
  HCLTECH     18     IT         03-Aug-20   559.9       666.4       404    ₹43,039       +19.0%      +0.6%     
  SYNGENE     6      HEALTH     03-Aug-20   473.6       563.1       478    ₹42,784       +18.9%      +1.1%     
  DIVISLAB    5      HEALTH     01-Oct-20   2,973.1     3,512.3     82     ₹44,214       +18.1%      +5.7%     
  MPHASIS     22     IT         03-Aug-20   1,024.9     1,173.6     221    ₹32,861       +14.5%      -1.1%     
  INFY        8      IT         01-Oct-20   867.6       980.5       282    ₹31,836       +13.0%      +2.3%     

  AFTER: Invested ₹5,470,215 | Cash ₹95,431 | Total ₹5,565,646 | Positions 17/20 | Slot ₹278,315

========================================================================
  REBALANCE #37  —  01 Feb 2021
  NAV: ₹5,814,771  |  Slot: ₹290,739  |  Cash: ₹95,431
========================================================================
  [SECTOR CAP≤4] dropped: MRF

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PIIND       90     MFG        03-Jun-19   1,116.0     1,966.5     187    ₹159,048      +76.2%    609d  
  BERGEPAINT  44     CON DUR    01-Oct-19   350.7       585.6       586    ₹137,610      +67.0%    489d  
  GUJGASLTD   50     OIL&GAS    02-Dec-19   206.7       344.2       1033   ₹142,062      +66.5%    427d  
  IPCALAB     85     HEALTH     03-Feb-20   583.3       924.7       390    ₹133,168      +58.5%    364d  
  COROMANDEL  42     MFG        03-Feb-20   585.9       786.2       389    ₹77,919       +34.2%    364d  
  HCLTECH     32     IT         03-Aug-20   559.9       744.7       404    ₹74,665       +33.0%    182d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TATACHEM    3      MFG        2.646    0.11   +55.0%    +53.3%    453.6       640    ₹290,332      -1.5%     
  CROMPTON    4      CON DUR    2.379    -0.05  +62.7%    +40.6%    400.6       725    ₹290,438      +3.1%     
  HINDZINC    5      METAL      2.297    0.10   +75.0%    +42.0%    188.7       1541   ₹290,735      +5.7%     
  BAJAJ-AUTO  7      AUTO       2.091    -0.00  +40.5%    +42.5%    3,548.3     81     ₹287,412      +8.6%     
  LTTS        9      IT         1.944    -0.13  +48.4%    +48.5%    2,321.4     125    ₹290,181      +1.8%     
  LT          10     INFRA      1.926    0.15   +12.2%    +58.9%    1,361.6     213    ₹290,016      +7.5%     
  HONAUT      13     MFG        1.749    -0.11  +48.0%    +43.6%    40,417.2    7      ₹282,920      +5.0%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       388.5       1451   ₹336,945      +148.6%     +5.3%     
  ADANIENT    2      METAL      01-Oct-20   307.3       535.7       796    ₹181,801      +74.3%      +4.7%     
  WIPRO       8      IT         03-Aug-20   129.8       194.7       1746   ₹113,388      +50.0%      -0.9%     
  MPHASIS     27     IT         03-Aug-20   1,024.9     1,370.0     221    ₹76,259       +33.7%      -3.1%     
  ADANIENSOL  6      ENERGY     01-Dec-20   379.0       481.5       734    ₹75,272       +27.1%      +6.6%     
  BALKRISIND  29     MFG        03-Aug-20   1,249.0     1,570.8     181    ₹58,254       +25.8%      -0.9%     
  INFY        12     IT         01-Oct-20   867.6       1,086.5     282    ₹61,736       +25.2%      -2.6%     
  SYNGENE     16     HEALTH     03-Aug-20   473.6       556.1       478    ₹39,447       +17.4%      -5.5%     
  DIVISLAB    11     HEALTH     01-Oct-20   2,973.1     3,359.8     82     ₹31,714       +13.0%      -3.7%     
  NAVINFLUOR  26     MFG        01-Oct-20   2,095.0     2,307.2     116    ₹24,615       +10.1%      -6.4%     
  APOLLOHOSP  18     HEALTH     01-Dec-20   2,431.2     2,629.1     114    ₹22,562       +8.1%       +3.7%     

  AFTER: Invested ₹5,707,603 | Cash ₹104,767 | Total ₹5,812,370 | Positions 18/20 | Slot ₹290,739

========================================================================
  REBALANCE #38  —  01 Apr 2021
  NAV: ₹7,961,337  |  Slot: ₹398,067  |  Cash: ₹104,767
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  WIPRO       33     IT         03-Aug-20   129.8       192.4       1746   ₹109,275      +48.2%    241d  
  NAVINFLUOR  47     MFG        01-Oct-20   2,095.0     2,731.4     116    ₹73,824       +30.4%    182d  
  BALKRISIND  48     MFG        03-Aug-20   1,249.0     1,615.7     181    ₹66,375       +29.4%    241d  
  DIVISLAB    70     HEALTH     01-Oct-20   2,973.1     3,507.8     82     ₹43,844       +18.0%    182d  
  SYNGENE     54     HEALTH     03-Aug-20   473.6       552.4       478    ₹37,696       +16.7%    241d  
  LT          38     INFRA      01-Feb-21   1,361.6     1,357.6     213    ₹-851         -0.3%     59d   
  HINDZINC    28     METAL      01-Feb-21   188.7       183.1       1541   ₹-8,558       -2.9%     59d   
  CROMPTON    44     CON DUR    01-Feb-21   400.6       378.3       725    ₹-16,180      -5.6%     59d   
  BAJAJ-AUTO  45     AUTO       01-Feb-21   3,548.3     3,227.8     81     ₹-25,960      -9.0%     59d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  DEEPAKNTR   5      MFG        2.783    0.00   +338.0%   +77.5%    1,618.8     245    ₹396,604      +7.1%     
  GRASIM      6      INFRA      2.606    0.06   +217.5%   +55.7%    1,412.7     281    ₹396,973      +5.6%     
  TATAELXSI   7      IT         2.572    0.02   +341.6%   +49.7%    2,599.1     153    ₹397,660      +2.7%     
  DIXON       8      CON DUR    2.507    0.03   +429.0%   +34.2%    3,580.2     111    ₹397,407      -5.8%     
  LAURUSLABS  9      HEALTH     1.956    0.01   +462.4%   +4.5%     359.2       1108   ₹398,010      +2.0%     
  GUJGASLTD   10     OIL&GAS    1.826    0.05   +141.0%   +43.3%    524.7       758    ₹397,698      +5.5%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       1,058.9     1451   ₹1,309,753    +577.7%     +31.6%    
  ADANIENT    2      METAL      01-Oct-20   307.3       1,104.0     796    ₹634,219      +259.3%     +16.5%    
  ADANIENSOL  3      ENERGY     01-Dec-20   379.0       999.2       734    ₹455,264      +163.7%     +21.3%    
  TATACHEM    4      MFG        01-Feb-21   453.6       715.4       640    ₹167,511      +57.7%      +5.5%     
  MPHASIS     27     IT         03-Aug-20   1,024.9     1,565.1     221    ₹119,371      +52.7%      +3.6%     
  INFY        25     IT         01-Oct-20   867.6       1,193.6     282    ₹91,940       +37.6%      +2.5%     
  APOLLOHOSP  19     HEALTH     01-Dec-20   2,431.2     2,856.7     114    ₹48,505       +17.5%      -1.0%     
  HONAUT      24     MFG        01-Feb-21   40,417.2    45,601.3    7      ₹36,289       +12.8%      +0.5%     
  LTTS        26     IT         01-Feb-21   2,321.4     2,550.6     125    ₹28,641       +9.9%       +3.8%     

  AFTER: Invested ₹7,637,041 | Cash ₹321,464 | Total ₹7,958,506 | Positions 15/20 | Slot ₹398,067

========================================================================
  REBALANCE #39  —  01 Jun 2021
  NAV: ₹9,506,007  |  Slot: ₹475,300  |  Cash: ₹321,464
========================================================================
  [SECTOR CAP≤4] dropped: POLYCAB

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIENT    —      METAL      01-Oct-20   307.3       1,412.2     796    ₹879,517      +359.6%   243d  
  TATACHEM    —      MFG        01-Feb-21   453.6       648.6       640    ₹124,789      +43.0%    120d  
  APOLLOHOSP  —      HEALTH     01-Dec-20   2,431.2     3,198.0     114    ₹87,412       +31.5%    182d  
  LTTS        —      IT         01-Feb-21   2,321.4     2,519.1     125    ₹24,703       +8.5%     120d  
  HONAUT      —      MFG        01-Feb-21   40,417.2    41,057.7    7      ₹4,484        +1.6%     120d  
  GRASIM      —      INFRA      01-Apr-21   1,412.7     1,403.1     281    ₹-2,716       -0.7%     61d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JSWSTEEL    3      METAL      3.491    0.45   +281.0%   +70.2%    657.4       722    ₹474,663      +0.4%     
  TATASTEEL   6      METAL      2.684    0.72   +282.6%   +51.3%    92.9        5118   ₹475,252      +0.6%     
  BSE         7      FIN SVC    2.506    0.40   +155.2%   +60.8%    97.3        4884   ₹475,260      +16.9%    
  SAIL        8      METAL      2.487    0.68   +308.4%   +69.7%    104.5       4547   ₹475,300      -0.8%     
  DALBHARAT   9      MFG        1.948    0.60   +227.2%   +25.2%    1,770.4     268    ₹474,469      +3.9%     
  ZYDUSLIFE   11     HEALTH     1.906    0.25   +79.8%    +42.9%    597.4       795    ₹474,972      +2.1%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       1,438.2     1451   ₹1,860,150    +820.5%     +10.6%    
  ADANIENSOL  2      ENERGY     01-Dec-20   379.0       1,499.4     734    ₹822,447      +295.7%     +13.2%    
  MPHASIS     29     IT         03-Aug-20   1,024.9     1,746.1     221    ₹159,386      +70.4%      +5.5%     
  LAURUSLABS  4      HEALTH     01-Apr-21   359.2       524.5       1108   ₹183,119      +46.0%      +7.1%     
  INFY        27     IT         01-Oct-20   867.6       1,208.2     282    ₹96,063       +39.3%      +2.4%     
  TATAELXSI   5      IT         01-Apr-21   2,599.1     3,384.3     153    ₹120,130      +30.2%      +1.8%     
  DIXON       18     CON DUR    01-Apr-21   3,580.2     4,098.5     111    ₹57,523       +14.5%      +3.4%     
  DEEPAKNTR   10     MFG        01-Apr-21   1,618.8     1,730.5     245    ₹27,360       +6.9%       -0.6%     
  GUJGASLTD   37     OIL&GAS    01-Apr-21   524.7       517.6       758    ₹-5,372       -1.4%       +3.1%     

  AFTER: Invested ₹9,134,121 | Cash ₹368,501 | Total ₹9,502,623 | Positions 15/20 | Slot ₹475,300

========================================================================
  REBALANCE #40  —  02 Aug 2021
  NAV: ₹9,455,466  |  Slot: ₹472,773  |  Cash: ₹368,501
========================================================================
  [SECTOR CAP≤4] dropped: POLYCAB, PIDILITIND

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIENSOL  53     ENERGY     01-Dec-20   379.0       908.8       734    ₹388,910      +139.8%   244d  
  DIXON       41     CON DUR    01-Apr-21   3,580.2     4,339.1     111    ₹84,237       +21.2%    123d  
  ZYDUSLIFE   79     HEALTH     01-Jun-21   597.4       574.2       795    ₹-18,494      -3.9%     62d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COFORGE     3      IT         2.735    0.56   +177.1%   +77.8%    969.2       487    ₹472,000      +10.9%    
  SRF         8      MFG        2.230    0.60   +130.0%   +37.9%    1,773.4     266    ₹471,735      +14.7%    
  ABBOTINDIA  10     HEALTH     2.103    -0.01  +37.1%    +32.3%    18,684.8    25     ₹467,121      +11.5%    
  AMBUJACEM   11     INFRA      2.047    0.59   +108.0%   +34.2%    402.3       1175   ₹472,729      +7.3%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        28     OIL&GAS    03-Aug-20   156.2       887.1       1451   ₹1,060,404    +467.7%     -5.7%     
  MPHASIS     9      IT         03-Aug-20   1,024.9     2,352.2     221    ₹293,322      +129.5%     +8.5%     
  LAURUSLABS  5      HEALTH     01-Apr-21   359.2       644.6       1108   ₹316,260      +79.5%      +1.3%     
  INFY        16     IT         01-Oct-20   867.6       1,421.0     282    ₹156,079      +63.8%      +3.3%     
  TATAELXSI   4      IT         01-Apr-21   2,599.1     4,010.4     153    ₹215,927      +54.3%      +0.6%     
  GUJGASLTD   7      OIL&GAS    01-Apr-21   524.7       722.7       758    ₹150,135      +37.8%      +8.3%     
  BSE         1      FIN SVC    01-Jun-21   97.3        133.3       4884   ₹175,643      +37.0%      +10.0%    
  TATASTEEL   2      METAL      01-Jun-21   92.9        121.6       5118   ₹147,080      +30.9%      +8.9%     
  DEEPAKNTR   24     MFG        01-Apr-21   1,618.8     2,032.0     245    ₹101,239      +25.5%      +6.9%     
  DALBHARAT   6      MFG        01-Jun-21   1,770.4     2,108.0     268    ₹90,474       +19.1%      +1.3%     
  SAIL        19     METAL      01-Jun-21   104.5       120.3       4547   ₹71,765       +15.1%      +6.6%     
  JSWSTEEL    13     METAL      01-Jun-21   657.4       713.8       722    ₹40,666       +8.6%       +5.1%     

  AFTER: Invested ₹9,365,368 | Cash ₹87,861 | Total ₹9,453,229 | Positions 16/20 | Slot ₹472,773

========================================================================
  REBALANCE #41  —  01 Oct 2021
  NAV: ₹10,433,963  |  Slot: ₹521,698  |  Cash: ₹87,861
========================================================================
  [SECTOR CAP≤4] dropped: TECHM

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LAURUSLABS  74     HEALTH     01-Apr-21   359.2       609.9       1108   ₹277,737      +69.8%    183d  
  GUJGASLTD   83     OIL&GAS    01-Apr-21   524.7       589.5       758    ₹49,127       +12.4%    183d  
  ABBOTINDIA  —      OTHER      02-Aug-21   18,684.8    20,842.3    25     ₹53,937       +11.5%    60d   
  JSWSTEEL    44     METAL      01-Jun-21   657.4       644.5       722    ₹-9,334       -2.0%     122d  
  SAIL        48     METAL      01-Jun-21   104.5       100.6       4547   ₹-17,949      -3.8%     122d  

  ENTRIES (5)
  [52w filter blocked 1: ADANIENSOL(-20.7%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IRCTC       3      PSE        3.381    0.73   +181.5%   +86.7%    725.8       718    ₹521,108      +6.9%     
  BAJAJFINSV  6      FIN SVC    2.758    1.15   +196.1%   +45.4%    1,712.9     304    ₹520,715      -0.1%     
  BAJAJHLDNG  7      FIN SVC    2.404    0.47   +99.7%    +34.8%    4,433.3     117    ₹518,695      +4.4%     
  PRESTIGE    8      REALTY     2.353    0.77   +97.5%    +69.3%    477.2       1093   ₹521,587      +8.8%     
  POLYCAB     11     MFG        2.085    0.48   +184.7%   +17.1%    2,283.5     228    ₹520,638      +0.5%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        2      OIL&GAS    03-Aug-20   156.2       1,421.5     1451   ₹1,835,915    +809.8%     +3.2%     
  MPHASIS     9      IT         03-Aug-20   1,024.9     2,770.0     221    ₹385,665      +170.3%     -1.8%     
  TATAELXSI   4      IT         01-Apr-21   2,599.1     5,491.9     153    ₹442,601      +111.3%     +6.9%     
  INFY        40     IT         01-Oct-20   867.6       1,450.3     282    ₹164,332      +67.2%      -2.0%     
  DEEPAKNTR   11     MFG        01-Apr-21   1,618.8     2,348.6     245    ₹178,803      +45.1%      +0.2%     
  BSE         15     FIN SVC    01-Jun-21   97.3        129.4       4884   ₹156,547      +32.9%      +1.3%     
  SRF         5      MFG        02-Aug-21   1,773.4     2,186.9     266    ₹109,985      +23.3%      +3.6%     
  TATASTEEL   12     METAL      01-Jun-21   92.9        111.9       5118   ₹97,560       +20.5%      -3.1%     
  DALBHARAT   21     MFG        01-Jun-21   1,770.4     2,064.4     268    ₹78,795       +16.6%      -1.7%     
  COFORGE     29     IT         02-Aug-21   969.2       997.8       487    ₹13,931       +3.0%       -0.5%     
  AMBUJACEM   18     INFRA      02-Aug-21   402.3       387.1       1175   ₹-17,921      -3.8%       -3.1%     

  AFTER: Invested ₹10,382,535 | Cash ₹48,337 | Total ₹10,430,872 | Positions 16/20 | Slot ₹521,698

========================================================================
  REBALANCE #42  —  01 Dec 2021
  NAV: ₹10,723,675  |  Slot: ₹536,184  |  Cash: ₹48,337
========================================================================
  [SECTOR CAP≤4] dropped: TECHM

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DALBHARAT   66     MFG        01-Jun-21   1,770.4     1,797.2     268    ₹7,167        +1.5%     183d  
  BAJAJFINSV  25     FIN SVC    01-Oct-21   1,712.9     1,733.3     304    ₹6,219        +1.2%     61d   
  TATASTEEL   74     METAL      01-Jun-21   92.9        93.4        5118   ₹2,537        +0.5%     183d  
  AMBUJACEM   81     INFRA      02-Aug-21   402.3       357.4       1175   ₹-52,802      -11.2%    121d  

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIENSOL  2      ENERGY     3.199    0.85   +373.4%   +19.4%    1,797.2     298    ₹535,551      -4.2%     
  ZEEL        5      MEDIA      2.823    1.08   +82.7%    +97.6%    324.0       1655   ₹536,140      +4.7%     
  DIXON       9      CON DUR    2.316    0.80   +137.1%   +23.4%    5,067.9     105    ₹532,130      -2.5%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        4      OIL&GAS    03-Aug-20   156.2       1,623.7     1451   ₹2,129,312    +939.2%     +1.9%     
  MPHASIS     15     IT         03-Aug-20   1,024.9     2,756.6     221    ₹382,701      +169.0%     -6.0%     
  TATAELXSI   1      IT         01-Apr-21   2,599.1     5,587.5     153    ₹457,225      +115.0%     -3.0%     
  BSE         5      FIN SVC    01-Jun-21   97.3        175.3       4884   ₹380,919      +80.1%      +9.5%     
  INFY        35     IT         01-Oct-20   867.6       1,506.9     282    ₹180,284      +73.7%      -0.7%     
  DEEPAKNTR   24     MFG        01-Apr-21   1,618.8     2,120.9     245    ₹123,012      +31.0%      -4.0%     
  SRF         30     MFG        02-Aug-21   1,773.4     1,988.5     266    ₹57,204       +12.1%      -5.0%     
  BAJAJHLDNG  8      FIN SVC    01-Oct-21   4,433.3     4,853.5     117    ₹49,166       +9.5%       +4.8%     
  IRCTC       6      PSE        01-Oct-21   725.8       776.7       718    ₹36,560       +7.0%       -4.6%     
  COFORGE     26     IT         02-Aug-21   969.2       1,019.2     487    ₹24,332       +5.2%       -0.2%     
  POLYCAB     9      MFG        01-Oct-21   2,283.5     2,268.6     228    ₹-3,394       -0.7%       -1.5%     
  PRESTIGE    23     REALTY     01-Oct-21   477.2       439.3       1093   ₹-41,428      -7.9%       -2.2%     

  AFTER: Invested ₹10,372,871 | Cash ₹348,899 | Total ₹10,721,770 | Positions 15/20 | Slot ₹536,184

========================================================================
  REBALANCE #43  —  01 Feb 2022
  NAV: ₹11,641,137  |  Slot: ₹582,057  |  Cash: ₹348,899
========================================================================

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DIXON       74     CON DUR    01-Dec-21   5,067.9     4,429.2     105    ₹-67,064      -12.6%    62d   
  ZEEL        69     MEDIA      01-Dec-21   324.0       278.3       1655   ₹-75,478      -14.1%    62d   

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ONGC        6      OIL&GAS    2.390    0.96   +100.5%   +15.7%    131.8       4416   ₹581,933      +5.2%     
  ESCORTS     8      MFG        2.308    0.58   +51.7%    +19.6%    1,800.2     323    ₹581,477      -0.5%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        2      OIL&GAS    03-Aug-20   156.2       1,861.9     1451   ₹2,474,942    +1091.7%    +3.3%     
  MPHASIS     29     IT         03-Aug-20   1,024.9     2,880.5     221    ₹410,081      +181.0%     +0.4%     
  TATAELXSI   3      IT         01-Apr-21   2,599.1     7,089.3     153    ₹687,008      +172.8%     +10.5%    
  BSE         1      FIN SVC    01-Jun-21   97.3        211.4       4884   ₹557,167      +117.2%     +0.2%     
  INFY        25     IT         01-Oct-20   867.6       1,557.1     282    ₹194,445      +79.5%      -1.4%     
  DEEPAKNTR   22     MFG        01-Apr-21   1,618.8     2,232.1     245    ₹150,267      +37.9%      -5.7%     
  SRF         5      MFG        02-Aug-21   1,773.4     2,413.0     266    ₹170,115      +36.1%      -0.3%     
  IRCTC       12     PSE        01-Oct-21   725.8       819.7       718    ₹67,432       +12.9%      +0.5%     
  ADANIENSOL  4      ENERGY     01-Dec-21   1,797.2     1,986.9     298    ₹56,546       +10.6%      +1.7%     
  BAJAJHLDNG  10     FIN SVC    01-Oct-21   4,433.3     4,889.1     117    ₹53,327       +10.3%      -0.4%     
  POLYCAB     7      MFG        01-Oct-21   2,283.5     2,471.0     228    ₹42,740       +8.2%       +0.5%     
  PRESTIGE    23     REALTY     01-Oct-21   477.2       480.5       1093   ₹3,633        +0.7%       -0.7%     
  COFORGE     35     IT         02-Aug-21   969.2       907.8       487    ₹-29,901      -6.3%       -7.7%     

  AFTER: Invested ₹11,529,921 | Cash ₹109,835 | Total ₹11,639,756 | Positions 15/20 | Slot ₹582,057

========================================================================
  REBALANCE #44  —  01 Apr 2022
  NAV: ₹12,981,940  |  Slot: ₹649,097  |  Cash: ₹109,835
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BSE         1      FIN SVC    01-Jun-21   97.3        297.8       4884   ₹978,959      +206.0%   304d  
  DEEPAKNTR   52     MFG        01-Apr-21   1,618.8     2,267.7     245    ₹158,986      +40.1%    365d  
  IRCTC       27     PSE        01-Oct-21   725.8       765.4       718    ₹28,448       +5.5%     182d  
  PRESTIGE    31     REALTY     01-Oct-21   477.2       501.1       1093   ₹26,082       +5.0%     182d  
  ESCORTS     73     MFG        01-Feb-22   1,800.2     1,654.2     323    ₹-47,170      -8.1%     59d   
  COFORGE     84     IT         02-Aug-21   969.2       838.9       487    ₹-63,471      -13.4%    242d  

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HAL         5      DEFENCE    2.725    0.84   +59.8%    +28.3%    723.8       896    ₹648,528      +7.9%     
  PERSISTENT  6      IT         2.636    0.94   +163.5%   -1.4%     2,292.8     283    ₹648,854      +4.7%     
  NTPC        8      ENERGY     2.240    0.80   +46.8%    +15.9%    126.0       5149   ₹649,018      +6.4%     
  POWERGRID   10     ENERGY     2.122    0.59   +50.1%    +12.6%    141.2       4595   ₹648,986      +6.2%     
  RELIANCE    12     OIL&GAS    1.959    1.09   +33.8%    +12.6%    1,202.9     539    ₹648,353      +5.3%     
  BEL         13     DEFENCE    1.945    0.86   +84.2%    +4.0%     68.5        9469   ₹649,055      +3.3%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        6      OIL&GAS    03-Aug-20   156.2       2,246.2     1451   ₹3,032,542    +1337.7%    +16.1%    
  TATAELXSI   2      IT         01-Apr-21   2,599.1     8,464.4     153    ₹897,400      +225.7%     +12.8%    
  MPHASIS     18     IT         03-Aug-20   1,024.9     3,060.9     221    ₹449,952      +198.6%     +2.7%     
  INFY        28     IT         01-Oct-20   867.6       1,672.6     282    ₹227,030      +92.8%      +2.7%     
  SRF         4      MFG        02-Aug-21   1,773.4     2,590.2     266    ₹217,256      +46.1%      +3.3%     
  ADANIENSOL  3      ENERGY     01-Dec-21   1,797.2     2,421.4     298    ₹186,041      +34.7%      +4.1%     
  BAJAJHLDNG  14     FIN SVC    01-Oct-21   4,433.3     5,052.3     117    ₹72,422       +14.0%      +6.5%     
  POLYCAB     25     MFG        01-Oct-21   2,283.5     2,369.9     228    ₹19,701       +3.8%       +2.3%     
  ONGC        10     OIL&GAS    01-Feb-22   131.8       130.8       4416   ₹-4,190       -0.7%       -1.3%     

  AFTER: Invested ₹12,715,032 | Cash ₹262,286 | Total ₹12,977,318 | Positions 15/20 | Slot ₹649,097

========================================================================
  REBALANCE #45  —  01 Jun 2022
  NAV: ₹12,677,148  |  Slot: ₹633,857  |  Cash: ₹262,286
========================================================================

  [REGIME OFF] Nifty 200 8,710.7 < EMA200 8,862.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        5      OIL&GAS    03-Aug-20   156.2       2,322.5     1451   ₹3,143,236    +1386.5%    -2.8%     
  TATAELXSI   2      IT         01-Apr-21   2,599.1     8,173.1     153    ₹852,821      +214.5%     +5.6%     
  MPHASIS     81     IT         03-Aug-20   1,024.9     2,315.3     221    ₹285,167      +125.9% ⚠   -1.7%     
  INFY        111    IT         01-Oct-20   867.6       1,312.9     282    ₹125,599      +51.3% ⚠    -0.8%     
  SRF         18     MFG        02-Aug-21   1,773.4     2,368.5     266    ₹158,286      +33.6%      +1.5%     
  HAL         1      DEFENCE    01-Apr-22   723.8       899.0       896    ₹157,016      +24.2%      +9.7%     
  BEL         4      DEFENCE    01-Apr-22   68.5        78.4        9469   ₹93,541       +14.4%      +6.0%     
  NTPC        7      ENERGY     01-Apr-22   126.0       138.3       5149   ₹62,882       +9.7%       +3.0%     
  ADANIENSOL  —      ENERGY     01-Dec-21   1,797.2     1,957.9     298    ₹47,904       +8.9%       -13.5%    
  BAJAJHLDNG  35     FIN SVC    01-Oct-21   4,433.3     4,726.9     117    ₹34,352       +6.6%       +0.4%     
  POLYCAB     22     MFG        01-Oct-21   2,283.5     2,426.5     228    ₹32,602       +6.3%       +0.4%     
  POWERGRID   10     ENERGY     01-Apr-22   141.2       143.8       4595   ₹11,968       +1.8%       -0.3%     
  RELIANCE    17     OIL&GAS    01-Apr-22   1,202.9     1,192.8     539    ₹-5,456       -0.8%       +1.5%     
  ONGC        49     OIL&GAS    01-Feb-22   131.8       116.7       4416   ₹-66,798      -11.5%      -3.5%     
  PERSISTENT  37     IT         01-Apr-22   2,292.8     1,815.3     283    ₹-135,136     -20.8%      -0.8%     
  ⚠  WAZ < 0 (momentum below universe mean): MPHASIS, INFY

  AFTER: Invested ₹12,414,862 | Cash ₹262,286 | Total ₹12,677,148 | Positions 15/20 | Slot ₹633,857

========================================================================
  REBALANCE #46  —  01 Aug 2022
  NAV: ₹14,453,891  |  Slot: ₹722,695  |  Cash: ₹262,286
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJ-AUTO

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MPHASIS     122    IT         03-Aug-20   1,024.9     2,155.2     221    ₹249,786      +110.3%   728d  
  ADANIENSOL  —      ENERGY     01-Dec-21   1,797.2     3,261.8     298    ₹436,451      +81.5%    243d  
  INFY        80     IT         01-Oct-20   867.6       1,377.3     282    ₹143,754      +58.8%    669d  
  SRF         29     MFG        02-Aug-21   1,773.4     2,427.5     266    ₹173,982      +36.9%    364d  
  POLYCAB     63     MFG        01-Oct-21   2,283.5     2,294.6     228    ₹2,534        +0.5%     304d  
  POWERGRID   62     ENERGY     01-Apr-22   141.2       137.4       4595   ₹-17,591      -2.7%     122d  
  RELIANCE    71     OIL&GAS    01-Apr-22   1,202.9     1,166.2     539    ₹-19,774      -3.0%     122d  
  ONGC        90     OIL&GAS    01-Feb-22   131.8       107.8       4416   ₹-105,841     -18.2%    181d  
  PERSISTENT  93     IT         01-Apr-22   2,292.8     1,780.3     283    ₹-145,039     -22.4%    122d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TVSMOTOR    2      AUTO       3.431    0.78   +62.0%    +47.7%    911.5       792    ₹721,900      +7.4%     
  M&M         3      AUTO       3.336    0.97   +71.6%    +39.5%    1,193.6     605    ₹722,128      +8.2%     
  ITC         5      FMCG       2.676    0.73   +54.3%    +21.3%    254.0       2845   ₹722,582      +3.9%     
  VBL         6      FMCG       2.646    0.63   +83.6%    +27.4%    183.1       3946   ₹722,591      +7.8%     
  SIEMENS     7      ENERGY     2.413    0.88   +42.9%    +25.4%    1,598.0     452    ₹722,290      +3.8%     
  CUMMINSIND  8      INFRA      2.302    0.76   +48.7%    +20.6%    1,157.0     624    ₹721,940      +6.1%     
  EICHERMOT   11     AUTO       1.955    0.96   +21.8%    +24.4%    2,962.8     243    ₹719,965      +2.7%     

  HOLDS (6)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       3,213.2     1451   ₹4,435,709    +1956.6%    +13.8%    
  TATAELXSI   10     IT         01-Apr-21   2,599.1     8,324.1     153    ₹875,927      +220.3%     +5.1%     
  HAL         4      DEFENCE    01-Apr-22   723.8       963.6       896    ₹214,815      +33.1%      +8.3%     
  BEL         9      DEFENCE    01-Apr-22   68.5        90.6        9469   ₹208,506      +32.1%      +10.0%    
  BAJAJHLDNG  43     FIN SVC    01-Oct-21   4,433.3     4,946.0     117    ₹59,989       +11.6%      +7.1%     
  NTPC        38     ENERGY     01-Apr-22   126.0       138.0       5149   ₹61,519       +9.5%       +4.9%     

  AFTER: Invested ₹13,999,523 | Cash ₹448,367 | Total ₹14,447,890 | Positions 13/20 | Slot ₹722,695

========================================================================
  REBALANCE #47  —  03 Oct 2022
  NAV: ₹14,831,700  |  Slot: ₹741,585  |  Cash: ₹448,367
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        5      OIL&GAS    03-Aug-20   156.2       3,096.6     1451   ₹4,266,392    +1881.9%    -10.2%    
  TATAELXSI   51     IT         01-Apr-21   2,599.1     7,934.6     153    ₹816,327      +205.3%     -4.7%     
  HAL         2      DEFENCE    01-Apr-22   723.8       1,092.9     896    ₹330,685      +51.0%      -3.3%     
  BAJAJHLDNG  4      FIN SVC    01-Oct-21   4,433.3     6,221.9     117    ₹209,267      +40.3%      +1.1%     
  BEL         8      DEFENCE    01-Apr-22   68.5        94.6        9469   ₹247,018      +38.1%      -5.2%     
  VBL         3      FMCG       01-Aug-22   183.1       212.1       3946   ₹114,521      +15.8%      +0.2%     
  NTPC        24     ENERGY     01-Apr-22   126.0       144.1       5149   ₹92,964       +14.3%      -1.8%     
  EICHERMOT   18     AUTO       01-Aug-22   2,962.8     3,344.6     243    ₹92,781       +12.9%      -2.5%     
  TVSMOTOR    6      AUTO       01-Aug-22   911.5       979.2       792    ₹53,637       +7.4%       -2.8%     
  ITC         11     FMCG       01-Aug-22   254.0       267.9       2845   ₹39,713       +5.5%       -1.9%     
  M&M         13     AUTO       01-Aug-22   1,193.6     1,206.7     605    ₹7,934        +1.1%       -1.5%     
  SIEMENS     31     ENERGY     01-Aug-22   1,598.0     1,568.0     452    ₹-13,575      -1.9%       -4.7%     
  CUMMINSIND  29     INFRA      01-Aug-22   1,157.0     1,129.1     624    ₹-17,388      -2.4%       -1.9%     

  AFTER: Invested ₹14,383,333 | Cash ₹448,367 | Total ₹14,831,700 | Positions 13/20 | Slot ₹741,585

========================================================================
  REBALANCE #48  —  01 Dec 2022
  NAV: ₹16,101,777  |  Slot: ₹805,089  |  Cash: ₹448,367
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ATGL        21     OIL&GAS    03-Aug-20   156.2       3,607.5     1451   ₹5,007,706    +2208.9%  850d  
  TATAELXSI   116    IT         01-Apr-21   2,599.1     6,750.1     153    ₹635,109      +159.7%   609d  
  SIEMENS     50     ENERGY     01-Aug-22   1,598.0     1,607.3     452    ₹4,215        +0.6%     122d  

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  UNIONBANK   1      PSU BNK    4.613    1.16   +96.9%    +92.4%    72.2        11154  ₹805,027      +14.1%    
  BANKINDIA   3      PSU BNK    3.185    1.19   +53.1%    +60.1%    73.9        10893  ₹805,041      +9.9%     
  INDIANB     4      PSU BNK    3.127    1.19   +94.8%    +42.1%    250.0       3220   ₹804,856      +4.6%     
  AMBUJACEM   5      INFRA      2.983    0.84   +59.3%    +41.3%    570.9       1410   ₹804,996      +3.4%     
  SUNPHARMA   9      HEALTH     2.352    0.63   +37.9%    +17.2%    1,009.8     797    ₹804,827      +2.3%     
  PFC         10     FIN SVC    2.296    0.88   +29.9%    +20.9%    94.9        8482   ₹805,036      +11.0%    
  TIINDIA     11     AUTO       2.254    0.88   +77.8%    +24.5%    2,806.1     286    ₹802,555      +5.2%     
  AXISBANK    12     PVT BNK    2.096    1.04   +36.8%    +20.3%    901.5       893    ₹805,005      +3.3%     
  FEDERALBNK  13     PVT BNK    1.940    1.17   +52.3%    +13.4%    130.1       6186   ₹805,083      -0.2%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         2      DEFENCE    01-Apr-22   723.8       1,323.9     896    ₹537,713      +82.9%      +4.1%     
  BEL         24     DEFENCE    01-Apr-22   68.5        100.1       9469   ₹298,899      +46.1%      -2.3%     
  VBL         7      FMCG       01-Aug-22   183.1       250.6       3946   ₹266,268      +36.8%      +9.5%     
  BAJAJHLDNG  26     FIN SVC    01-Oct-21   4,433.3     6,038.0     117    ₹187,754      +36.2%      -2.6%     
  NTPC        25     ENERGY     01-Apr-22   126.0       154.8       5149   ₹147,797      +22.8%      +1.2%     
  CUMMINSIND  8      INFRA      01-Aug-22   1,157.0     1,372.2     624    ₹134,282      +18.6%      +5.9%     
  TVSMOTOR    20     AUTO       01-Aug-22   911.5       1,032.8     792    ₹96,095       +13.3%      -2.1%     
  EICHERMOT   31     AUTO       01-Aug-22   2,962.8     3,319.6     243    ₹86,703       +12.0%      -1.4%     
  ITC         11     FMCG       01-Aug-22   254.0       280.5       2845   ₹75,313       +10.4%      -0.9%     
  M&M         29     AUTO       01-Aug-22   1,193.6     1,247.3     605    ₹32,496       +4.5%       +1.6%     

  AFTER: Invested ₹15,902,149 | Cash ₹191,028 | Total ₹16,093,177 | Positions 19/20 | Slot ₹805,089

========================================================================
  REBALANCE #49  —  01 Feb 2023
  NAV: ₹15,174,971  |  Slot: ₹758,749  |  Cash: ₹191,028
========================================================================

  [REGIME OFF] Nifty 200 9,172.1 < EMA200 9,287.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         14     DEFENCE    01-Apr-22   723.8       1,136.0     896    ₹369,361      +57.0%      -5.1%     
  BAJAJHLDNG  67     FIN SVC    01-Oct-21   4,433.3     5,815.5     117    ₹161,717      +31.2%      +3.8%     
  BEL         64     DEFENCE    01-Apr-22   68.5        87.4        9469   ₹178,755      +27.5%      -7.3%     
  VBL         7      FMCG       01-Aug-22   183.1       232.5       3946   ₹194,814      +27.0%      -4.7%     
  NTPC        41     ENERGY     01-Apr-22   126.0       152.8       5149   ₹137,617      +21.2%      +1.1%     
  CUMMINSIND  6      INFRA      01-Aug-22   1,157.0     1,361.3     624    ₹127,524      +17.7%      -0.0%     
  ITC         1      FMCG       01-Aug-22   254.0       298.5       2845   ₹126,658      +17.5%      +6.2%     
  TVSMOTOR    22     AUTO       01-Aug-22   911.5       1,001.7     792    ₹71,438       +9.9%       -0.2%     
  M&M         12     AUTO       01-Aug-22   1,193.6     1,303.8     605    ₹66,685       +9.2%       +2.7%     
  EICHERMOT   66     AUTO       01-Aug-22   2,962.8     3,189.9     243    ₹55,175       +7.7%       +2.7%     
  INDIANB     2      PSU BNK    01-Dec-22   250.0       265.1       3220   ₹48,691       +6.0%       +0.9%     
  FEDERALBNK  35     PVT BNK    01-Dec-22   130.1       128.8       6186   ₹-8,190       -1.0%       -2.4%     
  PFC         8      FIN SVC    01-Dec-22   94.9        93.4        8482   ₹-12,497      -1.6%       -6.0%     
  SUNPHARMA   38     HEALTH     01-Dec-22   1,009.8     979.4       797    ₹-24,265      -3.0%       -1.5%     
  BANKINDIA   9      PSU BNK    01-Dec-22   73.9        70.9        10893  ₹-32,938      -4.1%       -8.0%     
  AXISBANK    59     PVT BNK    01-Dec-22   901.5       855.0       893    ₹-41,504      -5.2%       -5.5%     
  TIINDIA     36     AUTO       01-Dec-22   2,806.1     2,605.8     286    ₹-57,299      -7.1%       -1.5%     
  UNIONBANK   3      PSU BNK    01-Dec-22   72.2        65.5        11154  ₹-74,576      -9.3%       -6.3%     
  AMBUJACEM   131    INFRA      01-Dec-22   570.9       328.3       1410   ₹-342,049     -42.5% ⚠    -28.6%    
  ⚠  WAZ < 0 (momentum below universe mean): AMBUJACEM

  AFTER: Invested ₹14,983,943 | Cash ₹191,028 | Total ₹15,174,971 | Positions 19/20 | Slot ₹758,749

========================================================================
  REBALANCE #50  —  03 Apr 2023
  NAV: ₹15,724,288  |  Slot: ₹786,214  |  Cash: ₹191,028
========================================================================
  [SECTOR CAP≤4] dropped: APOLLOTYRE

  [REGIME OFF] Nifty 200 9,030.7 < EMA200 9,208.7 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         5      DEFENCE    01-Apr-22   723.8       1,312.6     896    ₹527,605      +81.4%      +1.9%     
  VBL         3      FMCG       01-Aug-22   183.1       280.4       3946   ₹383,793      +53.1%      +5.0%     
  BEL         30     DEFENCE    01-Apr-22   68.5        94.1        9469   ₹241,523      +37.2%      +3.4%     
  CUMMINSIND  4      INFRA      01-Aug-22   1,157.0     1,543.4     624    ₹241,127      +33.4%      -1.1%     
  NTPC        11     ENERGY     01-Apr-22   126.0       164.0       5149   ₹195,207      +30.1%      +1.8%     
  BAJAJHLDNG  49     FIN SVC    01-Oct-21   4,433.3     5,564.3     117    ₹132,329      +25.5%      -3.3%     
  ITC         1      FMCG       01-Aug-22   254.0       318.1       2845   ₹182,371      +25.2%      -0.3%     
  TVSMOTOR    10     AUTO       01-Aug-22   911.5       1,072.3     792    ₹127,336      +17.6%      +2.1%     
  PFC         8      FIN SVC    01-Dec-22   94.9        108.8       8482   ₹118,095      +14.7%      +1.5%     
  INDIANB     9      PSU BNK    01-Dec-22   250.0       262.7       3220   ₹41,110       +5.1%       +4.0%     
  FEDERALBNK  41     PVT BNK    01-Dec-22   130.1       130.6       6186   ₹2,730        +0.3%       +2.8%     
  EICHERMOT   59     AUTO       01-Aug-22   2,962.8     2,901.3     243    ₹-14,944      -2.1% ⚠     +0.2%     
  AXISBANK    67     PVT BNK    01-Dec-22   901.5       862.3       893    ₹-34,958      -4.3% ⚠     +2.0%     
  M&M         28     AUTO       01-Aug-22   1,193.6     1,128.2     605    ₹-39,585      -5.5%       -1.4%     
  SUNPHARMA   57     HEALTH     01-Dec-22   1,009.8     951.9       797    ₹-46,184      -5.7%       +0.5%     
  BANKINDIA   45     PSU BNK    01-Dec-22   73.9        67.5        10893  ₹-69,266      -8.6%       +2.5%     
  TIINDIA     43     AUTO       01-Dec-22   2,806.1     2,553.0     286    ₹-72,388      -9.0%       -0.7%     
  UNIONBANK   36     PSU BNK    01-Dec-22   72.2        60.0        11154  ₹-135,817     -16.9%      +2.0%     
  AMBUJACEM   97     INFRA      01-Dec-22   570.9       368.3       1410   ₹-285,653     -35.5% ⚠    +1.3%     
  ⚠  WAZ < 0 (momentum below universe mean): EICHERMOT, AXISBANK, AMBUJACEM

  AFTER: Invested ₹15,533,260 | Cash ₹191,028 | Total ₹15,724,288 | Positions 19/20 | Slot ₹786,214

========================================================================
  REBALANCE #51  —  01 Jun 2023
  NAV: ₹17,352,944  |  Slot: ₹867,647  |  Cash: ₹191,028
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJ-AUTO

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  NTPC        95     ENERGY     01-Apr-22   126.0       160.8       5149   ₹179,063      +27.6%    426d  
  M&M         63     AUTO       01-Aug-22   1,193.6     1,271.9     605    ₹47,344       +6.6%     304d  
  AXISBANK    —      PVT BNK    01-Dec-22   901.5       917.3       893    ₹14,117       +1.8%     182d  
  INDIANB     65     PSU BNK    01-Dec-22   250.0       248.9       3220   ₹-3,353       -0.4%     182d  
  SUNPHARMA   101    HEALTH     01-Dec-22   1,009.8     960.3       797    ₹-39,481      -4.9%     182d  
  FEDERALBNK  80     PVT BNK    01-Dec-22   130.1       123.1       6186   ₹-43,682      -5.4%     182d  
  BANKINDIA   69     PSU BNK    01-Dec-22   73.9        66.3        10893  ₹-83,313      -10.3%    182d  
  UNIONBANK   30     PSU BNK    01-Dec-22   72.2        64.0        11154  ₹-90,874      -11.3%    182d  
  AMBUJACEM   —      INFRA      01-Dec-22   570.9       421.8       1410   ₹-210,204     -26.1%    182d  

  ENTRIES (8)
  [52w filter blocked 1: ADANIGREEN(-61.4%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RECLTD      2      FIN SVC    2.858    1.03   +77.0%    +24.4%    120.3       7210   ₹867,639      +5.1%     
  CHOLAFIN    4      FIN SVC    2.758    1.11   +63.8%    +36.8%    1,039.4     834    ₹866,827      +3.0%     
  SYNGENE     6      HEALTH     2.421    0.43   +37.7%    +28.3%    720.6       1203   ₹866,932      +3.7%     
  APOLLOTYRE  7      AUTO       2.374    1.12   +86.4%    +22.7%    375.8       2308   ₹867,338      +4.2%     
  TORNTPHARM  8      HEALTH     2.192    0.39   +24.5%    +19.9%    1,716.8     505    ₹866,999      +4.8%     
  NESTLEIND   11     FMCG       2.105    0.49   +25.3%    +17.7%    1,062.6     816    ₹867,045      +1.6%     
  MAXHEALTH   16     HEALTH     1.793    0.25   +45.8%    +23.5%    530.5       1635   ₹867,424      +2.0%     
  TRENT       17     CONSUMP    1.776    1.09   +49.1%    +19.8%    1,558.5     556    ₹866,516      +4.3%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         13     DEFENCE    01-Apr-22   723.8       1,490.9     896    ₹687,327      +106.0%     +1.9%     
  VBL         4      FMCG       01-Aug-22   183.1       335.0       3946   ₹599,466      +83.0%      +5.7%     
  BEL         18     DEFENCE    01-Apr-22   68.5        109.9       9469   ₹391,797      +60.4%      +3.8%     
  BAJAJHLDNG  41     FIN SVC    01-Oct-21   4,433.3     6,686.4     117    ₹263,619      +50.8%      +2.4%     
  ITC         1      FMCG       01-Aug-22   254.0       377.4       2845   ₹351,173      +48.6%      +3.5%     
  CUMMINSIND  16     INFRA      01-Aug-22   1,157.0     1,683.2     624    ₹328,390      +45.5%      +4.5%     
  TVSMOTOR    10     AUTO       01-Aug-22   911.5       1,257.5     792    ₹274,010      +38.0%      +1.9%     
  PFC         5      FIN SVC    01-Dec-22   94.9        128.6       8482   ₹285,693      +35.5%      +7.0%     
  EICHERMOT   35     AUTO       01-Aug-22   2,962.8     3,588.7     243    ₹152,082      +21.1%      +4.1%     
  TIINDIA     33     AUTO       01-Dec-22   2,806.1     2,875.8     286    ₹19,915       +2.5%       +5.1%     

  AFTER: Invested ₹17,323,038 | Cash ₹21,669 | Total ₹17,344,707 | Positions 18/20 | Slot ₹867,647

========================================================================
  REBALANCE #52  —  01 Aug 2023
  NAV: ₹19,484,517  |  Slot: ₹974,226  |  Cash: ₹21,669
========================================================================
  [SECTOR CAP≤4] dropped: SUNPHARMA

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VBL         49     FMCG       01-Aug-22   183.1       317.7       3946   ₹530,897      +73.5%    365d  
  BAJAJHLDNG  61     FIN SVC    01-Oct-21   4,433.3     7,119.1     117    ₹314,245      +60.6%    669d  
  EICHERMOT   106    AUTO       01-Aug-22   2,962.8     3,298.8     243    ₹81,646       +11.3%    365d  
  TIINDIA     52     AUTO       01-Dec-22   2,806.1     3,050.7     286    ₹69,939       +8.7%     243d  
  NESTLEIND   85     FMCG       01-Jun-23   1,062.6     1,097.5     816    ₹28,515       +3.3%     61d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  POLYCAB     3      MFG        3.148    0.82   +109.5%   +41.5%    4,561.7     213    ₹971,640      +7.3%     
  TATACOMM    4      CONSUMP    2.528    0.90   +69.3%    +44.3%    1,705.7     571    ₹973,939      +6.9%     
  NTPC        5      ENERGY     2.500    0.70   +56.9%    +27.7%    207.6       4692   ₹974,168      +12.7%    
  COLPAL      7      FMCG       2.167    0.36   +31.3%    +28.3%    1,880.1     518    ₹973,875      +7.0%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         8      DEFENCE    01-Apr-22   723.8       1,870.9     896    ₹1,027,779    +158.5%     +0.8%     
  PFC         1      FIN SVC    01-Dec-22   94.9        185.9       8482   ₹771,406      +95.8%      +10.0%    
  BEL         23     DEFENCE    01-Apr-22   68.5        127.0       9469   ₹553,594      +85.3%      +3.1%     
  CUMMINSIND  12     INFRA      01-Aug-22   1,157.0     1,865.6     624    ₹442,181      +61.2%      +1.2%     
  ITC         15     FMCG       01-Aug-22   254.0       399.0       2845   ₹412,468      +57.1%      -0.7%     
  TVSMOTOR    24     AUTO       01-Aug-22   911.5       1,353.9     792    ₹350,420      +48.5%      +1.7%     
  RECLTD      2      FIN SVC    01-Jun-23   120.3       175.8       7210   ₹399,912      +46.1%      +14.9%    
  TORNTPHARM  14     HEALTH     01-Jun-23   1,716.8     1,927.0     505    ₹106,131      +12.2%      +2.0%     
  SYNGENE     25     HEALTH     01-Jun-23   720.6       800.9       1203   ₹96,566       +11.1%      +3.4%     
  APOLLOTYRE  10     AUTO       01-Jun-23   375.8       415.0       2308   ₹90,565       +10.4%      +2.1%     
  TRENT       33     CONSUMP    01-Jun-23   1,558.5     1,703.3     556    ₹80,529       +9.3%       -0.1%     
  CHOLAFIN    13     FIN SVC    01-Jun-23   1,039.4     1,126.2     834    ₹72,443       +8.4%       -0.7%     
  MAXHEALTH   35     HEALTH     01-Jun-23   530.5       569.7       1635   ₹64,046       +7.4%       -4.7%     

  AFTER: Invested ₹18,700,376 | Cash ₹779,518 | Total ₹19,479,894 | Positions 17/20 | Slot ₹974,226

========================================================================
  REBALANCE #53  —  03 Oct 2023
  NAV: ₹20,913,958  |  Slot: ₹1,045,698  |  Cash: ₹779,518
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PFC         4      FIN SVC    01-Dec-22   94.9        225.2       8482   ₹1,105,376    +137.3%   306d  
  ITC         75     FMCG       01-Aug-22   254.0       377.5       2845   ₹351,295      +48.6%    428d  
  CUMMINSIND  91     INFRA      01-Aug-22   1,157.0     1,624.9     624    ₹292,019      +40.4%    428d  
  MAXHEALTH   79     HEALTH     01-Jun-23   530.5       590.5       1635   ₹98,036       +11.3%    124d  
  TORNTPHARM  76     HEALTH     01-Jun-23   1,716.8     1,826.7     505    ₹55,475       +6.4%     124d  
  APOLLOTYRE  85     AUTO       01-Jun-23   375.8       361.3       2308   ₹-33,531      -3.9%     124d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RVNL        3      PSE        3.481    1.14   +426.1%   +41.9%    170.4       6136   ₹1,045,591    +6.7%     
  LT          4      INFRA      2.898    0.83   +67.9%    +26.7%    2,991.9     349    ₹1,044,167    +5.9%     
  INDIANB     5      PSU BNK    2.753    1.10   +144.4%   +48.2%    412.0       2538   ₹1,045,666    +7.3%     
  COALINDIA   7      ENERGY     2.353    0.77   +50.3%    +28.2%    242.6       4310   ₹1,045,579    +5.4%     
  SUNTV       8      MEDIA      2.338    0.76   +31.4%    +43.0%    586.4       1783   ₹1,045,590    +4.2%     
  OIL         9      OIL&GAS    2.070    0.42   +78.2%    +22.4%    180.7       5786   ₹1,045,664    +3.8%     
  PGHH        10     FMCG       1.897    0.31   +23.2%    +24.1%    16,905.8    61     ₹1,031,253    +2.8%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         48     DEFENCE    01-Apr-22   723.8       1,901.3     896    ₹1,055,018    +162.7%     +0.3%     
  RECLTD      1      FIN SVC    01-Jun-23   120.3       260.0       7210   ₹1,006,788    +116.0%     +11.0%    
  BEL         46     DEFENCE    01-Apr-22   68.5        136.2       9469   ₹640,154      +98.6%      +1.8%     
  TVSMOTOR    25     AUTO       01-Aug-22   911.5       1,512.4     792    ₹475,917      +65.9%      +2.7%     
  TRENT       24     CONSUMP    01-Jun-23   1,558.5     2,053.9     556    ₹275,473      +31.8%      -0.3%     
  CHOLAFIN    30     FIN SVC    01-Jun-23   1,039.4     1,249.1     834    ₹174,932      +20.2%      +6.0%     
  POLYCAB     5      MFG        01-Aug-23   4,561.7     5,293.7     213    ₹155,928      +16.0%      +3.5%     
  SYNGENE     50     HEALTH     01-Jun-23   720.6       805.4       1203   ₹101,944      +11.8%      +1.7%     
  NTPC        10     ENERGY     01-Aug-23   207.6       225.5       4692   ₹84,047       +8.6%       +2.0%     
  TATACOMM    21     CONSUMP    01-Aug-23   1,705.7     1,840.2     571    ₹76,799       +7.9%       +1.5%     
  COLPAL      22     FMCG       01-Aug-23   1,880.1     1,853.9     518    ₹-13,567      -1.4%       -0.8%     

  AFTER: Invested ₹20,717,960 | Cash ₹187,326 | Total ₹20,905,286 | Positions 18/20 | Slot ₹1,045,698

========================================================================
  REBALANCE #54  —  01 Dec 2023
  NAV: ₹22,934,970  |  Slot: ₹1,146,748  |  Cash: ₹187,326
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  RECLTD      2      FIN SVC    01-Jun-23   120.3       336.8       7210   ₹1,560,912    +179.9%   183d  
  SYNGENE     88     HEALTH     01-Jun-23   720.6       741.8       1203   ₹25,512       +2.9%     183d  
  RVNL        25     PSE        03-Oct-23   170.4       163.0       6136   ₹-45,460      -4.3%     59d   
  TATACOMM    96     CONSUMP    01-Aug-23   1,705.7     1,606.1     571    ₹-56,853      -5.8%     122d  
  INDIANB     73     PSU BNK    03-Oct-23   412.0       374.5       2538   ₹-95,136      -9.1%     59d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BAJAJ-AUTO  4      AUTO       3.054    0.63   +72.1%    +29.6%    5,767.9     198    ₹1,142,044    +5.8%     
  TORNTPOWER  6      ENERGY     2.487    0.93   +86.0%    +43.1%    914.4       1254   ₹1,146,644    +14.4%    
  HEROMOTOCO  10     AUTO       2.157    0.89   +46.0%    +25.8%    3,443.0     333    ₹1,146,509    +9.6%     
  PERSISTENT  11     IT         2.104    1.12   +67.2%    +26.0%    3,167.3     362    ₹1,146,546    +1.8%     
  BOSCHLTD    12     AUTO       2.095    0.67   +36.2%    +18.5%    21,526.2    53     ₹1,140,888    +6.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         13     DEFENCE    01-Apr-22   723.8       2,392.1     896    ₹1,494,778    +230.5%     +13.8%    
  BEL         52     DEFENCE    01-Apr-22   68.5        144.2       9469   ₹716,562      +110.4%     +4.4%     
  TVSMOTOR    4      AUTO       01-Aug-22   911.5       1,887.8     792    ₹773,257      +107.1%     +9.7%     
  TRENT       5      CONSUMP    01-Jun-23   1,558.5     2,800.7     556    ₹690,650      +79.7%      +10.1%    
  COALINDIA   3      ENERGY     03-Oct-23   242.6       301.3       4310   ₹253,063      +24.2%      +6.1%     
  NTPC        8      ENERGY     01-Aug-23   207.6       253.9       4692   ₹217,189      +22.3%      +7.3%     
  COLPAL      16     FMCG       01-Aug-23   1,880.1     2,158.3     518    ₹144,150      +14.8%      +5.5%     
  POLYCAB     23     MFG        01-Aug-23   4,561.7     5,157.1     213    ₹126,822      +13.1%      +0.5%     
  SUNTV       32     MEDIA      03-Oct-23   586.4       639.8       1783   ₹95,105       +9.1%       +2.0%     
  CHOLAFIN    61     FIN SVC    01-Jun-23   1,039.4     1,124.0     834    ₹70,573       +8.1%       -0.4%     
  OIL         28     OIL&GAS    03-Oct-23   180.7       192.7       5786   ₹69,052       +6.6%       +1.5%     
  LT          12     INFRA      03-Oct-23   2,991.9     3,106.2     349    ₹39,888       +3.8%       +4.3%     
  PGHH        49     FMCG       03-Oct-23   16,905.8    16,628.0    61     ₹-16,946      -1.6%       -1.2%     

  AFTER: Invested ₹22,281,534 | Cash ₹646,640 | Total ₹22,928,175 | Positions 18/20 | Slot ₹1,146,748

========================================================================
  REBALANCE #55  —  01 Feb 2024
  NAV: ₹25,905,004  |  Slot: ₹1,295,250  |  Cash: ₹646,640
========================================================================
  [SECTOR CAP≤4] dropped: TATAPOWER, APOLLOTYRE

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LT          51     INFRA      03-Oct-23   2,991.9     3,308.0     349    ₹110,337      +10.6%    121d  
  CHOLAFIN    80     FIN SVC    01-Jun-23   1,039.4     1,140.8     834    ₹84,574       +9.8%     245d  
  SUNTV       85     MEDIA      03-Oct-23   586.4       621.7       1783   ₹62,937       +6.0%     121d  
  PGHH        90     FMCG       03-Oct-23   16,905.8    16,275.4    61     ₹-38,453      -3.7%     121d  
  POLYCAB     106    MFG        01-Aug-23   4,561.7     4,200.9     213    ₹-76,857      -7.9%     184d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IRFC        1      FIN SVC    4.124    1.11   +423.4%   +136.1%   164.1       7891   ₹1,295,134    +18.3%    
  NHPC        5      ENERGY     2.833    0.83   +124.2%   +79.2%    85.8        15097  ₹1,295,175    +17.7%    
  RVNL        6      PSE        2.644    1.17   +297.9%   +90.3%    293.9       4406   ₹1,295,066    +16.8%    
  MRF         11     MFG        2.093    0.44   +59.4%    +30.7%    142,044.1   9      ₹1,278,397    +4.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         6      DEFENCE    01-Apr-22   723.8       2,912.1     896    ₹1,960,738    +302.3%     +2.0%     
  BEL         22     DEFENCE    01-Apr-22   68.5        179.4       9469   ₹1,049,515    +161.7%     -0.8%     
  TVSMOTOR    16     AUTO       01-Aug-22   911.5       1,972.3     792    ₹840,155      +116.4%     +0.1%     
  TRENT       4      CONSUMP    01-Jun-23   1,558.5     3,095.0     556    ₹854,314      +98.6%      -0.6%     
  OIL         21     OIL&GAS    03-Oct-23   180.7       271.3       5786   ₹523,895      +50.1%      +9.7%     
  NTPC        11     ENERGY     01-Aug-23   207.6       304.0       4692   ₹452,182      +46.4%      +3.3%     
  COALINDIA   15     ENERGY     03-Oct-23   242.6       353.5       4310   ₹478,026      +45.7%      +4.9%     
  PERSISTENT  20     IT         01-Dec-23   3,167.3     4,099.4     362    ₹337,425      +29.4%      +5.1%     
  BAJAJ-AUTO  5      AUTO       01-Dec-23   5,767.9     7,303.5     198    ₹304,058      +26.6%      +5.9%     
  COLPAL      24     FMCG       01-Aug-23   1,880.1     2,370.6     518    ₹254,100      +26.1%      +0.8%     
  HEROMOTOCO  13     AUTO       01-Dec-23   3,443.0     4,200.0     333    ₹252,080      +22.0%      +5.2%     
  TORNTPOWER  17     ENERGY     01-Dec-23   914.4       1,009.7     1254   ₹119,486      +10.4%      +5.3%     
  BOSCHLTD    38     AUTO       01-Dec-23   21,526.2    23,082.1    53     ₹82,461       +7.2%       +3.1%     

  AFTER: Invested ₹25,320,120 | Cash ₹578,752 | Total ₹25,898,872 | Positions 17/20 | Slot ₹1,295,250

========================================================================
  REBALANCE #56  —  01 Apr 2024
  NAV: ₹28,731,938  |  Slot: ₹1,436,597  |  Cash: ₹578,752
========================================================================
  [SECTOR CAP≤4] dropped: MARUTI

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BEL         24     DEFENCE    01-Apr-22   68.5        208.0       9469   ₹1,320,498    +203.4%   731d  
  COALINDIA   25     ENERGY     03-Oct-23   242.6       388.7       4310   ₹629,533      +60.2%    181d  
  NTPC        34     ENERGY     01-Aug-23   207.6       326.4       4692   ₹557,133      +57.2%    244d  
  NHPC        28     ENERGY     01-Feb-24   85.8        86.2        15097  ₹6,922        +0.5%     60d   
  RVNL        13     PSE        01-Feb-24   293.9       258.6       4406   ₹-155,817     -12.0%    60d   
  IRFC        2      FIN SVC    01-Feb-24   164.1       139.9       7891   ₹-191,335     -14.8%    60d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SUNPHARMA   2      HEALTH     2.815    0.47   +71.1%    +30.8%    1,598.7     898    ₹1,435,623    +3.3%     
  CUMMINSIND  5      INFRA      2.732    0.77   +84.2%    +52.5%    2,927.3     490    ₹1,434,393    +6.4%     
  KALYANKJIL  9      CON DUR    2.226    0.53   +260.5%   +20.9%    423.7       3390   ₹1,436,369    +8.3%     
  DIXON       10     CON DUR    2.118    0.79   +160.8%   +17.5%    7,586.0     189    ₹1,433,747    +7.3%     
  UNIONBANK   11     PSU BNK    2.116    1.16   +155.2%   +31.7%    143.5       10012  ₹1,436,588    +4.9%     
  SIEMENS     12     ENERGY     2.115    0.74   +66.6%    +37.9%    3,195.5     449    ₹1,434,763    +11.4%    

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         14     DEFENCE    01-Apr-22   723.8       3,330.6     896    ₹2,335,700    +360.2%     +6.8%     
  TRENT       7      CONSUMP    01-Jun-23   1,558.5     3,877.1     556    ₹1,289,125    +148.8%     -0.9%     
  TVSMOTOR    37     AUTO       01-Aug-22   911.5       2,122.8     792    ₹959,394      +132.9%     +1.1%     
  OIL         10     OIL&GAS    03-Oct-23   180.7       373.1       5786   ₹1,113,198    +106.5%     +2.3%     
  TORNTPOWER  6      ENERGY     01-Dec-23   914.4       1,379.4     1254   ₹583,064      +50.8%      +13.6%    
  BAJAJ-AUTO  1      AUTO       01-Dec-23   5,767.9     8,626.2     198    ₹565,936      +49.6%      +4.3%     
  BOSCHLTD    5      AUTO       01-Dec-23   21,526.2    29,732.3    53     ₹434,924      +38.1%      +2.7%     
  COLPAL      36     FMCG       01-Aug-23   1,880.1     2,572.1     518    ₹358,484      +36.8%      +2.5%     
  HEROMOTOCO  27     AUTO       01-Dec-23   3,443.0     4,380.0     333    ₹312,034      +27.2%      +1.7%     
  PERSISTENT  55     IT         01-Dec-23   3,167.3     3,949.9     362    ₹283,321      +24.7%      -2.3%     
  MRF         42     MFG        01-Feb-24   142,044.1   135,309.0   9      ₹-60,616      -4.7%       -1.2%     

  AFTER: Invested ₹28,043,559 | Cash ₹678,153 | Total ₹28,721,712 | Positions 17/20 | Slot ₹1,436,597

========================================================================
  REBALANCE #57  —  03 Jun 2024
  NAV: ₹32,285,226  |  Slot: ₹1,614,261  |  Cash: ₹678,153
========================================================================
  [SECTOR CAP≤4] dropped: ASHOKLEY

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HAL         1      DEFENCE    01-Apr-22   723.8       5,160.9     896    ₹3,975,673    +613.0%   794d  
  OIL         —      OIL&GAS    03-Oct-23   180.7       422.6       5786   ₹1,399,403    +133.8%   244d  
  TORNTPOWER  —      ENERGY     01-Dec-23   914.4       1,469.3     1254   ₹695,845      +60.7%    185d  
  UNIONBANK   43     PSU BNK    01-Apr-24   143.5       155.6       10012  ₹121,434      +8.5%     63d   
  PERSISTENT  138    IT         01-Dec-23   3,167.3     3,385.8     362    ₹79,108       +6.9%     185d  
  SUNPHARMA   98     HEALTH     01-Apr-24   1,598.7     1,425.8     898    ₹-155,260     -10.8%    63d   
  MRF         137    MFG        01-Feb-24   142,044.1   126,586.5   9      ₹-139,119     -10.9%    123d  

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BSE         3      FIN SVC    2.845    0.70   +415.1%   +18.8%    894.9       1803   ₹1,613,438    -0.2%     
  MAZDOCK     4      DEFENCE    2.815    1.11   +327.5%   +55.7%    1,602.9     1007   ₹1,614,117    +14.1%    
  BDL         6      DEFENCE    2.702    1.16   +198.4%   +70.1%    1,583.6     1019   ₹1,613,658    +22.1%    
  PRESTIGE    8      REALTY     2.584    0.94   +250.6%   +41.9%    1,731.5     932    ₹1,613,800    +12.4%    
  CGPOWER     9      ENERGY     2.375    0.47   +92.6%    +60.2%    683.4       2362   ₹1,614,141    +10.2%    
  ESCORTS     11     MFG        2.253    0.74   +89.4%    +34.6%    3,820.6     422    ₹1,612,278    +5.0%     
  BHARATFORG  12     DEFENCE    2.192    1.11   +111.0%   +36.8%    1,589.1     1015   ₹1,612,928    +8.8%     
  HAVELLS     13     CON DUR    2.041    0.57   +51.7%    +32.4%    1,852.4     871    ₹1,613,461    +4.5%     
  HINDALCO    14     METAL      2.032    1.06   +70.7%    +37.8%    686.9       2349   ₹1,613,635    +4.2%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       11     CONSUMP    01-Jun-23   1,558.5     4,652.6     556    ₹1,720,349    +198.5%     +2.2%     
  TVSMOTOR    58     AUTO       01-Aug-22   911.5       2,232.9     792    ₹1,046,560    +145.0%     +3.9%     
  BAJAJ-AUTO  31     AUTO       01-Dec-23   5,767.9     8,906.0     198    ₹621,338      +54.4%      +3.9%     
  HEROMOTOCO  38     AUTO       01-Dec-23   3,443.0     4,829.0     333    ₹461,558      +40.3%      +3.5%     
  COLPAL      55     FMCG       01-Aug-23   1,880.1     2,578.9     518    ₹361,970      +37.2%      -0.3%     
  BOSCHLTD    60     AUTO       01-Dec-23   21,526.2    29,444.7    53     ₹419,679      +36.8%      -2.0%     
  SIEMENS     3      ENERGY     01-Apr-24   3,195.5     4,254.8     449    ₹475,663      +33.2%      +6.5%     
  DIXON       6      CON DUR    01-Apr-24   7,586.0     9,878.1     189    ₹433,209      +30.2%      +10.6%    
  CUMMINSIND  13     INFRA      01-Apr-24   2,927.3     3,618.2     490    ₹338,540      +23.6%      +2.9%     
  KALYANKJIL  32     CON DUR    01-Apr-24   423.7       388.9       3390   ₹-117,871     -8.2%       -2.3%     

  AFTER: Invested ₹32,013,452 | Cash ₹254,532 | Total ₹32,267,983 | Positions 19/20 | Slot ₹1,614,261

========================================================================
  REBALANCE #58  —  01 Aug 2024
  NAV: ₹35,810,434  |  Slot: ₹1,790,522  |  Cash: ₹254,532
========================================================================
  [SECTOR CAP≤4] dropped: MOTHERSON, M&M

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MAZDOCK     3      DEFENCE    03-Jun-24   1,602.9     2,513.2     1007   ₹916,686      +56.8%    59d   
  DIXON       6      CON DUR    01-Apr-24   7,586.0     11,661.2    189    ₹770,218      +53.7%    122d  
  SIEMENS     40     ENERGY     01-Apr-24   3,195.5     4,111.4     449    ₹411,264      +28.7%    122d  
  BHARATFORG  21     DEFENCE    03-Jun-24   1,589.1     1,703.7     1015   ₹116,347      +7.2%     59d   
  CGPOWER     50     ENERGY     03-Jun-24   683.4       725.6       2362   ₹99,612       +6.2%     59d   
  PRESTIGE    22     REALTY     03-Jun-24   1,731.5     1,751.7     932    ₹18,740       +1.2%     59d   
  HAVELLS     87     CON DUR    03-Jun-24   1,852.4     1,811.8     871    ₹-35,415      -2.2%     59d   
  HINDALCO    105    METAL      03-Jun-24   686.9       664.8       2349   ₹-52,053      -3.2%     59d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  OFSS        4      IT         2.603    0.95   +186.5%   +48.1%    10,120.6    176    ₹1,781,233    +1.7%     
  ZYDUSLIFE   5      HEALTH     2.445    0.63   +103.4%   +30.5%    1,227.3     1458   ₹1,789,462    +5.2%     
  PERSISTENT  6      IT         2.316    0.84   +91.3%    +42.7%    4,750.7     376    ₹1,786,269    +2.8%     
  LUPIN       8      HEALTH     2.075    0.39   +107.4%   +19.2%    1,941.3     922    ₹1,789,857    +7.9%     
  SUNTV       10     MEDIA      1.987    0.48   +78.1%    +35.8%    852.7       2099   ₹1,789,818    +8.2%     
  INFY        14     IT         1.872    0.77   +32.1%    +33.0%    1,741.4     1028   ₹1,790,169    +4.6%     
  TORNTPHARM  15     HEALTH     1.781    0.34   +67.8%    +21.5%    3,146.3     569    ₹1,790,237    +5.1%     
  SUNPHARMA   18     HEALTH     1.719    0.50   +58.2%    +14.5%    1,688.4     1060   ₹1,789,665    +5.2%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       2      CONSUMP    01-Jun-23   1,558.5     5,759.0     556    ₹2,335,466    +269.5%     +5.3%     
  TVSMOTOR    11     AUTO       01-Aug-22   911.5       2,564.6     792    ₹1,309,235    +181.4%     +5.0%     
  COLPAL      15     FMCG       01-Aug-23   1,880.1     3,238.2     518    ₹703,518      +72.2%      +7.2%     
  BAJAJ-AUTO  25     AUTO       01-Dec-23   5,767.9     9,358.3     198    ₹710,897      +62.2%      +2.2%     
  BOSCHLTD    19     AUTO       01-Dec-23   21,526.2    33,910.4    53     ₹656,366      +57.5%      -0.3%     
  HEROMOTOCO  30     AUTO       01-Dec-23   3,443.0     5,063.6     333    ₹539,673      +47.1%      -1.2%     
  KALYANKJIL  7      CON DUR    01-Apr-24   423.7       561.8       3390   ₹468,110      +32.6%      +5.2%     
  CUMMINSIND  43     INFRA      01-Apr-24   2,927.3     3,738.1     490    ₹397,260      +27.7%      +0.9%     
  ESCORTS     26     MFG        03-Jun-24   3,820.6     4,093.5     422    ₹115,180      +7.1%       +1.4%     
  BSE         48     FIN SVC    03-Jun-24   894.9       878.3       1803   ₹-29,839      -1.8%       +8.3%     
  BDL         18     DEFENCE    03-Jun-24   1,583.6     1,438.5     1019   ₹-147,822     -9.2%       -3.8%     

  AFTER: Invested ₹35,066,621 | Cash ₹726,825 | Total ₹35,793,446 | Positions 19/20 | Slot ₹1,790,522

========================================================================
  REBALANCE #59  —  01 Oct 2024
  NAV: ₹39,705,039  |  Slot: ₹1,985,252  |  Cash: ₹726,825
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  CUMMINSIND  38     INFRA      01-Apr-24   2,927.3     3,797.7     490    ₹426,475      +29.7%    183d  
  ESCORTS     99     MFG        03-Jun-24   3,820.6     4,151.9     422    ₹139,822      +8.7%     120d  
  SUNTV       —      OTHER      01-Aug-24   852.7       819.0       2099   ₹-70,780      -4.0%     61d   
  ZYDUSLIFE   62     HEALTH     01-Aug-24   1,227.3     1,068.1     1458   ₹-232,152     -13.0%    61d   
  BDL         127    DEFENCE    03-Jun-24   1,583.6     1,118.9     1019   ₹-473,501     -29.3%    120d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  VOLTAS      8      CON DUR    2.162    0.85   +113.5%   +27.9%    1,838.5     1079   ₹1,983,720    +0.4%     
  UNITDSPR    11     FMCG       1.936    0.54   +56.3%    +26.8%    1,589.7     1248   ₹1,983,922    +3.2%     
  ALKEM       12     HEALTH     1.839    0.57   +68.2%    +25.1%    6,044.3     328    ₹1,982,523    +0.5%     
  HCLTECH     15     IT         1.750    0.70   +45.9%    +23.6%    1,693.6     1172   ₹1,984,944    +2.4%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       1      CONSUMP    01-Jun-23   1,558.5     7,597.1     556    ₹3,357,462    +387.5%     +2.9%     
  TVSMOTOR    11     AUTO       01-Aug-22   911.5       2,817.1     792    ₹1,509,228    +209.1%     +0.7%     
  BAJAJ-AUTO  3      AUTO       01-Dec-23   5,767.9     11,692.4    198    ₹1,173,051    +102.7%     +2.8%     
  COLPAL      4      FMCG       01-Aug-23   1,880.1     3,666.2     518    ₹925,210      +95.0%      +3.9%     
  KALYANKJIL  2      CON DUR    01-Apr-24   423.7       748.0       3390   ₹1,099,220    +76.5%      +6.8%     
  BOSCHLTD    15     AUTO       01-Dec-23   21,526.2    37,334.5    53     ₹837,839      +73.4%      +6.5%     
  HEROMOTOCO  29     AUTO       01-Dec-23   3,443.0     5,420.2     333    ₹658,403      +57.4%      -1.5%     
  BSE         7      FIN SVC    03-Jun-24   894.9       1,282.3     1803   ₹698,470      +43.3%      +11.1%    
  PERSISTENT  16     IT         01-Aug-24   4,750.7     5,435.4     376    ₹257,450      +14.4%      +3.4%     
  LUPIN       6      HEALTH     01-Aug-24   1,941.3     2,180.8     922    ₹220,874      +12.3%      -0.1%     
  SUNPHARMA   5      HEALTH     01-Aug-24   1,688.4     1,889.9     1060   ₹213,639      +11.9%      +2.9%     
  TORNTPHARM  10     HEALTH     01-Aug-24   3,146.3     3,308.2     569    ₹92,109       +5.1%       -1.3%     
  OFSS        20     IT         01-Aug-24   10,120.6    10,613.9    176    ₹86,810       +4.9%       +0.6%     
  INFY        37     IT         01-Aug-24   1,741.4     1,790.1     1028   ₹50,006       +2.8%       +0.1%     

  AFTER: Invested ₹38,883,848 | Cash ₹811,769 | Total ₹39,695,617 | Positions 18/20 | Slot ₹1,985,252

========================================================================
  REBALANCE #60  —  02 Dec 2024
  NAV: ₹37,543,514  |  Slot: ₹1,877,176  |  Cash: ₹811,769
========================================================================
  [SECTOR CAP≤4] dropped: COFORGE, WIPRO

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TVSMOTOR    79     AUTO       01-Aug-22   911.5       2,474.5     792    ₹1,237,868    +171.5%   854d  
  BAJAJ-AUTO  74     AUTO       01-Dec-23   5,767.9     8,781.1     198    ₹596,612      +52.2%    367d  
  COLPAL      134    FMCG       01-Aug-23   1,880.1     2,792.9     518    ₹472,868      +48.6%    489d  
  HEROMOTOCO  89     AUTO       01-Dec-23   3,443.0     4,476.0     333    ₹343,993      +30.0%    367d  
  ALKEM       95     HEALTH     01-Oct-24   6,044.3     5,593.8     328    ₹-147,773     -7.5%     62d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  INDHOTEL    3      CONSUMP    2.912    1.13   +90.9%    +23.7%    795.2       2360   ₹1,876,578    +6.1%     
  CGPOWER     10     ENERGY     2.015    0.82   +93.6%    +8.5%     752.0       2496   ₹1,877,093    +2.9%     
  ICICIBANK   12     PVT BNK    1.958    0.98   +42.1%    +6.1%     1,294.7     1449   ₹1,875,963    +1.8%     
  OBEROIRLTY  13     REALTY     1.931    1.19   +48.0%    +16.9%    2,054.7     913    ₹1,875,930    +4.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       8      CONSUMP    01-Jun-23   1,558.5     6,791.3     556    ₹2,909,468    +335.8%     +0.3%     
  KALYANKJIL  6      CON DUR    01-Apr-24   423.7       720.0       3390   ₹1,004,408    +69.9%      +3.3%     
  BSE         2      FIN SVC    03-Jun-24   894.9       1,516.5     1803   ₹1,120,813    +69.5%      +0.2%     
  BOSCHLTD    10     AUTO       01-Dec-23   21,526.2    34,460.4    53     ₹685,515      +60.1%      +0.1%     
  PERSISTENT  9      IT         01-Aug-24   4,750.7     5,875.8     376    ₹423,045      +23.7%      +3.1%     
  OFSS        3      IT         01-Aug-24   10,120.6    11,378.1    176    ₹221,305      +12.4%      +5.8%     
  LUPIN       35     HEALTH     01-Aug-24   1,941.3     2,056.8     922    ₹106,474      +5.9%       -0.3%     
  SUNPHARMA   19     HEALTH     01-Aug-24   1,688.4     1,780.3     1060   ₹97,403       +5.4%       +0.8%     
  TORNTPHARM  31     HEALTH     01-Aug-24   3,146.3     3,278.0     569    ₹74,933       +4.2%       +3.5%     
  HCLTECH     15     IT         01-Oct-24   1,693.6     1,756.4     1172   ₹73,499       +3.7%       +1.0%     
  INFY        55     IT         01-Aug-24   1,741.4     1,787.1     1028   ₹46,966       +2.6%       +1.0%     
  UNITDSPR    22     FMCG       01-Oct-24   1,589.7     1,512.0     1248   ₹-96,982      -4.9%       +2.7%     
  VOLTAS      12     CON DUR    01-Oct-24   1,838.5     1,706.2     1079   ₹-142,745     -7.2%       +1.4%     

  AFTER: Invested ₹35,766,889 | Cash ₹1,767,713 | Total ₹37,534,602 | Positions 17/20 | Slot ₹1,877,176

========================================================================
  REBALANCE #61  —  01 Feb 2025
  NAV: ₹34,654,953  |  Slot: ₹1,732,748  |  Cash: ₹1,767,713
========================================================================
  [SECTOR CAP≤4] dropped: BAJFINANCE

  [REGIME OFF] Nifty 200 13,064.5 < EMA200 13,264.1 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       20     CONSUMP    01-Jun-23   1,558.5     6,176.8     556    ₹2,567,785    +296.3%     +2.9%     
  BSE         3      FIN SVC    03-Jun-24   894.9       1,795.7     1803   ₹1,624,185    +100.7%     -1.5%     
  BOSCHLTD    105    AUTO       01-Dec-23   21,526.2    28,305.1    53     ₹359,281      +31.5% ⚠    -6.3%     
  PERSISTENT  15     IT         01-Aug-24   4,750.7     5,896.4     376    ₹430,794      +24.1%      -2.5%     
  KALYANKJIL  87     CON DUR    01-Apr-24   423.7       504.0       3390   ₹272,107      +18.9% ⚠    -5.2%     
  LUPIN       30     HEALTH     01-Aug-24   1,941.3     2,043.5     922    ₹94,287       +5.3%       -3.1%     
  SUNPHARMA   39     HEALTH     01-Aug-24   1,688.4     1,714.9     1060   ₹28,172       +1.6%       -1.9%     
  INFY        25     IT         01-Aug-24   1,741.4     1,760.0     1028   ₹19,161       +1.1%       -1.3%     
  TORNTPHARM  21     HEALTH     01-Aug-24   3,146.3     3,173.5     569    ₹15,488       +0.9%       -1.6%     
  INDHOTEL    5      CONSUMP    02-Dec-24   795.2       795.6       2360   ₹937          +0.0%       +1.3%     
  ICICIBANK   33     PVT BNK    02-Dec-24   1,294.7     1,245.9     1449   ₹-70,601      -3.8%       +0.8%     
  HCLTECH     50     IT         01-Oct-24   1,693.6     1,605.9     1172   ₹-102,829     -5.2%       -5.1%     
  UNITDSPR    17     FMCG       01-Oct-24   1,589.7     1,478.3     1248   ₹-138,977     -7.0%       +1.9%     
  OBEROIRLTY  36     REALTY     02-Dec-24   2,054.7     1,832.9     913    ₹-202,465     -10.8%      -2.8%     
  OFSS        76     IT         01-Aug-24   10,120.6    8,249.1     176    ₹-329,387     -18.5% ⚠    -11.8%    
  CGPOWER     74     ENERGY     02-Dec-24   752.0       609.4       2496   ₹-355,932     -19.0% ⚠    -4.9%     
  VOLTAS      88     CON DUR    01-Oct-24   1,838.5     1,312.5     1079   ₹-567,556     -28.6% ⚠    -11.6%    
  ⚠  WAZ < 0 (momentum below universe mean): CGPOWER, OFSS, KALYANKJIL, VOLTAS, BOSCHLTD

  AFTER: Invested ₹32,887,241 | Cash ₹1,767,713 | Total ₹34,654,953 | Positions 17/20 | Slot ₹1,732,748

========================================================================
  REBALANCE #62  —  01 Apr 2025
  NAV: ₹32,808,852  |  Slot: ₹1,640,443  |  Cash: ₹1,767,713
========================================================================
  [SECTOR CAP≤4] dropped: CHOLAFIN, HDFCLIFE, BAJAJHLDNG

  [REGIME OFF] Nifty 200 12,809.3 < EMA200 13,066.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       54     CONSUMP    01-Jun-23   1,558.5     5,565.3     556    ₹2,227,796    +257.1%     +6.8%     
  BSE         6      FIN SVC    03-Jun-24   894.9       1,816.3     1803   ₹1,661,300    +103.0%     +15.8%    
  BOSCHLTD    —      AUTO       01-Dec-23   21,526.2    27,510.7    53     ₹317,179      +27.8%      +1.1%     
  PERSISTENT  60     IT         01-Aug-24   4,750.7     5,179.0     376    ₹161,019      +9.0% ⚠     -3.6%     
  KALYANKJIL  114    CON DUR    01-Apr-24   423.7       456.7       3390   ₹111,890      +7.8% ⚠     -1.1%     
  ICICIBANK   16     PVT BNK    02-Dec-24   1,294.7     1,308.4     1449   ₹19,843       +1.1%       +1.7%     
  INDHOTEL    29     CONSUMP    02-Dec-24   795.2       799.8       2360   ₹11,011       +0.6%       +2.7%     
  TORNTPHARM  31     HEALTH     01-Aug-24   3,146.3     3,150.0     569    ₹2,090        +0.1%       +0.8%     
  LUPIN       72     HEALTH     01-Aug-24   1,941.3     1,943.3     922    ₹1,878        +0.1% ⚠     -3.4%     
  SUNPHARMA   86     HEALTH     01-Aug-24   1,688.4     1,681.9     1060   ₹-6,902       -0.4% ⚠     -0.7%     
  UNITDSPR    56     FMCG       01-Oct-24   1,589.7     1,387.6     1248   ₹-252,216     -12.7% ⚠    +2.6%     
  HCLTECH     125    IT         01-Oct-24   1,693.6     1,450.8     1172   ₹-284,652     -14.3% ⚠    -3.9%     
  INFY        118    IT         01-Aug-24   1,741.4     1,451.2     1028   ₹-298,316     -16.7% ⚠    -6.5%     
  CGPOWER     —      ENERGY     02-Dec-24   752.0       613.6       2496   ₹-345,433     -18.4%      -1.2%     
  OBEROIRLTY  121    REALTY     02-Dec-24   2,054.7     1,564.3     913    ₹-447,734     -23.9%      -2.1%     
  VOLTAS      82     CON DUR    01-Oct-24   1,838.5     1,340.3     1079   ₹-537,499     -27.1% ⚠    -4.3%     
  OFSS        131    IT         01-Aug-24   10,120.6    7,036.0     176    ₹-542,905     -30.5% ⚠    -3.4%     
  ⚠  WAZ < 0 (momentum below universe mean): UNITDSPR, PERSISTENT, LUPIN, VOLTAS, SUNPHARMA, KALYANKJIL, INFY, HCLTECH, OFSS

  AFTER: Invested ₹31,041,139 | Cash ₹1,767,713 | Total ₹32,808,852 | Positions 17/20 | Slot ₹1,640,443

========================================================================
  REBALANCE #63  —  02 Jun 2025
  NAV: ₹35,888,366  |  Slot: ₹1,794,418  |  Cash: ₹1,767,713
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TRENT       65     CONSUMP    01-Jun-23   1,558.5     5,610.0     556    ₹2,252,625    +260.0%   732d  
  BSE         2      FIN SVC    03-Jun-24   894.9       2,693.3     1803   ₹3,242,582    +201.0%   364d  
  BOSCHLTD    —      AUTO       01-Dec-23   21,526.2    30,811.8    53     ₹492,138      +43.1%    549d  
  SUNPHARMA   70     HEALTH     01-Aug-24   1,688.4     1,658.3     1060   ₹-31,832      -1.8%     305d  
  INDHOTEL    42     CONSUMP    02-Dec-24   795.2       777.8       2360   ₹-40,879      -2.2%     182d  
  CGPOWER     —      ENERGY     02-Dec-24   752.0       677.6       2496   ₹-185,792     -9.9%     182d  
  INFY        124    IT         01-Aug-24   1,741.4     1,498.0     1028   ₹-250,269     -14.0%    305d  
  OBEROIRLTY  80     REALTY     02-Dec-24   2,054.7     1,759.4     913    ₹-269,560     -14.4%    182d  
  OFSS        82     IT         01-Aug-24   10,120.6    8,038.0     176    ₹-366,540     -20.6%    305d  
  VOLTAS      125    CON DUR    01-Oct-24   1,838.5     1,235.2     1079   ₹-650,936     -32.8%    244d  

  ENTRIES (12)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SOLARINDS   1      DEFENCE    4.247    1.12   +68.3%    +83.8%    16,284.3    110    ₹1,791,268    +10.5%    
  BHARTIHEXA  2      IT         2.926    1.13   +80.6%    +46.6%    1,840.5     974    ₹1,792,608    +7.6%     
  HDFCLIFE    3      FIN SVC    2.572    0.72   +36.3%    +24.3%    761.9       2355   ₹1,794,207    +1.4%     
  DIVISLAB    5      HEALTH     2.298    0.58   +54.6%    +14.7%    6,509.4     275    ₹1,790,074    +2.0%     
  BHARTIARTL  6      CONSUMP    2.159    0.94   +34.7%    +15.8%    1,838.7     975    ₹1,792,760    +0.7%     
  SBILIFE     7      FIN SVC    2.100    0.79   +28.1%    +21.5%    1,800.0     996    ₹1,792,798    +2.0%     
  HDFCBANK    8      PVT BNK    2.081    0.88   +26.5%    +15.2%    937.7       1913   ₹1,793,739    +0.5%     
  PAGEIND     10     MFG        1.635    0.56   +28.4%    +12.0%    45,270.3    39     ₹1,765,541    -1.0%     
  BAJFINANCE  11     FIN SVC    1.633    1.06   +33.7%    +9.8%     906.3       1979   ₹1,793,526    +0.4%     
  MAXHEALTH   12     HEALTH     1.623    0.78   +43.4%    +16.6%    1,150.3     1559   ₹1,793,275    +0.4%     
  TVSMOTOR    13     AUTO       1.575    1.07   +23.4%    +17.5%    2,753.8     651    ₹1,792,707    +0.1%     
  FEDERALBNK  14     PVT BNK    1.560    0.78   +26.8%    +13.8%    205.0       8752   ₹1,794,238    +3.3%     

  HOLDS (7)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  KALYANKJIL  40     CON DUR    01-Apr-24   423.7       555.0       3390   ₹445,169      +31.0%      +1.4%     
  PERSISTENT  56     IT         01-Aug-24   4,750.7     5,484.5     376    ₹275,898      +15.4%      -0.9%     
  ICICIBANK   7      PVT BNK    02-Dec-24   1,294.7     1,439.4     1449   ₹209,719      +11.2%      +0.9%     
  LUPIN       69     HEALTH     01-Aug-24   1,941.3     1,949.0     922    ₹7,101        +0.4% ⚠     -1.8%     
  TORNTPHARM  59     HEALTH     01-Aug-24   3,146.3     3,094.5     569    ₹-29,490      -1.6% ⚠     -2.5%     
  UNITDSPR    12     FMCG       01-Oct-24   1,589.7     1,533.0     1248   ₹-70,712      -3.6%       +0.6%     
  HCLTECH     67     IT         01-Oct-24   1,693.6     1,564.5     1172   ₹-151,357     -7.6% ⚠     +0.2%     
  ⚠  WAZ < 0 (momentum below universe mean): TORNTPHARM, HCLTECH, LUPIN

  AFTER: Invested ₹34,820,630 | Cash ₹1,042,223 | Total ₹35,862,853 | Positions 19/20 | Slot ₹1,794,418

========================================================================
  REBALANCE #64  —  01 Aug 2025
  NAV: ₹35,455,503  |  Slot: ₹1,772,775  |  Cash: ₹1,042,223
========================================================================
  [SECTOR CAP≤4] dropped: APOLLOHOSP

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  KALYANKJIL  31     CON DUR    01-Apr-24   423.7       581.1       3390   ₹533,728      +37.2%    487d  
  PERSISTENT  79     IT         01-Aug-24   4,750.7     5,043.4     376    ₹110,053      +6.2%     365d  
  TVSMOTOR    33     AUTO       02-Jun-25   2,753.8     2,848.2     651    ₹61,500       +3.4%     60d   
  LUPIN       101    HEALTH     01-Aug-24   1,941.3     1,867.3     922    ₹-68,206      -3.8%     365d  
  HCLTECH     109    IT         01-Oct-24   1,693.6     1,403.4     1172   ₹-340,195     -17.1%    304d  
  UNITDSPR    128    FMCG       01-Oct-24   1,589.7     1,316.4     1248   ₹-340,995     -17.2%    304d  

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MUTHOOTFIN  1      FIN SVC    2.921    0.69   +45.1%    +15.3%    2,571.0     689    ₹1,771,392    -1.5%     
  ETERNAL     2      CONSUMP    2.758    1.13   +34.2%    +31.0%    304.8       5817   ₹1,772,731    +5.8%     
  MANKIND     10     HEALTH     1.937    0.62   +24.8%    +8.3%     2,565.3     691    ₹1,772,615    +0.4%     
  INDIANB     11     PSU BNK    1.808    1.03   +6.0%     +13.5%    608.2       2914   ₹1,772,420    -1.4%     
  BRITANNIA   15     FMCG       1.613    0.46   +0.5%     +7.5%     5,723.0     309    ₹1,768,407    +1.3%     
  POLYCAB     16     MFG        1.608    1.10   +0.8%     +14.0%    6,666.4     265    ₹1,766,588    -1.3%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TORNTPHARM  10     HEALTH     01-Aug-24   3,146.3     3,645.9     569    ₹284,283      +15.9%      +3.8%     
  ICICIBANK   13     PVT BNK    02-Dec-24   1,294.7     1,460.3     1449   ₹240,058      +12.8%      +0.7%     
  MAXHEALTH   14     HEALTH     02-Jun-25   1,150.3     1,246.0     1559   ₹149,239      +8.3%       -0.4%     
  HDFCBANK    9      PVT BNK    02-Jun-25   937.7       989.7       1913   ₹99,615       +5.6%       +0.7%     
  BHARTIARTL  16     CONSUMP    02-Jun-25   1,838.7     1,884.4     975    ₹44,530       +2.5%       -2.0%     
  PAGEIND     36     MFG        02-Jun-25   45,270.3    46,152.7    39     ₹34,415       +1.9%       -1.3%     
  BHARTIHEXA  11     IT         02-Jun-25   1,840.5     1,844.6     974    ₹4,032        +0.2%       +2.0%     
  SBILIFE     42     FIN SVC    02-Jun-25   1,800.0     1,793.7     996    ₹-6,266       -0.3%       -1.3%     
  DIVISLAB    15     HEALTH     02-Jun-25   6,509.4     6,361.5     275    ₹-40,662      -2.3%       -4.2%     
  HDFCLIFE    43     FIN SVC    02-Jun-25   761.9       739.1       2355   ₹-53,707      -3.0%       -2.5%     
  BAJFINANCE  23     FIN SVC    02-Jun-25   906.3       870.3       1979   ₹-71,300      -4.0%       -4.2%     
  FEDERALBNK  53     PVT BNK    02-Jun-25   205.0       194.9       8752   ₹-88,646      -4.9%       -5.7%     
  SOLARINDS   22     DEFENCE    02-Jun-25   16,284.3    13,807.0    110    ₹-272,498     -15.2%      -8.1%     

  AFTER: Invested ₹34,307,480 | Cash ₹1,135,408 | Total ₹35,442,888 | Positions 19/20 | Slot ₹1,772,775

========================================================================
  REBALANCE #65  —  01 Oct 2025
  NAV: ₹35,648,052  |  Slot: ₹1,782,403  |  Cash: ₹1,135,408
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ETERNAL     —      CONSUMP    01-Aug-25   304.8       329.0       5817   ₹141,062      +8.0%     61d   
  BHARTIARTL  66     CONSUMP    02-Jun-25   1,838.7     1,867.6     975    ₹28,150       +1.6%     121d  
  SBILIFE     74     FIN SVC    02-Jun-25   1,800.0     1,798.6     996    ₹-1,392       -0.1%     121d  
  MAXHEALTH   69     HEALTH     02-Jun-25   1,150.3     1,113.2     1559   ₹-57,796      -3.2%     121d  
  MANKIND     —      HEALTH     01-Aug-25   2,565.3     2,439.6     691    ₹-86,851      -4.9%     61d   
  FEDERALBNK  —      PVT BNK    02-Jun-25   205.0       193.8       8752   ₹-98,451      -5.5%     121d  
  PAGEIND     86     MFG        02-Jun-25   45,270.3    41,660.5    39     ₹-140,782     -8.0%     121d  
  DIVISLAB    97     HEALTH     02-Jun-25   6,509.4     5,710.0     275    ₹-219,824     -12.3%    121d  
  SOLARINDS   89     DEFENCE    02-Jun-25   16,284.3    13,374.0    110    ₹-320,128     -17.9%    121d  

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  EICHERMOT   1      AUTO       3.627    1.09   +42.4%    +24.3%    7,021.5     253    ₹1,776,440    +3.0%     
  FORTIS      2      HEALTH     3.545    0.73   +62.4%    +25.0%    989.7       1801   ₹1,782,360    +3.4%     
  HEROMOTOCO  5      AUTO       2.461    1.02   -6.6%     +29.9%    5,328.6     334    ₹1,779,765    +2.3%     
  BOSCHLTD    6      AUTO       2.335    0.87   +4.6%     +19.7%    38,320.0    46     ₹1,762,720    -2.3%     
  NYKAA       7      CONSUMP    2.312    0.65   +20.0%    +14.0%    241.3       7387   ₹1,782,188    +2.1%     
  GODFRYPHLP  9      FMCG       1.937    1.15   +43.6%    +15.1%    3,357.6     530    ₹1,779,508    -1.7%     
  COROMANDEL  10     MFG        1.875    0.92   +38.7%    -0.6%     2,242.2     794    ₹1,780,277    -0.5%     
  SBIN        11     PSU BNK    1.863    0.97   +9.9%     +6.3%     848.8       2099   ₹1,781,632    +1.9%     
  BANKINDIA   12     PSU BNK    1.676    1.12   +16.9%    +4.5%     120.7       14768  ₹1,782,315    +4.9%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  3      FIN SVC    01-Aug-25   2,571.0     3,118.1     689    ₹377,003      +21.3%      +5.7%     
  INDIANB     5      PSU BNK    01-Aug-25   608.2       721.7       2914   ₹330,545      +18.6%      +4.8%     
  TORNTPHARM  25     HEALTH     01-Aug-24   3,146.3     3,535.8     569    ₹221,622      +12.4%      -0.6%     
  POLYCAB     27     MFG        01-Aug-25   6,666.4     7,316.3     265    ₹172,228      +9.7%       +0.2%     
  BAJFINANCE  10     FIN SVC    02-Jun-25   906.3       981.7       1979   ₹149,190      +8.3%       +0.7%     
  ICICIBANK   55     PVT BNK    02-Dec-24   1,294.7     1,372.0     1449   ₹112,065      +6.0%       -1.3%     
  BRITANNIA   39     FMCG       01-Aug-25   5,723.0     5,966.5     309    ₹75,242       +4.3%       -0.3%     
  HDFCBANK    37     PVT BNK    02-Jun-25   937.7       949.5       1913   ₹22,740       +1.3%       +0.4%     
  HDFCLIFE    52     FIN SVC    02-Jun-25   761.9       761.3       2355   ₹-1,377       -0.1%       -0.7%     
  BHARTIHEXA  63     IT         02-Jun-25   1,840.5     1,660.8     974    ₹-174,989     -9.8% ⚠     -2.7%     
  ⚠  WAZ < 0 (momentum below universe mean): BHARTIHEXA

  AFTER: Invested ₹35,210,559 | Cash ₹418,485 | Total ₹35,629,044 | Positions 19/20 | Slot ₹1,782,403

========================================================================
  REBALANCE #66  —  01 Dec 2025
  NAV: ₹37,452,384  |  Slot: ₹1,872,619  |  Cash: ₹418,485
========================================================================
  [SECTOR CAP≤4] dropped: PNB

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  POLYCAB     77     MFG        01-Aug-25   6,666.4     7,366.0     265    ₹185,415      +10.5%    122d  
  ICICIBANK   78     PVT BNK    02-Dec-24   1,294.7     1,390.1     1449   ₹138,292      +7.4%     364d  
  HDFCLIFE    76     FIN SVC    02-Jun-25   761.9       764.0       2355   ₹5,077        +0.3%     182d  
  BHARTIHEXA  64     IT         02-Jun-25   1,840.5     1,748.7     974    ₹-89,374      -5.0%     182d  
  BOSCHLTD    115    AUTO       01-Oct-25   38,320.0    36,335.0    46     ₹-91,310      -5.2%     61d   
  GODFRYPHLP  89     FMCG       01-Oct-25   3,357.6     2,837.9     530    ₹-275,421     -15.5%    61d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  CANBK       2      PSU BNK    3.171    1.00   +53.2%    +44.9%    145.7       12853  ₹1,872,603    +3.7%     
  M&MFIN      4      FIN SVC    2.857    1.20   +39.5%    +44.9%    367.9       5090   ₹1,872,611    +9.0%     
  MARUTI      7      AUTO       2.296    0.79   +48.7%    +8.8%     16,097.0    116    ₹1,867,252    +1.2%     
  TVSMOTOR    12     AUTO       2.060    1.19   +51.7%    +11.8%    3,649.0     513    ₹1,871,960    +4.5%     
  RELIANCE    13     OIL&GAS    1.890    1.16   +21.4%    +15.4%    1,558.9     1201   ₹1,872,230    +2.7%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  2      FIN SVC    01-Aug-25   2,571.0     3,779.1     689    ₹832,439      +47.0%      +6.6%     
  INDIANB     4      PSU BNK    01-Aug-25   608.2       868.8       2914   ₹759,384      +42.8%      +2.6%     
  BANKINDIA   8      PSU BNK    01-Oct-25   120.7       142.6       14768  ₹323,342      +18.1%      +1.8%     
  TORNTPHARM  54     HEALTH     01-Aug-24   3,146.3     3,704.1     569    ₹317,390      +17.7%      +0.5%     
  HEROMOTOCO  11     AUTO       01-Oct-25   5,328.6     6,175.1     334    ₹282,731      +15.9%      +7.1%     
  SBIN        9      PSU BNK    01-Oct-25   848.8       955.9       2099   ₹224,740      +12.6%      +1.1%     
  BAJFINANCE  10     FIN SVC    02-Jun-25   906.3       1,014.9     1979   ₹214,885      +12.0%      -0.3%     
  NYKAA       16     CONSUMP    01-Oct-25   241.3       264.9       7387   ₹174,629      +9.8%       +0.6%     
  COROMANDEL  53     MFG        01-Oct-25   2,242.2     2,382.9     794    ₹111,752      +6.3%       +5.3%     
  HDFCBANK    42     PVT BNK    02-Jun-25   937.7       985.8       1913   ₹92,087       +5.1%       +0.5%     
  BRITANNIA   62     FMCG       01-Aug-25   5,723.0     5,813.5     309    ₹27,964       +1.6% ⚠     -1.0%     
  EICHERMOT   7      AUTO       01-Oct-25   7,021.5     7,125.5     253    ₹26,312       +1.5%       +1.6%     
  FORTIS      58     HEALTH     01-Oct-25   989.7       904.8       1801   ₹-152,725     -8.6%       -4.9%     
  ⚠  WAZ < 0 (momentum below universe mean): BRITANNIA

  AFTER: Invested ₹35,746,283 | Cash ₹1,694,991 | Total ₹37,441,274 | Positions 18/20 | Slot ₹1,872,619

========================================================================
  REBALANCE #67  —  02 Feb 2026
  NAV: ₹35,817,574  |  Slot: ₹1,790,879  |  Cash: ₹1,694,991
========================================================================
  [SECTOR CAP≤4] dropped: UNIONBANK, HINDZINC

  [REGIME OFF] Nifty 200 13,949.8 < EMA200 14,017.7 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  11     FIN SVC    01-Aug-25   2,571.0     3,505.5     689    ₹643,869      +36.3%      -8.4%     
  INDIANB     21     PSU BNK    01-Aug-25   608.2       817.4       2914   ₹609,448      +34.4%      -3.0%     
  TORNTPHARM  18     HEALTH     01-Aug-24   3,146.3     3,952.2     569    ₹458,560      +25.6%      +0.8%     
  BANKINDIA   20     PSU BNK    01-Oct-25   120.7       146.9       14768  ₹386,695      +21.7%      -3.1%     
  SBIN        6      PSU BNK    01-Oct-25   848.8       1,010.5     2099   ₹339,378      +19.0%      -0.3%     
  HEROMOTOCO  26     AUTO       01-Oct-25   5,328.6     5,515.5     334    ₹62,411       +3.5%       -0.1%     
  BRITANNIA   47     FMCG       01-Aug-25   5,723.0     5,888.5     309    ₹51,140       +2.9%       -0.1%     
  TVSMOTOR    10     AUTO       01-Dec-25   3,649.0     3,633.1     513    ₹-8,179       -0.4%       -0.6%     
  EICHERMOT   28     AUTO       01-Oct-25   7,021.5     6,985.5     253    ₹-9,108       -0.5%       -2.8%     
  BAJFINANCE  82     FIN SVC    02-Jun-25   906.3       898.2       1979   ₹-16,030      -0.9% ⚠     -4.4%     
  NYKAA       36     CONSUMP    01-Oct-25   241.3       237.6       7387   ₹-26,889      -1.5%       -3.3%     
  COROMANDEL  38     MFG        01-Oct-25   2,242.2     2,206.9     794    ₹-27,997      -1.6%       -2.7%     
  HDFCBANK    66     PVT BNK    02-Jun-25   937.7       913.0       1913   ₹-47,171      -2.6% ⚠     -1.2%     
  CANBK       9      PSU BNK    01-Dec-25   145.7       141.8       12853  ₹-50,143      -2.7%       -3.6%     
  M&MFIN      23     FIN SVC    01-Dec-25   367.9       353.6       5090   ₹-72,787      -3.9%       -2.9%     
  MARUTI      81     AUTO       01-Dec-25   16,097.0    14,384.0    116    ₹-198,708     -10.6% ⚠    -7.8%     
  RELIANCE    72     OIL&GAS    01-Dec-25   1,558.9     1,384.0     1201   ₹-210,045     -11.2% ⚠    -3.1%     
  FORTIS      69     HEALTH     01-Oct-25   989.7       838.0       1801   ₹-273,212     -15.3% ⚠    -3.6%     
  ⚠  WAZ < 0 (momentum below universe mean): HDFCBANK, FORTIS, RELIANCE, MARUTI, BAJFINANCE

  AFTER: Invested ₹34,122,583 | Cash ₹1,694,991 | Total ₹35,817,574 | Positions 18/20 | Slot ₹1,790,879

========================================================================
  REBALANCE #68  —  01 Apr 2026
  NAV: ₹33,678,052  |  Slot: ₹1,683,903  |  Cash: ₹1,694,991
========================================================================

  [REGIME OFF] Nifty 200 12,720.3 < EMA200 13,909.6 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  INDIANB     5      PSU BNK    01-Aug-25   608.2       869.5       2914   ₹761,239      +42.9%      -0.2%     
  TORNTPHARM  10     HEALTH     01-Aug-24   3,146.3     4,103.1     569    ₹544,452      +30.4%      -3.5%     
  MUTHOOTFIN  45     FIN SVC    01-Aug-25   2,571.0     3,228.7     689    ₹453,182      +25.6%      -1.7%     
  SBIN        11     PSU BNK    01-Oct-25   848.8       999.8       2099   ₹316,904      +17.8%      -4.7%     
  BANKINDIA   27     PSU BNK    01-Oct-25   120.7       137.2       14768  ₹244,115      +13.7%      -6.2%     
  NYKAA       33     CONSUMP    01-Oct-25   241.3       240.0       7387   ₹-9,529       -0.5%       -2.1%     
  EICHERMOT   37     AUTO       01-Oct-25   7,021.5     6,825.5     253    ₹-49,588      -2.8%       -3.2%     
  HEROMOTOCO  26     AUTO       01-Oct-25   5,328.6     5,122.0     334    ₹-69,017      -3.9%       -3.5%     
  BRITANNIA   52     FMCG       01-Aug-25   5,723.0     5,474.0     309    ₹-76,941      -4.4%       -4.1%     
  TVSMOTOR    23     AUTO       01-Dec-25   3,649.0     3,425.8     513    ₹-114,525     -6.1%       -2.8%     
  BAJFINANCE  99     FIN SVC    02-Jun-25   906.3       812.3       1979   ₹-185,971     -10.4% ⚠    -6.7%     
  RELIANCE    70     OIL&GAS    01-Dec-25   1,558.9     1,362.9     1201   ₹-235,389     -12.6% ⚠    -1.6%     
  COROMANDEL  81     MFG        01-Oct-25   2,242.2     1,918.7     794    ₹-256,829     -14.4% ⚠    -4.5%     
  CANBK       44     PSU BNK    01-Dec-25   145.7       123.2       12853  ₹-288,667     -15.4%      -6.8%     
  FORTIS      49     HEALTH     01-Oct-25   989.7       795.0       1801   ₹-350,565     -19.7%      -5.3%     
  M&MFIN      —      FIN SVC    01-Dec-25   367.9       289.7       5090   ₹-398,038     -21.3%      -10.3%    
  HDFCBANK    138    PVT BNK    02-Jun-25   937.7       730.2       1913   ₹-396,918     -22.1% ⚠    -8.1%     
  MARUTI      109    AUTO       01-Dec-25   16,097.0    12,509.0    116    ₹-416,208     -22.3% ⚠    -4.6%     
  ⚠  WAZ < 0 (momentum below universe mean): RELIANCE, COROMANDEL, BAJFINANCE, MARUTI, HDFCBANK

  AFTER: Invested ₹31,983,061 | Cash ₹1,694,991 | Total ₹33,678,052 | Positions 18/20 | Slot ₹1,683,903

========================================================================
  REBALANCE #69  —  01 Jun 2026
  NAV: ₹33,838,534  |  Slot: ₹1,691,927  |  Cash: ₹1,694,991
========================================================================
  [SECTOR CAP≤4] dropped: ADANIGREEN, PREMIERENE, CGPOWER

  [REGIME OFF] Nifty 200 13,528.3 < EMA200 13,828.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TORNTPHARM  29     HEALTH     01-Aug-24   3,146.3     4,350.4     569    ₹685,140      +38.3%      -1.1%     
  INDIANB     72     PSU BNK    01-Aug-25   608.2       792.9       2914   ₹537,974      +30.4% ⚠    -3.3%     
  MUTHOOTFIN  39     FIN SVC    01-Aug-25   2,571.0     3,246.4     689    ₹465,377      +26.3%      -3.3%     
  BANKINDIA   89     PSU BNK    01-Oct-25   120.7       136.7       14768  ₹237,062      +13.3% ⚠    -1.3%     
  SBIN        99     PSU BNK    01-Oct-25   848.8       954.1       2099   ₹221,024      +12.4% ⚠    -2.5%     
  NYKAA       45     CONSUMP    01-Oct-25   241.3       266.7       7387   ₹187,925      +10.5%      -0.3%     
  EICHERMOT   62     AUTO       01-Oct-25   7,021.5     7,100.5     253    ₹19,987       +1.1%       -0.9%     
  BAJFINANCE  110    FIN SVC    02-Jun-25   906.3       883.6       1979   ₹-44,845      -2.5% ⚠     -3.4%     
  FORTIS      42     HEALTH     01-Oct-25   989.7       929.2       1801   ₹-108,870     -6.1%       -2.1%     
  TVSMOTOR    86     AUTO       01-Dec-25   3,649.0     3,344.4     513    ₹-156,283     -8.3% ⚠     -2.8%     
  HEROMOTOCO  98     AUTO       01-Oct-25   5,328.6     4,819.9     334    ₹-169,918     -9.5% ⚠     -4.1%     
  BRITANNIA   136    FMCG       01-Aug-25   5,723.0     5,157.5     309    ₹-174,740     -9.9% ⚠     -4.3%     
  CANBK       96     PSU BNK    01-Dec-25   145.7       123.9       12853  ₹-280,455     -15.0% ⚠    -2.8%     
  RELIANCE    106    OIL&GAS    01-Dec-25   1,558.9     1,313.9     1201   ₹-294,206     -15.7% ⚠    -2.8%     
  MARUTI      109    AUTO       01-Dec-25   16,097.0    12,946.0    116    ₹-365,516     -19.6% ⚠    -1.8%     
  M&MFIN      103    FIN SVC    01-Dec-25   367.9       295.1       5090   ₹-370,552     -19.8% ⚠    -4.5%     
  COROMANDEL  —      MFG        01-Oct-25   2,242.2     1,787.7     794    ₹-360,843     -20.3%      -4.3%     
  HDFCBANK    149    PVT BNK    02-Jun-25   937.7       730.6       1913   ₹-396,071     -22.1% ⚠    -3.0%     
  ⚠  WAZ < 0 (momentum below universe mean): INDIANB, TVSMOTOR, BANKINDIA, CANBK, HEROMOTOCO, SBIN, M&MFIN, RELIANCE, MARUTI, BAJFINANCE, BRITANNIA, HDFCBANK

  AFTER: Invested ₹32,143,543 | Cash ₹1,694,991 | Total ₹33,838,534 | Positions 18/20 | Slot ₹1,691,927

========================================================================
  FINAL SUMMARY
========================================================================
  Period        : 2015-01-01 → 2026-07-01  (11.5 years)
  Starting Cap  : ₹2,000,000
  Final Value   : ₹35,437,134
  Total Return  : +1671.9%  (on total invested)
  CAGR          : +28.4%

  Closed Trades : 309  |  Open: 18
  Win Rate      : 59.9%  (185W / 124L)
  Profit Factor : 4.98
  Avg hold      : 220 days
  Total charges : ₹640,222
  Closed net P&L: ₹32,038,616
  Open unreal   : ₹1,230,790

  YEAR-BY-YEAR:
  2015  +  5.1%  █████
  2016  + 12.7%  ████████████
  2017  + 52.9%  ████████████████████████████████████████
  2018  +  3.3%  ███
  2019  + 14.1%  ██████████████
  2020  + 30.3%  ██████████████████████████████
  2021  + 92.7%  ████████████████████████████████████████
  2022  + 50.2%  ████████████████████████████████████████
  2023  + 42.4%  ████████████████████████████████████████
  2024  + 63.7%  ████████████████████████████████████████
  2025  -  0.2%  
  2026  -  9.6%  ░░░░░░░░░

  Rebalance NAV exported → mom20_rebal.csv (69 rows)
