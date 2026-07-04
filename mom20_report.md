=== Mom20 — 2-Monthly Rebalance, β≤1.2 | Regime ON [EMA200] | Sector≤4 ===
    top_n=20 buffer_in=15 buffer_out=40 beta_cap=1.2
Loading PIT universe...
  388 unique PIT tickers across all periods
Loading EPS data...
  871 stocks with EPS data
  Sector map loaded: 43 PIT dates
Loading cached data from /Users/jay/dev/relative_strength/data/cache/mom15_daily.pkl...
Fetching Nifty 200 (beta)...
  3133 bars
Fetching Nifty 200 (regime filter)...
  3133 bars
  Trading days in backtest: 2840 (2015-01-01 → 2026-07-01)
  Rebalance dates: 69

==============================================================================================
  MOM20 PIT BACKTEST  |  NAV/20 slot  |  2-Month Rebalance  |  Beta≤1.2  |  Regime ON [EMA200]
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
  BOSCHLTD    1      AUTO       3.997    0.37   +163.6%   +67.5%    22,437.0    4      ₹89,748       +14.3%    
  HONAUT      2      MFG        3.183    0.41   +184.0%   +40.5%    7,224.6     13     ₹93,919       +3.3%     
  AXISBANK    3      PVT BNK    3.080    0.39   +168.1%   +43.5%    597.6       167    ₹99,792       +11.9%    
  WHIRLPOOL   4      CON DUR    3.049    0.58   +277.4%   +50.5%    693.1       144    ₹99,799       +3.3%     
  BEL         5      DEFENCE    3.045    0.47   +245.9%   +67.9%    28.5        3505   ₹99,988       +6.7%     
  EICHERMOT   6      AUTO       2.714    0.58   +229.2%   +32.2%    1,530.4     65     ₹99,477       +6.3%     
  SHREECEM    7      INFRA      2.649    0.56   +147.9%   +25.9%    10,437.7    9      ₹93,939       +6.1%     
  IBULHSGFIN  8      FIN SVC    2.640    0.59   +214.3%   +50.8%    382.5       261    ₹99,838       +8.0%     
  ASHOKLEY    9      AUTO       2.616    0.70   +293.2%   +42.7%    26.9        3716   ₹99,986       +6.9%     
  BAJFINANCE  10     FIN SVC    2.606    0.32   +167.1%   +48.8%    39.9        2507   ₹99,987       +7.9%     
  BBTC        11     FMCG       2.544    0.90   +322.4%   +84.3%    438.1       228    ₹99,890       +1.1%     
  BHARATFORG  12     DEFENCE    2.543    0.40   +210.7%   +33.0%    490.3       203    ₹99,535       +6.6%     
  BAJAJFINSV  13     FIN SVC    2.438    0.33   +121.0%   +40.6%    149.9       667    ₹99,993       +11.8%    
  AUROPHARMA  14     HEALTH     2.396    0.21   +209.3%   +32.8%    596.9       167    ₹99,688       +7.4%     
  AMARAJABAT  15     AUTO       2.338    -0.09  +169.8%   +37.1%    801.8       124    ₹99,418       +4.9%     
  SRF         16     MFG        2.337    0.41   +378.9%   +7.4%     181.9       549    ₹99,844       +4.3%     
  WOCKPHARMA  17     HEALTH     2.198    0.39   +196.2%   +63.7%    1,099.6     90     ₹98,968       +9.7%     
  SPARC       18     HEALTH     2.190    0.61   +127.8%   +83.7%    358.5       278    ₹99,652       +30.6%    
  REPCOHOME   19     FIN SVC    2.173    0.43   +109.9%   +38.5%    609.4       164    ₹99,943       -0.6%     
  ALSTOMT&D   20     ENERGY     2.136    0.50   +124.9%   +41.8%    673.8       148    ₹99,719       +2.6%     

  HOLDS (0)
    —

  AFTER: Invested ₹1,973,127 | Cash ₹24,531 | Total ₹1,997,657 | Positions 20/20 | Slot ₹100,000

========================================================================
  REBALANCE #02  —  01 Apr 2015
  NAV: ₹2,126,929  |  Slot: ₹106,346  |  Cash: ₹24,531
========================================================================
  [SECTOR CAP≤4] dropped: NATCOPHARM, SUNPHARMA

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ALSTOMT&D   51     ENERGY     02-Feb-15   673.8       710.7       148    ₹5,467        +5.5%     58d   
  AUROPHARMA  44     HEALTH     02-Feb-15   596.9       590.1       167    ₹-1,145       -1.1%     58d   
  EICHERMOT   31     AUTO       02-Feb-15   1,530.4     1,485.0     65     ₹-2,954       -3.0%     58d   
  REPCOHOME   83     FIN SVC    02-Feb-15   609.4       580.5       164    ₹-4,746       -4.7%     58d   
  AMARAJABAT  61     AUTO       02-Feb-15   801.8       761.8       124    ₹-4,955       -5.0%     58d   
  BAJAJFINSV  52     FIN SVC    02-Feb-15   149.9       141.8       667    ₹-5,416       -5.4%     58d   
  AXISBANK    38     PVT BNK    02-Feb-15   597.6       551.4       167    ₹-7,703       -7.7%     58d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  LUPIN       2      HEALTH     3.589    0.42   +116.3%   +42.7%    1,917.4     55     ₹105,460      +6.5%     
  SADBHAV     7      INFRA      2.803    0.39   +291.5%   +40.1%    342.4       310    ₹106,148      +1.8%     
  SIEMENS     9      ENERGY     2.631    0.81   +108.8%   +59.3%    737.5       144    ₹106,205      +3.3%     
  BRITANNIA   10     FMCG       2.417    0.37   +157.9%   +22.1%    946.0       112    ₹105,946      +1.4%     
  AJANTPHARM  11     HEALTH     2.353    0.36   +208.0%   +30.3%    737.1       144    ₹106,143      +2.9%     
  EMAMILTD    16     FMCG       2.177    0.23   +126.1%   +26.4%    420.8       252    ₹106,047      -1.9%     

  HOLDS (13)
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
  BAJFINANCE  25     FIN SVC    02-Feb-15   39.9        39.7        2507   ₹-482         -0.5%       +1.9%     
  BBTC        20     FMCG       02-Feb-15   438.1       435.4       228    ₹-619         -0.6%       -1.0%     
  IBULHSGFIN  22     FIN SVC    02-Feb-15   382.5       368.0       261    ₹-3,800       -3.8%       +0.6%     

  AFTER: Invested ₹2,061,769 | Cash ₹64,405 | Total ₹2,126,174 | Positions 19/20 | Slot ₹106,346

========================================================================
  REBALANCE #03  —  01 Jun 2015
  NAV: ₹2,114,891  |  Slot: ₹105,745  |  Cash: ₹64,405
========================================================================
  [SECTOR CAP≤4] dropped: APLLTD, AUROPHARMA, ZYDUSLIFE

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  WOCKPHARMA  70     HEALTH     02-Feb-15   1,099.6     1,341.2     90     ₹21,738       +22.0%    119d  
  BBTC        10     FMCG       02-Feb-15   438.1       500.5       228    ₹14,223       +14.2%    119d  
  BEL         37     DEFENCE    02-Feb-15   28.5        30.1        3505   ₹5,413        +5.4%     119d  
  IBULHSGFIN  60     FIN SVC    02-Feb-15   382.5       397.3       261    ₹3,849        +3.9%     119d  
  HONAUT      97     MFG        02-Feb-15   7,224.6     7,173.1     13     ₹-669         -0.7%     119d  
  SIEMENS     63     ENERGY     01-Apr-15   737.5       713.3       144    ₹-3,493       -3.3%     61d   
  BOSCHLTD    58     AUTO       02-Feb-15   22,437.0    20,961.6    4      ₹-5,902       -6.6%     119d  
  SADBHAV     98     INFRA      01-Apr-15   342.4       280.9       310    ₹-19,071      -18.0%    61d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  PAGEIND     1      MFG        4.056    0.21   +185.4%   +45.2%    15,087.9    7      ₹105,615      +14.7%    
  NATCOPHARM  4      HEALTH     3.470    1.02   +212.4%   +63.3%    407.7       259    ₹105,589      -0.5%     
  EICHERMOT   5      AUTO       3.062    1.08   +196.3%   +20.8%    1,740.7     60     ₹104,440      +5.2%     
  MARICO      6      FMCG       2.932    0.41   +101.2%   +25.6%    190.3       555    ₹105,635      +8.3%     
  INDUSTOWER  11     INFRA      2.580    0.33   +107.5%   +26.6%    327.7       322    ₹105,506      +9.6%     
  UPL         14     MFG        2.359    0.67   +79.6%    +32.0%    325.1       325    ₹105,666      +4.3%     
  DCBBANK     16     PVT BNK    2.246    0.94   +99.8%    +22.6%    124.9       846    ₹105,675      +3.6%     
  MARUTI      18     AUTO       2.164    0.67   +76.1%    +8.5%     3,524.0     30     ₹105,721      +4.7%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  AJANTPHARM  2      HEALTH     01-Apr-15   737.1       976.8       144    ₹34,516       +32.5%      +15.1%    
  SRF         14     MFG        02-Feb-15   181.9       216.0       549    ₹18,716       +18.7%      +12.8%    
  BRITANNIA   3      FMCG       01-Apr-15   946.0       1,115.5     112    ₹18,992       +17.9%      +6.8%     
  BHARATFORG  28     DEFENCE    02-Feb-15   490.3       560.0       203    ₹14,146       +14.2%      -2.3%     
  EMAMILTD    8      FMCG       01-Apr-15   420.8       479.2       252    ₹14,704       +13.9%      +9.5%     
  SPARC       36     HEALTH     02-Feb-15   358.5       398.2       278    ₹11,059       +11.1%      -2.5%     
  ASHOKLEY    26     AUTO       02-Feb-15   26.9        29.3        3716   ₹8,931        +8.9%       +2.1%     
  BAJFINANCE  18     FIN SVC    02-Feb-15   39.9        42.0        2507   ₹5,283        +5.3%       -0.1%     
  WHIRLPOOL   13     CON DUR    02-Feb-15   693.1       729.2       144    ₹5,201        +5.2%       +2.7%     
  SHREECEM    29     INFRA      02-Feb-15   10,437.7    10,938.8    9      ₹4,510        +4.8%       +2.0%     
  LUPIN       16     HEALTH     01-Apr-15   1,917.4     1,686.5     55     ₹-12,704      -12.0%      +1.8%     

  AFTER: Invested ₹2,083,540 | Cash ₹30,349 | Total ₹2,113,889 | Positions 19/20 | Slot ₹105,745

========================================================================
  REBALANCE #04  —  03 Aug 2015
  NAV: ₹2,230,538  |  Slot: ₹111,527  |  Cash: ₹30,349
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ASHOKLEY    12     AUTO       02-Feb-15   26.9        34.8        3716   ₹29,508       +29.5%    182d  
  SPARC       64     HEALTH     02-Feb-15   358.5       398.4       278    ₹11,114       +11.2%    182d  
  BHARATFORG  96     DEFENCE    02-Feb-15   490.3       539.6       203    ₹10,008       +10.1%    182d  
  SHREECEM    51     INFRA      02-Feb-15   10,437.7    10,979.2    9      ₹4,873        +5.2%     182d  
  NATCOPHARM  50     HEALTH     01-Jun-15   407.7       425.0       259    ₹4,497        +4.3%     63d   
  DCBBANK     55     PVT BNK    01-Jun-15   124.9       128.0       846    ₹2,625        +2.5%     63d   
  UPL         60     MFG        01-Jun-15   325.1       327.1       325    ₹627          +0.6%     63d   
  LUPIN       91     HEALTH     01-Apr-15   1,917.4     1,572.2     55     ₹-18,989      -18.0%    124d  
  PAGEIND     81     MFG        01-Jun-15   15,087.9    12,198.5    7      ₹-20,226      -19.2%    63d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RAJESHEXPO  1      CON DUR    4.528    0.94   +239.0%   +148.2%   561.3       198    ₹111,134      +23.9%    
  HINDPETRO   4      OIL&GAS    3.007    0.81   +142.5%   +53.0%    81.6        1366   ₹111,460      +3.8%     
  APLLTD      6      HEALTH     2.805    0.62   +112.2%   +57.6%    657.5       169    ₹111,125      +0.6%     
  IBULHSGFIN  9      FIN SVC    2.484    0.13   +106.7%   +32.1%    498.5       223    ₹111,172      +9.1%     
  BAJAJFINSV  10     FIN SVC    2.443    0.61   +100.6%   +32.5%    189.4       588    ₹111,374      +9.3%     
  ABBOTINDIA  12     HEALTH     2.241    0.20   +114.7%   +16.7%    4,000.7     27     ₹108,018      +5.7%     
  OFSS        13     IT         2.176    0.37   +45.8%    +22.7%    2,481.5     44     ₹109,187      +5.4%     
  INDUSINDBK  14     PVT BNK    2.149    0.97   +74.0%    +21.3%    915.6       121    ₹110,782      +3.9%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BRITANNIA   2      FMCG       01-Apr-15   946.0       1,370.3     112    ₹47,522       +44.9%      +7.7%     
  SRF         9      MFG        02-Feb-15   181.9       262.4       549    ₹44,239       +44.3%      +5.9%     
  BAJFINANCE  5      FIN SVC    02-Feb-15   39.9        54.7        2507   ₹37,059       +37.1%      +7.3%     
  EMAMILTD    3      FMCG       01-Apr-15   420.8       562.9       252    ₹35,810       +33.8%      +9.9%     
  AJANTPHARM  23     HEALTH     01-Apr-15   737.1       938.5       144    ₹28,998       +27.3%      -1.9%     
  MARUTI      7      AUTO       01-Jun-15   3,524.0     4,018.0     30     ₹14,817       +14.0%      +5.9%     
  WHIRLPOOL   39     CON DUR    02-Feb-15   693.1       738.6       144    ₹6,555        +6.6%       +2.4%     
  INDUSTOWER  29     INFRA      01-Jun-15   327.7       334.8       322    ₹2,304        +2.2%       +7.3%     
  EICHERMOT   13     AUTO       01-Jun-15   1,740.7     1,769.0     60     ₹1,697        +1.6%       -4.1%     
  MARICO      30     FMCG       01-Jun-15   190.3       184.9       555    ₹-3,037       -2.9%       -0.3%     

  AFTER: Invested ₹2,139,287 | Cash ₹90,201 | Total ₹2,229,488 | Positions 18/20 | Slot ₹111,527

========================================================================
  REBALANCE #05  —  01 Oct 2015
  NAV: ₹2,138,248  |  Slot: ₹106,912  |  Cash: ₹90,201
========================================================================
  [SECTOR CAP≤4] dropped: TORNTPHARM, ZYDUSLIFE, DRREDDY, LUPIN

  [REGIME OFF] Nifty 200 4,179.2 < EMA200 4,258.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BRITANNIA   3      FMCG       01-Apr-15   946.0       1,364.1     112    ₹46,835       +44.2%      +4.4%     
  ABBOTINDIA  1      HEALTH     03-Aug-15   4,000.7     5,121.9     27     ₹30,274       +28.0%      +5.2%     
  AJANTPHARM  23     HEALTH     01-Apr-15   737.1       913.9       144    ₹25,462       +24.0%      +3.4%     
  BAJFINANCE  25     FIN SVC    02-Feb-15   39.9        49.1        2507   ₹23,196       +23.2%      +1.3%     
  MARUTI      6      AUTO       01-Jun-15   3,524.0     4,182.0     30     ₹19,738       +18.7%      +3.1%     
  SRF         66     MFG        02-Feb-15   181.9       215.0       549    ₹18,179       +18.2%      -1.2%     
  EMAMILTD    35     FMCG       01-Apr-15   420.8       496.1       252    ₹18,982       +17.9%      +0.6%     
  IBULHSGFIN  5      FIN SVC    03-Aug-15   498.5       524.9       223    ₹5,886        +5.3%       +7.0%     
  OFSS        45     IT         03-Aug-15   2,481.5     2,482.2     44     ₹30           +0.0%       +2.3%     
  INDUSINDBK  18     PVT BNK    03-Aug-15   915.6       904.1       121    ₹-1,389       -1.3%       +5.7%     
  RAJESHEXPO  2      CON DUR    03-Aug-15   561.3       548.4       198    ₹-2,541       -2.3%       +12.6%    
  EICHERMOT   65     AUTO       01-Jun-15   1,740.7     1,695.1     60     ₹-2,734       -2.6%       +0.2%     
  WHIRLPOOL   55     CON DUR    02-Feb-15   693.1       664.0       144    ₹-4,190       -4.2%       +4.3%     
  APLLTD      24     HEALTH     03-Aug-15   657.5       626.0       169    ₹-5,325       -4.8%       +2.5%     
  BAJAJFINSV  19     FIN SVC    03-Aug-15   189.4       175.2       588    ₹-8,328       -7.5%       -1.1%     
  MARICO      69     FMCG       01-Jun-15   190.3       170.9       555    ₹-10,771      -10.2%      -0.5%     
  HINDPETRO   31     OIL&GAS    03-Aug-15   81.6        71.8        1366   ₹-13,318      -11.9%      -1.0%     
  INDUSTOWER  102    INFRA      01-Jun-15   327.7       280.3       322    ₹-15,261      -14.5% ⚠    +1.8%     
  ⚠  WAZ < 0 (momentum below universe mean): INDUSTOWER

  AFTER: Invested ₹2,048,048 | Cash ₹90,201 | Total ₹2,138,248 | Positions 18/20 | Slot ₹106,912

========================================================================
  REBALANCE #06  —  01 Dec 2015
  NAV: ₹2,136,723  |  Slot: ₹106,836  |  Cash: ₹90,201
========================================================================

  [REGIME OFF] Nifty 200 4,198.9 < EMA200 4,245.7 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BRITANNIA   14     FMCG       01-Apr-15   946.0       1,298.0     112    ₹39,430       +37.2%      -0.8%     
  BAJFINANCE  7      FIN SVC    02-Feb-15   39.9        53.9        2507   ₹35,165       +35.2%      +3.7%     
  SRF         37     MFG        02-Feb-15   181.9       240.6       549    ₹32,266       +32.3%      -0.3%     
  RAJESHEXPO  1      CON DUR    03-Aug-15   561.3       698.5       198    ₹27,162       +24.4%      +4.8%     
  MARUTI      16     AUTO       01-Jun-15   3,524.0     4,159.9     30     ₹19,077       +18.0%      -0.5%     
  ABBOTINDIA  8      HEALTH     03-Aug-15   4,000.7     4,604.4     27     ₹16,302       +15.1%      -1.7%     
  AJANTPHARM  83     HEALTH     01-Apr-15   737.1       814.9       144    ₹11,207       +10.6%      -3.0%     
  BAJAJFINSV  5      FIN SVC    03-Aug-15   189.4       198.0       588    ₹5,034        +4.5%       +2.7%     
  OFSS        58     IT         03-Aug-15   2,481.5     2,470.1     44     ₹-504         -0.5%       +0.6%     
  INDUSINDBK  28     PVT BNK    03-Aug-15   915.6       890.0       121    ₹-3,097       -2.8%       +2.2%     
  IBULHSGFIN  25     FIN SVC    03-Aug-15   498.5       480.1       223    ₹-4,118       -3.7%       +4.9%     
  HINDPETRO   22     OIL&GAS    03-Aug-15   81.6        78.4        1366   ₹-4,369       -3.9%       +6.0%     
  APLLTD      27     HEALTH     03-Aug-15   657.5       629.3       169    ₹-4,767       -4.3%       +3.8%     
  MARICO      38     FMCG       01-Jun-15   190.3       182.2       555    ₹-4,495       -4.3%       +2.7%     
  EMAMILTD    145    FMCG       01-Apr-15   420.8       400.9       252    ₹-5,021       -4.7% ⚠     -4.8%     
  WHIRLPOOL   112    CON DUR    02-Feb-15   693.1       656.8       144    ₹-5,215       -5.2% ⚠     +0.7%     
  EICHERMOT   137    AUTO       01-Jun-15   1,740.7     1,505.2     60     ₹-14,130      -13.5% ⚠    -2.6%     
  INDUSTOWER  86     INFRA      01-Jun-15   327.7       275.7       322    ₹-16,726      -15.9%      +0.8%     
  ⚠  WAZ < 0 (momentum below universe mean): WHIRLPOOL, EICHERMOT, EMAMILTD

  AFTER: Invested ₹2,046,523 | Cash ₹90,201 | Total ₹2,136,723 | Positions 18/20 | Slot ₹106,836

========================================================================
  REBALANCE #07  —  01 Feb 2016
  NAV: ₹2,088,454  |  Slot: ₹104,423  |  Cash: ₹90,201
========================================================================

  [REGIME OFF] Nifty 200 3,969.2 < EMA200 4,175.0 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  3      FIN SVC    02-Feb-15   39.9        57.8        2507   ₹44,796       +44.8%      +2.0%     
  BRITANNIA   36     FMCG       01-Apr-15   946.0       1,223.3     112    ₹31,066       +29.3%      +1.0%     
  RAJESHEXPO  1      CON DUR    03-Aug-15   561.3       720.9       198    ₹31,597       +28.4%      +2.5%     
  SRF         64     MFG        02-Feb-15   181.9       221.6       549    ₹21,788       +21.8%      -1.6%     
  ABBOTINDIA  40     HEALTH     03-Aug-15   4,000.7     4,529.7     27     ₹14,283       +13.2%      -3.2%     
  AJANTPHARM  65     HEALTH     01-Apr-15   737.1       786.7       144    ₹7,135        +6.7%       +4.6%     
  EMAMILTD    44     FMCG       01-Apr-15   420.8       437.8       252    ₹4,268        +4.0%       +3.4%     
  MARUTI      97     AUTO       01-Jun-15   3,524.0     3,604.9     30     ₹2,425        +2.3% ⚠     -6.0%     
  MARICO      6      FMCG       01-Jun-15   190.3       194.1       555    ₹2,069        +2.0%       +2.2%     
  BAJAJFINSV  24     FIN SVC    03-Aug-15   189.4       185.3       588    ₹-2,426       -2.2%       -1.9%     
  IBULHSGFIN  26     FIN SVC    03-Aug-15   498.5       480.1       223    ₹-4,119       -3.7%       +1.6%     
  INDUSINDBK  39     PVT BNK    03-Aug-15   915.6       875.3       121    ₹-4,873       -4.4%       +1.1%     
  OFSS        71     IT         03-Aug-15   2,481.5     2,302.4     44     ₹-7,882       -7.2%       -0.3%     
  WHIRLPOOL   63     CON DUR    02-Feb-15   693.1       636.4       144    ₹-8,162       -8.2%       +4.5%     
  HINDPETRO   13     OIL&GAS    03-Aug-15   81.6        74.6        1366   ₹-9,513       -8.5%       -2.4%     
  EICHERMOT   50     AUTO       01-Jun-15   1,740.7     1,582.7     60     ₹-9,478       -9.1%       +3.8%     
  APLLTD      38     HEALTH     03-Aug-15   657.5       560.0       169    ₹-16,481      -14.8%      -1.1%     
  INDUSTOWER  69     INFRA      01-Jun-15   327.7       260.7       322    ₹-21,563      -20.4%      -3.0%     
  ⚠  WAZ < 0 (momentum below universe mean): MARUTI

  AFTER: Invested ₹1,998,253 | Cash ₹90,201 | Total ₹2,088,454 | Positions 18/20 | Slot ₹104,423

========================================================================
  REBALANCE #08  —  01 Apr 2016
  NAV: ₹2,089,760  |  Slot: ₹104,488  |  Cash: ₹90,201
========================================================================

  [REGIME OFF] Nifty 200 4,043.2 < EMA200 4,072.6 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  3      FIN SVC    02-Feb-15   39.9        66.3        2507   ₹66,346       +66.4%      +4.6%     
  SRF         27     MFG        02-Feb-15   181.9       244.6       549    ₹34,420       +34.5%      +4.6%     
  BRITANNIA   61     FMCG       01-Apr-15   946.0       1,170.3     112    ₹25,124       +23.7%      -1.3%     
  AJANTPHARM  29     HEALTH     01-Apr-15   737.1       847.5       144    ₹15,902       +15.0%      +1.7%     
  MARICO      16     FMCG       01-Jun-15   190.3       210.1       555    ₹10,954       +10.4%      +0.4%     
  RAJESHEXPO  4      CON DUR    03-Aug-15   561.3       615.3       198    ₹10,693       +9.6%       -1.8%     
  ABBOTINDIA  115    HEALTH     03-Aug-15   4,000.7     4,152.0     27     ₹4,086        +3.8% ⚠     -2.0%     
  EICHERMOT   13     AUTO       01-Jun-15   1,740.7     1,794.6     60     ₹3,235        +3.1%       +2.5%     
  INDUSINDBK  46     PVT BNK    03-Aug-15   915.6       907.6       121    ₹-966         -0.9%       +4.3%     
  WHIRLPOOL   39     CON DUR    02-Feb-15   693.1       681.6       144    ₹-1,642       -1.6%       +5.2%     
  MARUTI      146    AUTO       01-Jun-15   3,524.0     3,399.4     30     ₹-3,740       -3.5% ⚠     +1.5%     
  EMAMILTD    106    FMCG       01-Apr-15   420.8       391.6       252    ₹-7,353       -6.9% ⚠     -2.4%     
  HINDPETRO   34     OIL&GAS    03-Aug-15   81.6        75.4        1366   ₹-8,515       -7.6%       +5.7%     
  BAJAJFINSV  52     FIN SVC    03-Aug-15   189.4       173.9       588    ₹-9,147       -8.2%       +3.5%     
  OFSS        60     IT         03-Aug-15   2,481.5     2,225.8     44     ₹-11,250      -10.3%      +2.1%     
  IBULHSGFIN  65     FIN SVC    03-Aug-15   498.5       443.6       223    ₹-12,246      -11.0%      +1.7%     
  INDUSTOWER  91     INFRA      01-Jun-15   327.7       271.9       322    ₹-17,941      -17.0% ⚠    +2.0%     
  APLLTD      75     HEALTH     03-Aug-15   657.5       529.0       169    ₹-21,723      -19.5% ⚠    -3.0%     
  ⚠  WAZ < 0 (momentum below universe mean): APLLTD, INDUSTOWER, EMAMILTD, ABBOTINDIA, MARUTI

  AFTER: Invested ₹1,999,559 | Cash ₹90,201 | Total ₹2,089,760 | Positions 18/20 | Slot ₹104,488

========================================================================
  REBALANCE #09  —  01 Jun 2016
  NAV: ₹2,186,085  |  Slot: ₹109,304  |  Cash: ₹90,201
========================================================================
  [SECTOR CAP≤4] dropped: MUTHOOTFIN, BERGEPAINT

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SRF         33     MFG        02-Feb-15   181.9       246.2       549    ₹35,307       +35.4%    485d  
  BRITANNIA   72     FMCG       01-Apr-15   946.0       1,207.9     112    ₹29,334       +27.7%    427d  
  AJANTPHARM  48     HEALTH     01-Apr-15   737.1       930.8       144    ₹27,887       +26.3%    427d  
  MARICO      57     FMCG       01-Jun-15   190.3       218.4       555    ₹15,555       +14.7%    366d  
  EMAMILTD    —      FMCG       01-Apr-15   420.8       437.7       252    ₹4,257        +4.0%     427d  
  ABBOTINDIA  66     HEALTH     03-Aug-15   4,000.7     4,141.9     27     ₹3,814        +3.5%     303d  
  EICHERMOT   88     AUTO       01-Jun-15   1,740.7     1,744.2     60     ₹211          +0.2%     366d  
  OFSS        —      IT         03-Aug-15   2,481.5     2,199.6     44     ₹-12,404      -11.4%    303d  
  INDUSTOWER  85     INFRA      01-Jun-15   327.7       271.5       322    ₹-18,088      -17.1%    366d  
  APLLTD      101    HEALTH     03-Aug-15   657.5       485.3       169    ₹-29,107      -26.2%    303d  

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BIOCON      1      HEALTH     3.680    0.78   +60.9%    +57.9%    117.0       933    ₹109,202      +10.2%    
  VGUARD      2      CON DUR    3.439    0.58   +46.9%    +66.5%    93.1        1173   ₹109,228      +13.7%    
  RAMCOCEM    3      MFG        2.843    0.72   +62.3%    +35.7%    473.9       230    ₹108,987      +0.7%     
  HDFC        7      PVT BNK    2.255    0.77   +15.7%    +24.7%    265.8       411    ₹109,228      +1.9%     
  HDFCBANK    8      PVT BNK    2.255    0.77   +15.7%    +24.7%    265.8       411    ₹109,228      +1.9%     
  SHRIRAMFIN  9      FIN SVC    2.116    0.99   +41.9%    +46.0%    197.6       553    ₹109,281      +3.9%     
  PIDILITIND  11     MFG        1.957    0.60   +31.7%    +20.3%    342.9       318    ₹109,032      +9.8%     
  HAVELLS     12     CON DUR    1.943    0.96   +36.4%    +31.7%    331.1       330    ₹109,253      -0.7%     
  BPCL        13     OIL&GAS    1.787    0.85   +30.4%    +29.8%    89.0        1228   ₹109,261      +3.9%     
  ASIANPAINT  15     CONSUMP    1.770    0.90   +35.2%    +18.6%    933.9       117    ₹109,262      +6.8%     
  PETRONET    16     OIL&GAS    1.750    0.85   +47.7%    +15.4%    92.8        1177   ₹109,238      +0.7%     

  HOLDS (8)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  4      FIN SVC    02-Feb-15   39.9        73.7        2507   ₹84,845       +84.9%      +1.7%     
  INDUSINDBK  5      PVT BNK    03-Aug-15   915.6       1,042.1     121    ₹15,307       +13.8%      +2.9%     
  WHIRLPOOL   38     CON DUR    02-Feb-15   693.1       748.2       144    ₹7,945        +8.0%       +1.8%     
  MARUTI      27     AUTO       01-Jun-15   3,524.0     3,803.1     30     ₹8,371        +7.9%       +4.7%     
  HINDPETRO   6      OIL&GAS    03-Aug-15   81.6        86.0        1366   ₹6,003        +5.4%       +5.4%     
  IBULHSGFIN  44     FIN SVC    03-Aug-15   498.5       498.5       223    ₹-13          -0.0%       +3.2%     
  RAJESHEXPO  19     CON DUR    03-Aug-15   561.3       558.5       198    ₹-544         -0.5%       -0.4%     
  BAJAJFINSV  43     FIN SVC    03-Aug-15   189.4       179.0       588    ₹-6,115       -5.5%       -1.6%     

  AFTER: Invested ₹2,178,428 | Cash ₹6,231 | Total ₹2,184,659 | Positions 19/20 | Slot ₹109,304

========================================================================
  REBALANCE #10  —  01 Aug 2016
  NAV: ₹2,560,164  |  Slot: ₹128,008  |  Cash: ₹6,231
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  WHIRLPOOL   37     CON DUR    02-Feb-15   693.1       833.3       144    ₹20,198       +20.2%    546d  
  HDFC        26     PVT BNK    01-Jun-16   265.8       283.3       411    ₹7,189        +6.6%     61d   
  HDFCBANK    27     PVT BNK    01-Jun-16   265.8       283.3       411    ₹7,189        +6.6%     61d   
  IBULHSGFIN  55     FIN SVC    03-Aug-15   498.5       523.4       223    ₹5,543        +5.0%     364d  
  RAJESHEXPO  108    CON DUR    03-Aug-15   561.3       436.5       198    ₹-24,703      -22.2%    364d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MUTHOOTFIN  3      FIN SVC    3.800    0.64   +84.3%    +81.3%    298.8       428    ₹127,900      +20.6%    
  SHREECEM    10     INFRA      2.173    0.84   +51.9%    +31.1%    15,652.0    8      ₹125,216      +3.3%     
  BERGEPAINT  11     CON DUR    2.021    0.83   +53.4%    +24.9%    186.1       687    ₹127,850      +3.1%     
  POWERGRID   13     ENERGY     1.904    0.65   +26.1%    +22.0%    63.0        2032   ₹128,006      +5.6%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  1      FIN SVC    02-Feb-15   39.9        108.4       2507   ₹171,743      +171.8%     +22.0%    
  HINDPETRO   6      OIL&GAS    03-Aug-15   81.6        121.9       1366   ₹55,114       +49.4%      +12.3%    
  BAJAJFINSV  4      FIN SVC    03-Aug-15   189.4       281.5       588    ₹54,176       +48.6%      +12.3%    
  MARUTI      22     AUTO       01-Jun-15   3,524.0     4,443.0     30     ₹27,569       +26.1%      +8.6%     
  INDUSINDBK  21     PVT BNK    03-Aug-15   915.6       1,138.1     121    ₹26,933       +24.3%      +5.1%     
  VGUARD      2      CON DUR    01-Jun-16   93.1        113.9       1173   ₹24,322       +22.3%      +11.9%    
  BPCL        17     OIL&GAS    01-Jun-16   89.0        108.8       1228   ₹24,350       +22.3%      +4.5%     
  BIOCON      5      HEALTH     01-Jun-16   117.0       135.3       933    ₹17,048       +15.6%      +7.7%     
  RAMCOCEM    8      MFG        01-Jun-16   473.9       544.0       230    ₹16,127       +14.8%      +0.2%     
  PETRONET    13     OIL&GAS    01-Jun-16   92.8        106.3       1177   ₹15,900       +14.6%      +5.9%     
  HAVELLS     18     CON DUR    01-Jun-16   331.1       378.3       330    ₹15,583       +14.3%      +8.5%     
  SHRIRAMFIN  19     FIN SVC    01-Jun-16   197.6       221.8       553    ₹13,379       +12.2%      +5.8%     
  ASIANPAINT  7      CONSUMP    01-Jun-16   933.9       1,030.0     117    ₹11,248       +10.3%      +6.5%     
  PIDILITIND  9      MFG        01-Jun-16   342.9       350.4       318    ₹2,393        +2.2%       +1.2%     

  AFTER: Invested ₹2,506,927 | Cash ₹52,633 | Total ₹2,559,560 | Positions 18/20 | Slot ₹128,008

========================================================================
  REBALANCE #11  —  03 Oct 2016
  NAV: ₹2,715,573  |  Slot: ₹135,779  |  Cash: ₹52,633
========================================================================
  [SECTOR CAP≤4] dropped: IGL

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  INDUSINDBK  41     PVT BNK    03-Aug-15   915.6       1,168.8     121    ₹30,644       +27.7%    427d  
  SHRIRAMFIN  69     FIN SVC    01-Jun-16   197.6       208.2       553    ₹5,879        +5.4%     124d  
  POWERGRID   44     ENERGY     01-Aug-16   63.0        64.9        2032   ₹3,968        +3.1%     63d   
  PIDILITIND  61     MFG        01-Jun-16   342.9       346.3       318    ₹1,081        +1.0%     124d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HINDZINC    1      METAL      2.831    1.07   +120.3%   +40.9%    113.8       1193   ₹135,739      +11.7%    
  MRF         4      MFG        2.581    0.93   +25.7%    +57.3%    51,226.4    2      ₹102,453      +16.3%    
  IOC         6      OIL&GAS    2.458    0.91   +60.9%    +38.9%    53.5        2536   ₹135,730      +4.8%     
  CHOLAFIN    8      FIN SVC    2.372    0.35   +96.7%    +24.8%    228.4       594    ₹135,656      +5.2%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  3      FIN SVC    02-Feb-15   39.9        104.9       2507   ₹163,047      +163.1%     +0.1%     
  BAJAJFINSV  7      FIN SVC    03-Aug-15   189.4       318.0       588    ₹75,590       +67.9%      +3.5%     
  HINDPETRO   15     OIL&GAS    03-Aug-15   81.6        124.8       1366   ₹58,990       +52.9%      +4.3%     
  MARUTI      19     AUTO       01-Jun-15   3,524.0     5,223.0     30     ₹50,968       +48.2%      +4.8%     
  VGUARD      4      CON DUR    01-Jun-16   93.1        127.4       1173   ₹40,166       +36.8%      +0.4%     
  BIOCON      1      HEALTH     01-Jun-16   117.0       157.9       933    ₹38,130       +34.9%      +3.1%     
  PETRONET    10     OIL&GAS    01-Jun-16   92.8        121.2       1177   ₹33,362       +30.5%      +3.5%     
  BPCL        33     OIL&GAS    01-Jun-16   89.0        114.9       1228   ₹31,799       +29.1%      +4.5%     
  RAMCOCEM    14     MFG        01-Jun-16   473.9       596.8       230    ₹28,284       +26.0%      +4.3%     
  HAVELLS     21     CON DUR    01-Jun-16   331.1       403.7       330    ₹23,955       +21.9%      +4.1%     
  ASIANPAINT  24     CONSUMP    01-Jun-16   933.9       1,096.3     117    ₹19,000       +17.4%      +1.9%     
  BERGEPAINT  13     CON DUR    01-Aug-16   186.1       209.6       687    ₹16,128       +12.6%      +1.5%     
  SHREECEM    32     INFRA      01-Aug-16   15,652.0    17,211.1    8      ₹12,473       +10.0%      +5.0%     
  MUTHOOTFIN  16     FIN SVC    01-Aug-16   298.8       295.2       428    ₹-1,565       -1.2%       -1.6%     

  AFTER: Invested ₹2,673,843 | Cash ₹41,124 | Total ₹2,714,968 | Positions 18/20 | Slot ₹135,779

========================================================================
  REBALANCE #12  —  01 Dec 2016
  NAV: ₹2,507,380  |  Slot: ₹125,369  |  Cash: ₹41,124
========================================================================
  [SECTOR CAP≤4] dropped: IGL

  [REGIME OFF] Nifty 200 4,386.1 < EMA200 4,400.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  30     FIN SVC    02-Feb-15   39.9        88.3        2507   ₹121,294      +121.3%     -0.3%     
  HINDPETRO   8      OIL&GAS    03-Aug-15   81.6        129.0       1366   ₹64,704       +58.1%      -1.9%     
  BAJAJFINSV  18     FIN SVC    03-Aug-15   189.4       292.6       588    ₹60,646       +54.5%      -2.1%     
  PETRONET    7      OIL&GAS    01-Jun-16   92.8        130.9       1177   ₹44,853       +41.1%      +2.5%     
  MARUTI      27     AUTO       01-Jun-15   3,524.0     4,828.7     30     ₹39,139       +37.0%      +1.7%     
  BIOCON      4      HEALTH     01-Jun-16   117.0       151.6       933    ₹32,200       +29.5%      +3.4%     
  BPCL        17     OIL&GAS    01-Jun-16   89.0        115.1       1228   ₹32,057       +29.3%      -2.4%     
  VGUARD      12     CON DUR    01-Jun-16   93.1        116.3       1173   ₹27,165       +24.9%      -6.0%     
  RAMCOCEM    10     MFG        01-Jun-16   473.9       581.0       230    ₹24,637       +22.6%      +2.6%     
  HINDZINC    1      METAL      03-Oct-16   113.8       123.8       1193   ₹12,013       +8.9%       +4.4%     
  BERGEPAINT  32     CON DUR    01-Aug-16   186.1       186.0       687    ₹-82          -0.1%       +4.3%     
  IOC         15     OIL&GAS    03-Oct-16   53.5        52.8        2536   ₹-1,799       -1.3%       -1.5%     
  HAVELLS     83     CON DUR    01-Jun-16   331.1       313.5       330    ₹-5,810       -5.3% ⚠     -4.2%     
  MRF         5      MFG        03-Oct-16   51,226.4    48,256.4    2      ₹-5,940       -5.8%       -0.3%     
  ASIANPAINT  84     CONSUMP    01-Jun-16   933.9       865.1       117    ₹-8,041       -7.4% ⚠     -3.7%     
  SHREECEM    44     INFRA      01-Aug-16   15,652.0    14,419.1    8      ₹-9,863       -7.9% ⚠     -2.8%     
  MUTHOOTFIN  24     FIN SVC    01-Aug-16   298.8       253.8       428    ₹-19,269      -15.1%      -3.8%     
  CHOLAFIN    42     FIN SVC    03-Oct-16   228.4       186.0       594    ₹-25,166      -18.6% ⚠    -5.0%     
  ⚠  WAZ < 0 (momentum below universe mean): CHOLAFIN, SHREECEM, HAVELLS, ASIANPAINT

  AFTER: Invested ₹2,466,256 | Cash ₹41,124 | Total ₹2,507,380 | Positions 18/20 | Slot ₹125,369

========================================================================
  REBALANCE #13  —  01 Feb 2017
  NAV: ₹2,818,509  |  Slot: ₹140,925  |  Cash: ₹41,124
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJFINANCE  21     FIN SVC    02-Feb-15   39.9        102.9       2507   ₹158,073      +158.1%   730d  
  HINDPETRO   2      OIL&GAS    03-Aug-15   81.6        155.5       1366   ₹100,903      +90.5%    548d  
  SHREECEM    33     INFRA      01-Aug-16   15,652.0    15,330.4    8      ₹-2,573       -2.1%     184d  
  ASIANPAINT  78     CONSUMP    01-Jun-16   933.9       912.9       117    ₹-2,453       -2.2%     245d  
  MUTHOOTFIN  35     FIN SVC    01-Aug-16   298.8       272.1       428    ₹-11,424      -8.9%     184d  
  BERGEPAINT  97     CON DUR    01-Aug-16   186.1       169.1       687    ₹-11,707      -9.2%     184d  
  CHOLAFIN    50     FIN SVC    03-Oct-16   228.4       198.7       594    ₹-17,602      -13.0%    121d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  VAKRANGEE   3      IT         2.866    0.52   +83.6%    +24.3%    137.7       1023   ₹140,825      +4.5%     
  POWERGRID   5      ENERGY     2.849    0.74   +56.8%    +17.5%    73.8        1910   ₹140,892      +3.7%     
  IGL         7      OIL&GAS    2.571    0.78   +70.6%    +12.3%    84.2        1674   ₹140,900      +3.1%     
  BEL         8      DEFENCE    2.320    1.06   +33.2%    +20.5%    40.0        3523   ₹140,892      +4.1%     
  HDFC        10     PVT BNK    1.914    0.71   +27.8%    +4.8%     297.7       473    ₹140,816      +4.3%     
  HDFCBANK    11     PVT BNK    1.914    0.71   +27.8%    +4.8%     297.7       473    ₹140,816      +4.3%     
  NBCC        12     INFRA      1.900    1.08   +51.2%    +14.7%    55.5        2536   ₹140,873      +4.2%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJAJFINSV  24     FIN SVC    03-Aug-15   189.4       331.3       588    ₹83,418       +74.9%      +7.6%     
  MARUTI      19     AUTO       01-Jun-15   3,524.0     5,681.0     30     ₹64,709       +61.2%      +7.8%     
  VGUARD      11     CON DUR    01-Jun-16   93.1        141.1       1173   ₹56,317       +51.6%      +12.8%    
  RAMCOCEM    7      MFG        01-Jun-16   473.9       693.0       230    ₹50,412       +46.3%      +13.1%    
  PETRONET    23     OIL&GAS    01-Jun-16   92.8        132.7       1177   ₹46,974       +43.0%      +3.6%     
  BPCL        18     OIL&GAS    01-Jun-16   89.0        126.8       1228   ₹46,405       +42.5%      +2.8%     
  BIOCON      6      HEALTH     01-Jun-16   117.0       166.6       933    ₹46,253       +42.4%      +2.4%     
  IOC         3      OIL&GAS    03-Oct-16   53.5        66.6        2536   ₹33,095       +24.4%      +5.1%     
  HINDZINC    1      METAL      03-Oct-16   113.8       138.1       1193   ₹29,045       +21.4%      +6.3%     
  HAVELLS     30     CON DUR    01-Jun-16   331.1       399.7       330    ₹22,657       +20.7%      +8.7%     
  MRF         25     MFG        03-Oct-16   51,226.4    51,910.7    2      ₹1,369        +1.3%       +0.7%     

  AFTER: Invested ₹2,712,854 | Cash ₹104,484 | Total ₹2,817,339 | Positions 18/20 | Slot ₹140,925

========================================================================
  REBALANCE #14  —  03 Apr 2017
  NAV: ₹2,988,180  |  Slot: ₹149,409  |  Cash: ₹104,484
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MARUTI      37     AUTO       01-Jun-15   3,524.0     5,582.5     30     ₹61,752       +58.4%    672d  
  RAMCOCEM    39     MFG        01-Jun-16   473.9       654.3       230    ₹41,500       +38.1%    306d  
  BPCL        60     OIL&GAS    01-Jun-16   89.0        122.4       1228   ₹41,009       +37.5%    306d  
  BEL         45     DEFENCE    01-Feb-17   40.0        41.1        3523   ₹3,847        +2.7%     61d   
  POWERGRID   46     ENERGY     01-Feb-17   73.8        70.7        1910   ₹-5,902       -4.2%     61d   
  NBCC        69     INFRA      01-Feb-17   55.5        52.7        2536   ₹-7,183       -5.1%     61d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  DALMIABHA   3      MFG        2.586    0.14   +147.5%   +45.8%    1,969.1     75     ₹147,681      +2.4%     
  IBULHSGFIN  4      FIN SVC    2.582    0.15   +63.1%    +54.9%    707.9       211    ₹149,360      +6.5%     
  GUJGASLTD   6      OIL&GAS    2.363    0.53   +44.0%    +44.8%    142.3       1049   ₹149,321      +4.2%     
  DHFL        7      FIN SVC    2.348    0.24   +86.6%    +53.3%    366.2       408    ₹149,392      +5.7%     
  NATCOPHARM  8      HEALTH     2.319    0.56   +85.9%    +50.4%    803.3       186    ₹149,408      +8.1%     
  KARURVYSYA  9      PVT BNK    2.301    0.62   +36.7%    +42.4%    73.3        2037   ₹149,378      +10.0%    

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJAJFINSV  3      FIN SVC    03-Aug-15   189.4       407.9       588    ₹128,473      +115.4%     +2.7%     
  VGUARD      1      CON DUR    01-Jun-16   93.1        168.4       1173   ₹88,336       +80.9%      +2.5%     
  BIOCON      9      HEALTH     01-Jun-16   117.0       185.3       933    ₹63,675       +58.3%      +0.9%     
  PETRONET    44     OIL&GAS    01-Jun-16   92.8        142.1       1177   ₹57,960       +53.1%      +4.0%     
  HAVELLS     26     CON DUR    01-Jun-16   331.1       438.9       330    ₹35,570       +32.6%      +4.9%     
  IOC         19     OIL&GAS    03-Oct-16   53.5        70.8        2536   ₹43,856       +32.3%      +2.1%     
  HINDZINC    21     METAL      03-Oct-16   113.8       143.4       1193   ₹35,366       +26.1%      +2.0%     
  MRF         32     MFG        03-Oct-16   51,226.4    60,214.9    2      ₹17,977       +17.5%      +6.5%     
  HDFC        17     PVT BNK    01-Feb-17   297.7       326.8       473    ₹13,751       +9.8%       +1.1%     
  HDFCBANK    18     PVT BNK    01-Feb-17   297.7       326.8       473    ₹13,751       +9.8%       +1.1%     
  VAKRANGEE   27     IT         01-Feb-17   137.7       147.3       1023   ₹9,890        +7.0%       +2.1%     
  IGL         28     OIL&GAS    01-Feb-17   84.2        88.9        1674   ₹7,866        +5.6%       -0.4%     

  AFTER: Invested ₹2,896,583 | Cash ₹90,534 | Total ₹2,987,117 | Positions 18/20 | Slot ₹149,409

========================================================================
  REBALANCE #15  —  01 Jun 2017
  NAV: ₹3,159,832  |  Slot: ₹157,992  |  Cash: ₹90,534
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HAVELLS     —      CON DUR    01-Jun-16   331.1       454.1       330    ₹40,606       +37.2%    365d  
  BIOCON      102    HEALTH     01-Jun-16   117.0       156.7       933    ₹37,030       +33.9%    365d  
  MRF         —      MFG        03-Oct-16   51,226.4    67,346.4    2      ₹32,240       +31.5%    241d  
  HINDZINC    97     METAL      03-Oct-16   113.8       117.4       1193   ₹4,325        +3.2%     241d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TVSMOTOR    5      AUTO       2.410    0.90   +81.3%    +24.9%    508.7       310    ₹157,690      +2.1%     
  HINDUNILVR  6      FMCG       2.353    0.54   +32.9%    +26.5%    940.4       167    ₹157,055      +7.3%     
  GODREJIND   9      CONSUMP    2.090    0.98   +76.6%    +22.6%    610.4       258    ₹157,493      +6.3%     
  KOTAKBANK   13     PVT BNK    1.970    0.76   +35.1%    +20.0%    191.4       825    ₹157,912      +1.9%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJAJFINSV  21     FIN SVC    03-Aug-15   189.4       418.6       588    ₹134,756      +121.0%     -0.1%     
  VGUARD      23     CON DUR    01-Jun-16   93.1        176.8       1173   ₹98,115       +89.8%      -3.8%     
  PETRONET    46     OIL&GAS    01-Jun-16   92.8        152.0       1177   ₹69,693       +63.8%      +1.5%     
  IOC         16     OIL&GAS    03-Oct-16   53.5        76.9        2536   ₹59,263       +43.7%      -3.3%     
  VAKRANGEE   2      IT         01-Feb-17   137.7       174.1       1023   ₹37,295       +26.5%      +9.6%     
  HDFC        6      PVT BNK    01-Feb-17   297.7       371.3       473    ₹34,824       +24.7%      +2.9%     
  HDFCBANK    7      PVT BNK    01-Feb-17   297.7       371.3       473    ₹34,824       +24.7%      +2.9%     
  DALMIABHA   5      MFG        03-Apr-17   1,969.1     2,419.4     75     ₹33,772       +22.9%      +1.8%     
  IBULHSGFIN  14     FIN SVC    03-Apr-17   707.9       817.2       211    ₹23,080       +15.5%      +6.4%     
  DHFL        11     FIN SVC    03-Apr-17   366.2       408.8       408    ₹17,392       +11.6%      +0.4%     
  IGL         26     OIL&GAS    01-Feb-17   84.2        93.2        1674   ₹15,087       +10.7%      +3.7%     
  NATCOPHARM  17     HEALTH     03-Apr-17   803.3       877.3       186    ₹13,761       +9.2%       +3.1%     
  KARURVYSYA  44     PVT BNK    03-Apr-17   73.3        74.6        2037   ₹2,630        +1.8%       +0.9%     
  GUJGASLTD   25     OIL&GAS    03-Apr-17   142.3       142.8       1049   ₹493          +0.3%       -1.6%     

  AFTER: Invested ₹3,128,601 | Cash ₹30,483 | Total ₹3,159,084 | Positions 18/20 | Slot ₹157,992

========================================================================
  REBALANCE #16  —  01 Aug 2017
  NAV: ₹3,394,134  |  Slot: ₹169,707  |  Cash: ₹30,483
========================================================================
  [SECTOR CAP≤4] dropped: INDUSINDBK

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJAJFINSV  19     FIN SVC    03-Aug-15   189.4       504.6       588    ₹185,308      +166.4%   729d  
  IOC         91     OIL&GAS    03-Oct-16   53.5        68.8        2536   ₹38,767       +28.6%    302d  
  GUJGASLTD   84     OIL&GAS    03-Apr-17   142.3       142.6       1049   ₹306          +0.2%     120d  

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RAJESHEXPO  6      CON DUR    2.644    0.76   +64.8%    +16.7%    714.7       237    ₹169,378      +4.4%     
  RELIANCE    8      OIL&GAS    2.238    0.71   +58.0%    +18.0%    352.5       481    ₹169,577      +3.9%     
  PGHH        10     FMCG       1.994    0.31   +33.7%    +15.5%    7,228.0     23     ₹166,244      +1.0%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VGUARD      49     CON DUR    01-Jun-16   93.1        174.9       1173   ₹95,967       +87.9%      -1.1%     
  PETRONET    60     OIL&GAS    01-Jun-16   92.8        145.3       1177   ₹61,736       +56.5% ⚠    +1.2%     
  VAKRANGEE   1      IT         01-Feb-17   137.7       199.9       1023   ₹63,647       +45.2%      +1.3%     
  HDFC        3      PVT BNK    01-Feb-17   297.7       412.5       473    ₹54,295       +38.6%      +4.2%     
  HDFCBANK    4      PVT BNK    01-Feb-17   297.7       412.5       473    ₹54,295       +38.6%      +4.2%     
  DALMIABHA   28     MFG        03-Apr-17   1,969.1     2,588.4     75     ₹46,448       +31.5%      -1.1%     
  DHFL        25     FIN SVC    03-Apr-17   366.2       459.5       408    ₹38,098       +25.5%      +2.8%     
  IGL         10     OIL&GAS    01-Feb-17   84.2        104.2       1674   ₹33,568       +23.8%      +4.8%     
  IBULHSGFIN  22     FIN SVC    03-Apr-17   707.9       876.7       211    ₹35,614       +23.8%      +6.3%     
  KARURVYSYA  26     PVT BNK    03-Apr-17   73.3        90.0        2037   ₹34,017       +22.8%      +1.0%     
  NATCOPHARM  33     HEALTH     03-Apr-17   803.3       899.3       186    ₹17,864       +12.0%      -1.8%     
  TVSMOTOR    5      AUTO       01-Jun-17   508.7       568.4       310    ₹18,509       +11.7%      +4.4%     
  HINDUNILVR  7      FMCG       01-Jun-17   940.4       1,017.0     167    ₹12,782       +8.1%       +2.8%     
  GODREJIND   24     CONSUMP    01-Jun-17   610.4       649.5       258    ₹10,072       +6.4%       -1.3%     
  KOTAKBANK   27     PVT BNK    01-Jun-17   191.4       202.0       825    ₹8,741        +5.5%       +2.6%     
  ⚠  WAZ < 0 (momentum below universe mean): PETRONET

  AFTER: Invested ₹3,248,043 | Cash ₹145,491 | Total ₹3,393,534 | Positions 18/20 | Slot ₹169,707

========================================================================
  REBALANCE #17  —  03 Oct 2017
  NAV: ₹3,538,029  |  Slot: ₹176,901  |  Cash: ₹145,491
========================================================================
  [SECTOR CAP≤4] dropped: MGL, GUJGASLTD

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VGUARD      50     CON DUR    01-Jun-16   93.1        179.9       1173   ₹101,755      +93.2%    489d  
  KARURVYSYA  46     PVT BNK    03-Apr-17   73.3        91.7        2037   ₹37,480       +25.1%    183d  
  GODREJIND   74     CONSUMP    01-Jun-17   610.4       588.7       258    ₹-5,609       -3.6%     124d  
  NATCOPHARM  105    HEALTH     03-Apr-17   803.3       738.5       186    ₹-12,051      -8.1%     183d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HONAUT      3      MFG        3.000    0.69   +69.3%    +31.0%    15,425.8    11     ₹169,684      +3.6%     
  GAIL        9      OIL&GAS    2.202    0.82   +58.0%    +19.5%    76.7        2306   ₹176,899      +8.2%     
  BRITANNIA   12     FMCG       1.983    0.89   +27.7%    +17.8%    1,920.1     92     ₹176,646      +0.9%     
  INDUSINDBK  13     PVT BNK    1.965    1.07   +41.4%    +14.0%    1,611.1     109    ₹175,606      -0.2%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  PETRONET    43     OIL&GAS    01-Jun-16   92.8        161.7       1177   ₹81,103       +74.2%      +1.9%     
  VAKRANGEE   6      IT         01-Feb-17   137.7       222.4       1023   ₹86,738       +61.6%      +0.6%     
  IGL         1      OIL&GAS    01-Feb-17   84.2        129.7       1674   ₹76,196       +54.1%      +4.3%     
  DHFL        15     FIN SVC    03-Apr-17   366.2       543.3       408    ₹72,278       +48.4%      +0.8%     
  HDFC        10     PVT BNK    01-Feb-17   297.7       415.2       473    ₹55,560       +39.5%      +0.2%     
  HDFCBANK    11     PVT BNK    01-Feb-17   297.7       415.2       473    ₹55,560       +39.5%      +0.2%     
  DALMIABHA   42     MFG        03-Apr-17   1,969.1     2,715.0     75     ₹55,944       +37.9%      +1.4%     
  IBULHSGFIN  25     FIN SVC    03-Apr-17   707.9       890.5       211    ₹38,537       +25.8%      -0.5%     
  TVSMOTOR    9      AUTO       01-Jun-17   508.7       622.8       310    ₹35,379       +22.4%      +2.2%     
  RAJESHEXPO  3      CON DUR    01-Aug-17   714.7       815.3       237    ₹23,853       +14.1%      +6.7%     
  HINDUNILVR  27     FMCG       01-Jun-17   940.4       1,027.7     167    ₹14,576       +9.3%       -2.7%     
  KOTAKBANK   38     PVT BNK    01-Jun-17   191.4       200.3       825    ₹7,305        +4.6%       +0.3%     
  PGHH        37     FMCG       01-Aug-17   7,228.0     7,500.9     23     ₹6,276        +3.8%       +0.8%     
  RELIANCE    18     OIL&GAS    01-Aug-17   352.5       351.0       481    ₹-730         -0.4%       -1.7%     

  AFTER: Invested ₹3,404,291 | Cash ₹132,908 | Total ₹3,537,199 | Positions 18/20 | Slot ₹176,901

========================================================================
  REBALANCE #18  —  01 Dec 2017
  NAV: ₹3,811,146  |  Slot: ₹190,557  |  Cash: ₹132,908
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PETRONET    52     OIL&GAS    01-Jun-16   92.8        169.2       1177   ₹89,966       +82.4%    548d  
  IBULHSGFIN  72     FIN SVC    03-Apr-17   707.9       831.7       211    ₹26,120       +17.5%    242d  
  KOTAKBANK   61     PVT BNK    01-Jun-17   191.4       200.0       825    ₹7,075        +4.5%     183d  
  INDUSINDBK  48     PVT BNK    03-Oct-17   1,611.1     1,580.8     109    ₹-3,304       -1.9%     59d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BBTC        2      FMCG       3.334    1.04   +197.5%   +48.2%    1,478.3     128    ₹189,228      -0.9%     
  FRETAIL     3      CONSUMP    2.506    1.17   +348.9%   +1.6%     544.8       349    ₹190,118      +2.1%     
  BALKRISIND  7      MFG        2.230    0.57   +130.1%   +28.2%    978.3       194    ₹189,795      +4.4%     
  MARUTI      10     AUTO       2.095    1.06   +70.9%    +10.2%    7,994.1     23     ₹183,864      +2.5%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VAKRANGEE   1      IT         01-Feb-17   137.7       320.5       1023   ₹187,031      +132.8%     +7.4%     
  IGL         6      OIL&GAS    01-Feb-17   84.2        139.5       1674   ₹92,667       +65.8%      +1.1%     
  DHFL        10     FIN SVC    03-Apr-17   366.2       595.4       408    ₹93,530       +62.6%      -2.9%     
  DALMIABHA   26     MFG        03-Apr-17   1,969.1     3,120.1     75     ₹86,330       +58.5%      +2.4%     
  HDFC        15     PVT BNK    01-Feb-17   297.7       424.2       473    ₹59,832       +42.5%      +0.4%     
  HDFCBANK    16     PVT BNK    01-Feb-17   297.7       424.2       473    ₹59,832       +42.5%      +0.4%     
  TVSMOTOR    4      AUTO       01-Jun-17   508.7       692.6       310    ₹57,006       +36.2%      +0.9%     
  PGHH        9      FMCG       01-Aug-17   7,228.0     8,492.5     23     ₹29,083       +17.5%      +4.5%     
  HINDUNILVR  37     FMCG       01-Jun-17   940.4       1,090.3     167    ₹25,033       +15.9%      -1.0%     
  RELIANCE    20     OIL&GAS    01-Aug-17   352.5       400.2       481    ₹22,905       +13.5%      -1.3%     
  BRITANNIA   13     FMCG       03-Oct-17   1,920.1     2,126.2     92     ₹18,963       +10.7%      +1.0%     
  HONAUT      5      MFG        03-Oct-17   15,425.8    16,639.1    11     ₹13,346       +7.9%       +1.7%     
  RAJESHEXPO  32     CON DUR    01-Aug-17   714.7       751.9       237    ₹8,817        +5.2%       -1.5%     
  GAIL        28     OIL&GAS    03-Oct-17   76.7        80.3        2306   ₹8,288        +4.7%       -0.8%     

  AFTER: Invested ₹3,719,270 | Cash ₹90,981 | Total ₹3,810,252 | Positions 18/20 | Slot ₹190,557

========================================================================
  REBALANCE #19  —  01 Feb 2018
  NAV: ₹3,796,746  |  Slot: ₹189,837  |  Cash: ₹90,981
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VAKRANGEE   41     IT         01-Feb-17   137.7       262.4       1023   ₹127,605      +90.6%    365d  
  IGL         56     OIL&GAS    01-Feb-17   84.2        133.1       1674   ₹81,927       +58.1%    365d  
  DHFL        57     FIN SVC    03-Apr-17   366.2       564.2       408    ₹80,802       +54.1%    304d  
  DALMIABHA   44     MFG        03-Apr-17   1,969.1     2,972.0     75     ₹75,219       +50.9%    304d  
  TVSMOTOR    40     AUTO       01-Jun-17   508.7       641.5       310    ₹41,170       +26.1%    245d  
  GAIL        54     OIL&GAS    03-Oct-17   76.7        87.1        2306   ₹23,856       +13.5%    121d  
  FRETAIL     11     CONSUMP    01-Dec-17   544.8       542.5       349    ₹-768         -0.4%     62d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BIOCON      1      HEALTH     2.753    1.12   +82.6%    +62.8%    305.0       622    ₹189,684      +6.0%     
  MPHASIS     6      IT         2.408    0.30   +61.5%    +26.3%    722.3       262    ₹189,241      +9.6%     
  IBULHSGFIN  8      FIN SVC    2.083    0.37   +87.9%    +12.5%    1,010.8     187    ₹189,016      +5.3%     
  ABBOTINDIA  10     HEALTH     1.965    0.14   +26.5%    +28.4%    5,020.8     37     ₹185,770      +1.3%     
  M&M         11     AUTO       1.957    0.81   +28.8%    +19.6%    734.5       258    ₹189,490      +4.9%     
  TCS         12     IT         1.946    0.23   +35.7%    +19.8%    1,271.5     149    ₹189,459      +5.7%     
  KOTAKBANK   13     PVT BNK    1.941    0.58   +42.9%    +9.9%     223.0       851    ₹189,793      +5.3%     
  OFSS        14     IT         1.921    0.57   +31.3%    +19.1%    2,861.2     66     ₹188,841      +1.7%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HDFC        4      PVT BNK    01-Feb-17   297.7       457.0       473    ₹75,351       +53.5%      +2.7%     
  HDFCBANK    5      PVT BNK    01-Feb-17   297.7       457.0       473    ₹75,351       +53.5%      +2.7%     
  HINDUNILVR  10     FMCG       01-Jun-17   940.4       1,195.6     167    ₹42,616       +27.1%      +0.3%     
  RELIANCE    27     OIL&GAS    01-Aug-17   352.5       415.0       481    ₹30,043       +17.7%      -0.3%     
  PGHH        26     FMCG       01-Aug-17   7,228.0     8,267.8     23     ₹23,916       +14.4%      -0.6%     
  RAJESHEXPO  13     CON DUR    01-Aug-17   714.7       814.1       237    ₹23,559       +13.9%      -0.1%     
  MARUTI      6      AUTO       01-Dec-17   7,994.1     8,730.3     23     ₹16,933       +9.2%       -0.1%     
  BRITANNIA   33     FMCG       03-Oct-17   1,920.1     2,097.2     92     ₹16,299       +9.2%       +0.8%     
  HONAUT      29     MFG        03-Oct-17   15,425.8    16,698.3    11     ₹13,998       +8.2%       -4.5%     
  BALKRISIND  3      MFG        01-Dec-17   978.3       1,047.3     194    ₹13,372       +7.0%       -1.5%     
  BBTC        37     FMCG       01-Dec-17   1,478.3     1,383.9     128    ₹-12,091      -6.4%       -8.5%     

  AFTER: Invested ₹3,683,744 | Cash ₹111,207 | Total ₹3,794,951 | Positions 19/20 | Slot ₹189,837

========================================================================
  REBALANCE #20  —  02 Apr 2018
  NAV: ₹3,653,344  |  Slot: ₹182,667  |  Cash: ₹111,207
========================================================================
  [SECTOR CAP≤4] dropped: GODREJCP

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HONAUT      46     MFG        03-Oct-17   15,425.8    16,897.3    11     ₹16,187       +9.5%     181d  
  RAJESHEXPO  63     CON DUR    01-Aug-17   714.7       733.3       237    ₹4,409        +2.6%     244d  
  ABBOTINDIA  48     HEALTH     01-Feb-18   5,020.8     4,963.2     37     ₹-2,130       -1.1%     60d   
  M&M         41     AUTO       01-Feb-18   734.5       687.8       258    ₹-12,028      -6.3%     60d   
  OFSS        67     IT         01-Feb-18   2,861.2     2,678.0     66     ₹-12,094      -6.4%     60d   
  BBTC        88     FMCG       01-Dec-17   1,478.3     1,206.8     128    ₹-34,756      -18.4%    122d  

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TITAN       2      CON DUR    2.766    1.18   +111.8%   +11.2%    917.9       199    ₹182,657      +6.9%     
  CHOLAFIN    6      FIN SVC    2.450    0.96   +49.0%    +14.9%    287.4       635    ₹182,491      +2.6%     
  VBL         7      FMCG       2.354    0.58   +70.4%    +8.5%     37.0        4931   ₹182,663      +2.3%     
  INDUSINDBK  8      PVT BNK    2.316    0.72   +30.4%    +9.3%     1,717.3     106    ₹182,035      +3.7%     
  PIDILITIND  14     MFG        1.907    0.82   +37.6%    +3.0%     452.7       403    ₹182,443      +3.8%     
  HCLTECH     15     IT         1.901    0.21   +13.7%    +11.1%    380.7       479    ₹182,349      +2.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HDFC        8      PVT BNK    01-Feb-17   297.7       443.3       473    ₹68,842       +48.9%      +2.9%     
  HDFCBANK    9      PVT BNK    01-Feb-17   297.7       443.3       473    ₹68,842       +48.9%      +2.9%     
  HINDUNILVR  14     FMCG       01-Jun-17   940.4       1,178.2     167    ₹39,711       +25.3%      +2.2%     
  BRITANNIA   3      FMCG       03-Oct-17   1,920.1     2,259.4     92     ₹31,218       +17.7%      +4.5%     
  PGHH        18     FMCG       01-Aug-17   7,228.0     8,386.1     23     ₹26,637       +16.0%      +0.5%     
  RELIANCE    36     OIL&GAS    01-Aug-17   352.5       392.6       481    ₹19,278       +11.4%      -1.5%     
  MARUTI      30     AUTO       01-Dec-17   7,994.1     8,364.8     23     ₹8,526        +4.6%       +2.1%     
  BALKRISIND  39     MFG        01-Dec-17   978.3       1,002.1     194    ₹4,610        +2.4%       +0.5%     
  KOTAKBANK   17     PVT BNK    01-Feb-18   223.0       218.2       851    ₹-4,131       -2.2%       +3.0%     
  BIOCON      23     HEALTH     01-Feb-18   305.0       294.8       622    ₹-6,312       -3.3%       +0.9%     
  MPHASIS     6      IT         01-Feb-18   722.3       695.3       262    ₹-7,067       -3.7%       +0.0%     
  TCS         16     IT         01-Feb-18   1,271.5     1,178.8     149    ₹-13,820      -7.3%       +0.4%     
  IBULHSGFIN  28     FIN SVC    01-Feb-18   1,010.8     913.5       187    ₹-18,182      -9.6%       +1.1%     

  AFTER: Invested ₹3,584,796 | Cash ₹67,248 | Total ₹3,652,044 | Positions 19/20 | Slot ₹182,667

========================================================================
  REBALANCE #21  —  01 Jun 2018
  NAV: ₹3,975,824  |  Slot: ₹198,791  |  Cash: ₹67,248
========================================================================
  [SECTOR CAP≤4] dropped: COLPAL

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  RELIANCE    45     OIL&GAS    01-Aug-17   352.5       409.2       481    ₹27,230       +16.1%    304d  
  PGHH        44     FMCG       01-Aug-17   7,228.0     8,342.6     23     ₹25,637       +15.4%    304d  
  BIOCON      —      HEALTH     01-Feb-18   305.0       319.7       622    ₹9,161        +4.8%     120d  
  MARUTI      49     AUTO       01-Dec-17   7,994.1     8,179.7     23     ₹4,268        +2.3%     182d  
  HCLTECH     76     IT         02-Apr-18   380.7       352.4       479    ₹-13,530      -7.4%     60d   
  IBULHSGFIN  62     FIN SVC    01-Feb-18   1,010.8     918.6       187    ₹-17,243      -9.1%     120d  

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    1      CONSUMP    3.171    0.90   +166.5%   +23.0%    245.3       810    ₹198,689      -1.3%     
  COFORGE     4      IT         2.529    0.90   +120.6%   +32.7%    201.6       986    ₹198,731      +3.3%     
  TECHM       6      IT         2.363    0.30   +89.2%    +14.4%    516.6       384    ₹198,379      +2.4%     
  DABUR       8      FMCG       2.339    0.60   +39.8%    +19.0%    354.9       560    ₹198,771      +3.3%     
  M&M         10     AUTO       2.222    0.80   +33.8%    +23.8%    829.0       239    ₹198,138      +4.7%     
  PAGEIND     17     MFG        1.822    1.19   +75.6%    +14.1%    22,759.4    8      ₹182,075      +3.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HDFC        13     PVT BNK    01-Feb-17   297.7       487.5       473    ₹89,782       +63.8%      +4.9%     
  HDFCBANK    14     PVT BNK    01-Feb-17   297.7       487.5       473    ₹89,782       +63.8%      +4.9%     
  HINDUNILVR  3      FMCG       01-Jun-17   940.4       1,385.4     167    ₹74,301       +47.3%      +2.3%     
  BRITANNIA   2      FMCG       03-Oct-17   1,920.1     2,571.5     92     ₹59,929       +33.9%      +2.9%     
  VBL         17     FMCG       02-Apr-18   37.0        44.6        4931   ₹37,088       +20.3%      +7.1%     
  MPHASIS     8      IT         01-Feb-18   722.3       868.5       262    ₹38,293       +20.2%      +0.6%     
  PIDILITIND  5      MFG        02-Apr-18   452.7       538.3       403    ₹34,487       +18.9%      +0.5%     
  KOTAKBANK   10     PVT BNK    01-Feb-18   223.0       262.2       851    ₹33,363       +17.6%      +3.5%     
  TCS         19     IT         01-Feb-18   1,271.5     1,415.4     149    ₹21,437       +11.3%      +0.3%     
  CHOLAFIN    28     FIN SVC    02-Apr-18   287.4       308.5       635    ₹13,388       +7.3%       +1.4%     
  BALKRISIND  39     MFG        01-Dec-17   978.3       1,039.7     194    ₹11,912       +6.3%       -3.2%     
  INDUSINDBK  22     PVT BNK    02-Apr-18   1,717.3     1,822.8     106    ₹11,177       +6.1%       +0.9%     
  TITAN       18     CON DUR    02-Apr-18   917.9       875.0       199    ₹-8,532       -4.7%       -2.9%     

  AFTER: Invested ₹3,967,102 | Cash ₹7,326 | Total ₹3,974,429 | Positions 19/20 | Slot ₹198,791

========================================================================
  REBALANCE #22  —  01 Aug 2018
  NAV: ₹4,227,778  |  Slot: ₹211,389  |  Cash: ₹7,326
========================================================================
  [SECTOR CAP≤4] dropped: INFY

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VBL         38     FMCG       02-Apr-18   37.0        43.0        4931   ₹29,508       +16.2%    121d  
  BALKRISIND  43     MFG        01-Dec-17   978.3       1,104.2     194    ₹24,428       +12.9%    243d  
  INDUSINDBK  34     PVT BNK    02-Apr-18   1,717.3     1,919.4     106    ₹21,425       +11.8%    121d  
  CHOLAFIN    70     FIN SVC    02-Apr-18   287.4       285.5       635    ₹-1,226       -0.7%     121d  
  TITAN       36     CON DUR    02-Apr-18   917.9       895.8       199    ₹-4,398       -2.4%     121d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BAJFINANCE  2      FIN SVC    2.862    1.17   +60.3%    +46.8%    264.0       800    ₹211,212      +5.8%     
  ABBOTINDIA  4      HEALTH     2.555    0.17   +80.6%    +23.2%    6,983.6     30     ₹209,509      +4.1%     
  RELIANCE    5      OIL&GAS    2.527    1.11   +50.6%    +25.8%    527.8       400    ₹211,111      +8.4%     
  BATAINDIA   10     CON DUR    2.194    1.14   +64.4%    +21.6%    878.9       240    ₹210,946      +7.6%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HDFC        20     PVT BNK    01-Feb-17   297.7       498.6       473    ₹95,038       +67.5%      +0.0%     
  HDFCBANK    21     PVT BNK    01-Feb-17   297.7       498.6       473    ₹95,038       +67.5%      +0.0%     
  HINDUNILVR  3      FMCG       01-Jun-17   940.4       1,523.6     167    ₹97,386       +62.0%      +3.2%     
  BRITANNIA   1      FMCG       03-Oct-17   1,920.1     2,882.3     92     ₹88,528       +50.1%      +2.0%     
  MPHASIS     5      IT         01-Feb-18   722.3       1,006.5     262    ₹74,460       +39.3%      +6.6%     
  TCS         8      IT         01-Feb-18   1,271.5     1,618.1     149    ₹51,631       +27.3%      +1.6%     
  PAGEIND     9      MFG        01-Jun-18   22,759.4    27,338.4    8      ₹36,632       +20.1%      +4.9%     
  PIDILITIND  23     MFG        02-Apr-18   452.7       543.6       403    ₹36,638       +20.1%      +3.8%     
  KOTAKBANK   28     PVT BNK    01-Feb-18   223.0       261.3       851    ₹32,579       +17.2%      -1.7%     
  DABUR       13     FMCG       01-Jun-18   354.9       403.3       560    ₹27,053       +13.6%      +11.8%    
  COFORGE     7      IT         01-Jun-18   201.6       227.3       986    ₹25,434       +12.8%      +6.2%     
  JUBLFOOD    11     CONSUMP    01-Jun-18   245.3       273.8       810    ₹23,062       +11.6%      -1.1%     
  M&M         22     AUTO       01-Jun-18   829.0       861.9       239    ₹7,846        +4.0%       +1.6%     
  TECHM       17     IT         01-Jun-18   516.6       513.2       384    ₹-1,303       -0.7%       +5.1%     

  AFTER: Invested ₹4,073,852 | Cash ₹152,925 | Total ₹4,226,777 | Positions 18/20 | Slot ₹211,389

========================================================================
  REBALANCE #23  —  01 Oct 2018
  NAV: ₹4,104,583  |  Slot: ₹205,229  |  Cash: ₹152,925
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
  HDFC        63     PVT BNK    01-Feb-17   297.7       470.2       473    ₹81,572       +57.9% ⚠    +1.2%     
  HDFCBANK    64     PVT BNK    01-Feb-17   297.7       470.2       473    ₹81,572       +57.9% ⚠    +1.2%     
  HINDUNILVR  23     FMCG       01-Jun-17   940.4       1,442.8     167    ₹83,889       +53.4%      -0.1%     
  TCS         1      IT         01-Feb-18   1,271.5     1,846.5     149    ₹85,669       +45.2%      +6.3%     
  BRITANNIA   29     FMCG       03-Oct-17   1,920.1     2,595.3     92     ₹62,120       +35.2%      -2.7%     
  MPHASIS     8      IT         01-Feb-18   722.3       968.9       262    ₹64,612       +34.1%      -3.5%     
  PAGEIND     5      MFG        01-Jun-18   22,759.4    30,462.0    8      ₹61,621       +33.8%      +0.7%     
  DABUR       11     FMCG       01-Jun-18   354.9       409.8       560    ₹30,710       +15.5%      -2.0%     
  PIDILITIND  27     MFG        02-Apr-18   452.7       503.3       403    ₹20,404       +11.2%      -5.4%     
  TECHM       6      IT         01-Jun-18   516.6       572.3       384    ₹21,397       +10.8%      +1.7%     
  COFORGE     7      IT         01-Jun-18   201.6       214.9       986    ₹13,169       +6.6%       -6.0%     
  RELIANCE    3      OIL&GAS    01-Aug-18   527.8       545.2       400    ₹6,967        +3.3%       -0.5%     
  BATAINDIA   14     CON DUR    01-Aug-18   878.9       903.9       240    ₹5,996        +2.8%       -2.9%     
  KOTAKBANK   93     PVT BNK    01-Feb-18   223.0       223.3       851    ₹275          +0.1% ⚠     -6.5%     
  ABBOTINDIA  10     HEALTH     01-Aug-18   6,983.6     6,960.2     30     ₹-702         -0.3%       -5.4%     
  JUBLFOOD    22     CONSUMP    01-Jun-18   245.3       243.8       810    ₹-1,194       -0.6%       -7.2%     
  M&M         36     AUTO       01-Jun-18   829.0       785.7       239    ₹-10,354      -5.2%       -7.5%     
  BAJFINANCE  50     FIN SVC    01-Aug-18   264.0       214.1       800    ₹-39,895      -18.9%      -10.5%    
  ⚠  WAZ < 0 (momentum below universe mean): HDFC, HDFCBANK, KOTAKBANK

  AFTER: Invested ₹3,951,657 | Cash ₹152,925 | Total ₹4,104,583 | Positions 18/20 | Slot ₹205,229

========================================================================
  REBALANCE #24  —  03 Dec 2018
  NAV: ₹4,078,325  |  Slot: ₹203,916  |  Cash: ₹152,925
========================================================================
  [SECTOR CAP≤4] dropped: INFY

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BRITANNIA   33     FMCG       03-Oct-17   1,920.1     2,748.7     92     ₹76,235       +43.2%    426d  
  MPHASIS     42     IT         01-Feb-18   722.3       834.0       262    ₹29,277       +15.5%    305d  
  DABUR       51     FMCG       01-Jun-18   354.9       386.6       560    ₹17,703       +8.9%     185d  
  PAGEIND     73     MFG        01-Jun-18   22,759.4    24,355.1    8      ₹12,766       +7.0%     185d  
  RELIANCE    45     OIL&GAS    01-Aug-18   527.8       511.9       400    ₹-6,356       -3.0%     124d  
  BAJFINANCE  37     FIN SVC    01-Aug-18   264.0       243.3       800    ₹-16,612      -7.9%     124d  
  M&M         96     AUTO       01-Jun-18   829.0       705.5       239    ₹-29,529      -14.9%    185d  

  ENTRIES (7)
  [52w filter blocked 1: JUBILANT(-22.6%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COLPAL      1      FMCG       2.773    0.38   +24.6%    +11.1%    1,072.8     190    ₹203,841      +8.5%     
  AUROPHARMA  4      HEALTH     2.311    1.18   +16.3%    +17.5%    771.7       264    ₹203,721      +2.1%     
  WIPRO       5      IT         2.256    0.30   +12.2%    +11.0%    112.7       1809   ₹203,868      +2.4%     
  ICICIGI     6      FIN SVC    2.243    0.56   +23.3%    +13.0%    818.5       249    ₹203,815      +4.7%     
  HONAUT      7      MFG        2.127    0.78   +33.7%    +4.4%     22,459.3    9      ₹202,134      +7.7%     
  VBL         9      FMCG       2.083    0.40   +57.6%    -3.8%     45.3        4497   ₹203,885      +0.7%     
  LT          10     INFRA      2.075    1.08   +18.9%    +5.3%     1,270.2     160    ₹203,231      +2.9%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  2      FMCG       01-Jun-17   940.4       1,613.0     167    ₹112,309      +71.5%      +7.1%     
  HDFC        16     PVT BNK    01-Feb-17   297.7       488.1       473    ₹90,077       +64.0%      +4.0%     
  HDFCBANK    17     PVT BNK    01-Feb-17   297.7       488.1       473    ₹90,077       +64.0%      +4.0%     
  TCS         8      IT         01-Feb-18   1,271.5     1,626.3     149    ₹52,860       +27.9%      +3.4%     
  PIDILITIND  11     MFG        02-Apr-18   452.7       556.8       403    ₹41,952       +23.0%      +3.7%     
  BATAINDIA   14     CON DUR    01-Aug-18   878.9       993.5       240    ₹27,490       +13.0%      +6.7%     
  KOTAKBANK   31     PVT BNK    01-Feb-18   223.0       244.1       851    ₹17,909       +9.4%       +4.2%     
  JUBLFOOD    28     CONSUMP    01-Jun-18   245.3       261.4       810    ₹13,005       +6.5%       +10.2%    
  TECHM       12     IT         01-Jun-18   516.6       537.0       384    ₹7,841        +4.0%       +1.4%     
  ABBOTINDIA  3      HEALTH     01-Aug-18   6,983.6     7,082.9     30     ₹2,979        +1.4%       +4.3%     
  COFORGE     24     IT         01-Jun-18   201.6       203.2       986    ₹1,586        +0.8%       -1.8%     

  AFTER: Invested ₹3,899,217 | Cash ₹177,417 | Total ₹4,076,634 | Positions 18/20 | Slot ₹203,916

========================================================================
  REBALANCE #25  —  01 Feb 2019
  NAV: ₹4,176,825  |  Slot: ₹208,841  |  Cash: ₹177,417
========================================================================
  [SECTOR CAP≤4] dropped: MARICO, UBL

  [REGIME OFF] Nifty 200 5,701.6 < EMA200 5,703.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  5      FMCG       01-Jun-17   940.4       1,589.7     167    ₹108,429      +69.0%      +1.8%     
  HDFC        22     PVT BNK    01-Feb-17   297.7       482.9       473    ₹87,581       +62.2%      -0.3%     
  HDFCBANK    23     PVT BNK    01-Feb-17   297.7       482.9       473    ₹87,581       +62.2%      -0.3%     
  TCS         20     IT         01-Feb-18   1,271.5     1,668.9     149    ₹59,204       +31.2%      +4.9%     
  BATAINDIA   3      CON DUR    01-Aug-18   878.9       1,068.3     240    ₹45,455       +21.5%      +1.3%     
  COFORGE     17     IT         01-Jun-18   201.6       243.0       986    ₹40,819       +20.5%      +5.6%     
  PIDILITIND  11     MFG        02-Apr-18   452.7       542.8       403    ₹36,302       +19.9%      -0.2%     
  WIPRO       12     IT         03-Dec-18   112.7       127.9       1809   ₹27,544       +13.5%      +6.9%     
  KOTAKBANK   19     PVT BNK    01-Feb-18   223.0       250.0       851    ₹22,924       +12.1%      +0.7%     
  TECHM       30     IT         01-Jun-18   516.6       562.6       384    ₹17,650       +8.9%       +3.9%     
  JUBLFOOD    14     CONSUMP    01-Jun-18   245.3       266.9       810    ₹17,461       +8.8%       +10.6%    
  VBL         32     FMCG       03-Dec-18   45.3        48.3        4497   ₹13,401       +6.6%       +4.4%     
  ABBOTINDIA  8      HEALTH     01-Aug-18   6,983.6     7,377.5     30     ₹11,815       +5.6%       +1.0%     
  ICICIGI     41     FIN SVC    03-Dec-18   818.5       839.3       249    ₹5,177        +2.5%       +2.5%     
  COLPAL      9      FMCG       03-Dec-18   1,072.8     1,090.0     190    ₹3,250        +1.6%       -0.1%     
  AUROPHARMA  38     HEALTH     03-Dec-18   771.7       764.9       264    ₹-1,784       -0.9%       +3.3%     
  HONAUT      26     MFG        03-Dec-18   22,459.3    21,361.3    9      ₹-9,882       -4.9%       -0.3%     
  LT          63     INFRA      03-Dec-18   1,270.2     1,178.6     160    ₹-14,650      -7.2% ⚠     -0.3%     
  ⚠  WAZ < 0 (momentum below universe mean): LT

  AFTER: Invested ₹3,999,408 | Cash ₹177,417 | Total ₹4,176,825 | Positions 18/20 | Slot ₹208,841

========================================================================
  REBALANCE #26  —  01 Apr 2019
  NAV: ₹4,356,084  |  Slot: ₹217,804  |  Cash: ₹177,417
========================================================================
  [SECTOR CAP≤4] dropped: INFY

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HINDUNILVR  57     FMCG       01-Jun-17   940.4       1,493.2     167    ₹92,302       +58.8%    669d  
  LT          73     INFRA      03-Dec-18   1,270.2     1,256.6     160    ₹-2,178       -1.1%     119d  
  COLPAL      55     FMCG       03-Dec-18   1,072.8     1,060.2     190    ₹-2,394       -1.2%     119d  
  AUROPHARMA  24     HEALTH     03-Dec-18   771.7       761.0       264    ₹-2,829       -1.4%     119d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TITAN       2      CON DUR    2.507    0.91   +29.0%    +26.2%    1,094.0     199    ₹217,705      +2.9%     
  RBLBANK     4      PVT BNK    2.408    1.19   +47.7%    +17.8%    659.7       330    ₹217,695      +6.6%     
  ASIANPAINT  9      CONSUMP    2.005    0.96   +36.2%    +8.8%     1,397.3     155    ₹216,585      +3.0%     
  ABFRL       12     CONSUMP    1.967    0.64   +58.5%    +9.5%     221.2       984    ₹217,654      +0.6%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HDFC        9      PVT BNK    01-Feb-17   297.7       534.0       473    ₹111,765      +79.4%      +3.5%     
  HDFCBANK    10     PVT BNK    01-Feb-17   297.7       534.0       473    ₹111,765      +79.4%      +3.5%     
  BATAINDIA   1      CON DUR    01-Aug-18   878.9       1,295.6     240    ₹100,010      +47.4%      +3.3%     
  PIDILITIND  13     MFG        02-Apr-18   452.7       605.6       403    ₹61,599       +33.8%      +6.2%     
  TCS         15     IT         01-Feb-18   1,271.5     1,670.3     149    ₹59,412       +31.4%      +1.5%     
  COFORGE     23     IT         01-Jun-18   201.6       245.6       986    ₹43,452       +21.9%      +1.1%     
  KOTAKBANK   25     PVT BNK    01-Feb-18   223.0       266.6       851    ₹37,102       +19.5%      +2.6%     
  ICICIGI     16     FIN SVC    03-Dec-18   818.5       968.3       249    ₹37,282       +18.3%      +4.1%     
  JUBLFOOD    21     CONSUMP    01-Jun-18   245.3       287.0       810    ₹33,753       +17.0%      +5.1%     
  VBL         5      FMCG       03-Dec-18   45.3        52.0        4497   ₹29,879       +14.7%      +9.5%     
  TECHM       27     IT         01-Jun-18   516.6       591.9       384    ₹28,906       +14.6%      -0.5%     
  WIPRO       35     IT         03-Dec-18   112.7       120.1       1809   ₹13,416       +6.6%       -0.3%     
  HONAUT      19     MFG        03-Dec-18   22,459.3    21,836.7    9      ₹-5,604       -2.8%       +0.4%     
  ABBOTINDIA  42     HEALTH     01-Aug-18   6,983.6     6,646.9     30     ₹-10,103      -4.8%       -0.9%     

  AFTER: Invested ₹4,195,556 | Cash ₹159,495 | Total ₹4,355,051 | Positions 18/20 | Slot ₹217,804

========================================================================
  REBALANCE #27  —  03 Jun 2019
  NAV: ₹4,582,353  |  Slot: ₹229,118  |  Cash: ₹159,495
========================================================================
  [SECTOR CAP≤4] dropped: AXISBANK

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PIDILITIND  —      MFG        02-Apr-18   452.7       626.4       403    ₹69,986       +38.4%    427d  
  COFORGE     68     IT         01-Jun-18   201.6       241.5       986    ₹39,385       +19.8%    367d  
  TECHM       94     IT         01-Jun-18   516.6       570.9       384    ₹20,833       +10.5%    367d  
  JUBLFOOD    74     CONSUMP    01-Jun-18   245.3       263.8       810    ₹15,002       +7.6%     367d  
  ASIANPAINT  63     CONSUMP    01-Apr-19   1,397.3     1,366.0     155    ₹-4,861       -2.2%     63d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  NAUKRI      2      IT         2.893    0.55   +97.5%    +39.0%    451.4       507    ₹228,853      +15.9%    
  GUJGASLTD   4      OIL&GAS    2.624    0.84   +10.2%    +55.1%    177.2       1293   ₹229,092      +9.8%     
  PIIND       8      MFG        2.321    0.50   +35.4%    +23.7%    1,116.0     205    ₹228,779      +4.8%     
  SRF         10     MFG        2.315    1.20   +51.7%    +27.8%    558.0       410    ₹228,761      +2.6%     
  LT          12     INFRA      2.106    1.02   +19.6%    +22.2%    1,387.6     165    ₹228,947      +5.7%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HDFC        7      PVT BNK    01-Feb-17   297.7       567.6       473    ₹127,640      +90.6%      +3.3%     
  HDFCBANK    8      PVT BNK    01-Feb-17   297.7       567.6       473    ₹127,640      +90.6%      +3.3%     
  TCS         25     IT         01-Feb-18   1,271.5     1,843.5     149    ₹85,216       +45.0%      +5.4%     
  BATAINDIA   12     CON DUR    01-Aug-18   878.9       1,272.7     240    ₹94,507       +44.8%      +0.7%     
  ICICIGI     1      FIN SVC    03-Dec-18   818.5       1,164.0     249    ₹86,009       +42.2%      +8.1%     
  KOTAKBANK   17     PVT BNK    01-Feb-18   223.0       304.8       851    ₹69,602       +36.7%      +3.8%     
  VBL         22     FMCG       03-Dec-18   45.3        54.8        4497   ₹42,550       +20.9%      +3.0%     
  WIPRO       26     IT         03-Dec-18   112.7       133.7       1809   ₹38,039       +18.7%      +1.9%     
  HONAUT      3      MFG        03-Dec-18   22,459.3    26,282.6    9      ₹34,409       +17.0%      +7.2%     
  TITAN       13     CON DUR    01-Apr-19   1,094.0     1,236.5     199    ₹28,351       +13.0%      +5.1%     
  ABBOTINDIA  49     HEALTH     01-Aug-18   6,983.6     7,181.9     30     ₹5,949        +2.8%       +3.0%     
  RBLBANK     19     PVT BNK    01-Apr-19   659.7       675.0       330    ₹5,060        +2.3%       +2.7%     
  ABFRL       52     CONSUMP    01-Apr-19   221.2       215.7       984    ₹-5,383       -2.5%       +4.7%     

  AFTER: Invested ₹4,432,117 | Cash ₹148,876 | Total ₹4,580,994 | Positions 18/20 | Slot ₹229,118

========================================================================
  REBALANCE #28  —  01 Aug 2019
  NAV: ₹4,213,707  |  Slot: ₹210,685  |  Cash: ₹148,876
========================================================================

  [REGIME OFF] Nifty 200 5,661.5 < EMA200 5,891.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HDFC        67     PVT BNK    01-Feb-17   297.7       517.5       473    ₹103,984      +73.8% ⚠    -4.1%     
  HDFCBANK    68     PVT BNK    01-Feb-17   297.7       517.5       473    ₹103,984      +73.8% ⚠    -4.1%     
  TCS         30     IT         01-Feb-18   1,271.5     1,811.0     149    ₹80,381       +42.4%      +1.5%     
  BATAINDIA   13     CON DUR    01-Aug-18   878.9       1,221.7     240    ₹82,250       +39.0%      -2.1%     
  ICICIGI     8      FIN SVC    03-Dec-18   818.5       1,101.5     249    ₹70,460       +34.6%      +2.9%     
  KOTAKBANK   26     PVT BNK    01-Feb-18   223.0       297.9       851    ₹63,733       +33.6%      -0.1%     
  VBL         16     FMCG       03-Dec-18   45.3        54.2        4497   ₹39,917       +19.6%      -0.8%     
  ABBOTINDIA  7      HEALTH     01-Aug-18   6,983.6     7,695.4     30     ₹21,354       +10.2%      -1.5%     
  WIPRO       41     IT         03-Dec-18   112.7       124.1       1809   ₹20,683       +10.1%      +1.1%     
  HONAUT      34     MFG        03-Dec-18   22,459.3    22,664.6    9      ₹1,847        +0.9%       -0.7%     
  PIIND       4      MFG        03-Jun-19   1,116.0     1,093.1     205    ₹-4,701       -2.1%       -0.1%     
  NAUKRI      1      IT         03-Jun-19   451.4       433.1       507    ₹-9,270       -4.1%       +0.1%     
  TITAN       47     CON DUR    01-Apr-19   1,094.0     1,037.1     199    ₹-11,319      -5.2%       -5.1%     
  SRF         3      MFG        03-Jun-19   558.0       515.0       410    ₹-17,601      -7.7%       -3.8%     
  GUJGASLTD   22     OIL&GAS    03-Jun-19   177.2       161.8       1293   ₹-19,862      -8.7%       +2.8%     
  LT          38     INFRA      03-Jun-19   1,387.6     1,224.0     165    ₹-26,983      -11.8%      -4.1%     
  ABFRL       40     CONSUMP    01-Apr-19   221.2       184.6       984    ₹-36,033      -16.6%      -6.5%     
  RBLBANK     130    PVT BNK    01-Apr-19   659.7       385.4       330    ₹-90,521      -41.6%      -21.7%    
  ⚠  WAZ < 0 (momentum below universe mean): HDFC, HDFCBANK

  AFTER: Invested ₹4,064,830 | Cash ₹148,876 | Total ₹4,213,707 | Positions 18/20 | Slot ₹210,685

========================================================================
  REBALANCE #29  —  01 Oct 2019
  NAV: ₹4,539,765  |  Slot: ₹226,988  |  Cash: ₹148,876
========================================================================
  [SECTOR CAP≤4] dropped: PGHH

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TCS         76     IT         01-Feb-18   1,271.5     1,711.2     149    ₹65,515       +34.6%    607d  
  VBL         57     FMCG       03-Dec-18   45.3        52.7        4497   ₹32,922       +16.1%    302d  
  LT          60     INFRA      03-Jun-19   1,387.6     1,319.6     165    ₹-11,219      -4.9%     120d  
  WIPRO       106    IT         03-Dec-18   112.7       107.2       1809   ₹-9,961       -4.9%     302d  
  SRF         55     MFG        03-Jun-19   558.0       524.9       410    ₹-13,536      -5.9%     120d  
  GUJGASLTD   42     OIL&GAS    03-Jun-19   177.2       163.7       1293   ₹-17,440      -7.6%     120d  
  ABFRL       64     CONSUMP    01-Apr-19   221.2       203.5       984    ₹-17,362      -8.0%     183d  
  NAUKRI      52     IT         03-Jun-19   451.4       403.6       507    ₹-24,250      -10.6%    120d  
  RBLBANK     132    PVT BNK    01-Apr-19   659.7       291.7       330    ₹-121,426     -55.8%    183d  

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COLPAL      1      FMCG       3.142    0.60   +36.9%    +32.6%    1,276.3     177    ₹225,912      +7.4%     
  HDFCAMC     2      FIN SVC    3.129    0.89   +74.0%    +33.9%    1,208.6     187    ₹226,008      +1.9%     
  BERGEPAINT  4      CON DUR    2.970    0.98   +43.0%    +37.3%    350.7       647    ₹226,935      +8.3%     
  ASIANPAINT  5      CONSUMP    2.806    0.97   +36.0%    +30.0%    1,662.1     136    ₹226,048      +6.0%     
  HDFCLIFE    6      FIN SVC    2.724    0.74   +45.5%    +28.9%    587.1       386    ₹226,617      +7.6%     
  NESTLEIND   8      FMCG       2.452    0.65   +38.1%    +16.9%    636.8       356    ₹226,711      +3.8%     
  LALPATHLAB  10     HEALTH     2.250    0.74   +38.6%    +25.3%    634.6       357    ₹226,555      +1.8%     
  DMART       11     FMCG       2.186    1.09   +24.3%    +35.6%    1,896.0     119    ₹225,624      +9.4%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HDFC        24     PVT BNK    01-Feb-17   297.7       581.8       473    ₹134,372      +95.4%      +5.4%     
  HDFCBANK    25     PVT BNK    01-Feb-17   297.7       581.8       473    ₹134,372      +95.4%      +5.4%     
  BATAINDIA   3      CON DUR    01-Aug-18   878.9       1,630.0     240    ₹180,256      +85.5%      +6.5%     
  KOTAKBANK   15     PVT BNK    01-Feb-18   223.0       328.3       851    ₹89,564       +47.2%      +6.1%     
  ABBOTINDIA  7      HEALTH     01-Aug-18   6,983.6     9,965.1     30     ₹89,444       +42.7%      +6.8%     
  ICICIGI     26     FIN SVC    03-Dec-18   818.5       1,151.0     249    ₹82,786       +40.6%      +1.4%     
  HONAUT      16     MFG        03-Dec-18   22,459.3    27,751.2    9      ₹47,627       +23.6%      +4.4%     
  TITAN       21     CON DUR    01-Apr-19   1,094.0     1,256.2     199    ₹32,269       +14.8%      +6.3%     
  PIIND       9      MFG        03-Jun-19   1,116.0     1,235.2     205    ₹24,427       +10.7%      +0.5%     

  AFTER: Invested ₹4,369,842 | Cash ₹167,774 | Total ₹4,537,616 | Positions 17/20 | Slot ₹226,988

========================================================================
  REBALANCE #30  —  02 Dec 2019
  NAV: ₹4,745,150  |  Slot: ₹237,257  |  Cash: ₹167,774
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJFINSV

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  KOTAKBANK   32     PVT BNK    01-Feb-18   223.0       325.2       851    ₹86,988       +45.8%    669d  
  HONAUT      37     MFG        03-Dec-18   22,459.3    26,924.4    9      ₹40,186       +19.9%    364d  
  TITAN       68     CON DUR    01-Apr-19   1,094.0     1,131.8     199    ₹7,518        +3.5%     245d  
  ASIANPAINT  36     CONSUMP    01-Oct-19   1,662.1     1,638.9     136    ₹-3,153       -1.4%     62d   
  DMART       30     FMCG       01-Oct-19   1,896.0     1,846.7     119    ₹-5,867       -2.6%     62d   
  HDFCLIFE    47     FIN SVC    01-Oct-19   587.1       558.8       386    ₹-10,910      -4.8%     62d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  WHIRLPOOL   5      CON DUR    2.899    0.32   +52.3%    +40.2%    2,121.8     111    ₹235,523      -0.8%     
  IGL         7      OIL&GAS    2.491    0.15   +56.4%    +26.9%    185.5       1279   ₹237,227      +2.4%     
  MANAPPURAM  8      FIN SVC    2.365    0.22   +87.0%    +28.7%    139.6       1699   ₹237,206      -1.9%     
  NAM-INDIA   9      FIN SVC    2.341    0.21   +116.1%   +37.1%    290.5       816    ₹237,025      -0.4%     
  GUJGASLTD   11     OIL&GAS    2.246    0.09   +74.3%    +21.1%    206.7       1148   ₹237,236      +8.8%     
  RELIANCE    12     OIL&GAS    2.215    0.17   +41.6%    +24.4%    706.5       335    ₹236,664      +4.8%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HDFC        26     PVT BNK    01-Feb-17   297.7       589.7       473    ₹138,107      +98.1%      +0.2%     
  HDFCBANK    27     PVT BNK    01-Feb-17   297.7       589.7       473    ₹138,107      +98.1%      +0.2%     
  BATAINDIA   21     CON DUR    01-Aug-18   878.9       1,526.7     240    ₹155,459      +73.7%      -2.5%     
  ABBOTINDIA  2      HEALTH     01-Aug-18   6,983.6     11,580.2    30     ₹137,896      +65.8%      +2.8%     
  ICICIGI     23     FIN SVC    03-Dec-18   818.5       1,314.4     249    ₹123,475      +60.6%      +2.1%     
  PIIND       3      MFG        03-Jun-19   1,116.0     1,478.3     205    ₹74,274       +32.5%      +5.3%     
  HDFCAMC     1      FIN SVC    01-Oct-19   1,208.6     1,498.3     187    ₹54,175       +24.0%      -0.9%     
  LALPATHLAB  4      HEALTH     01-Oct-19   634.6       760.0       357    ₹44,752       +19.8%      +2.5%     
  BERGEPAINT  6      CON DUR    01-Oct-19   350.7       400.4       647    ₹32,120       +14.2%      +1.3%     
  NESTLEIND   14     FMCG       01-Oct-19   636.8       677.6       356    ₹14,517       +6.4%       +0.9%     
  COLPAL      16     FMCG       01-Oct-19   1,276.3     1,248.1     177    ₹-4,990       -2.2%       -3.8%     

  AFTER: Invested ₹4,595,575 | Cash ₹147,887 | Total ₹4,743,463 | Positions 17/20 | Slot ₹237,257

========================================================================
  REBALANCE #31  —  03 Feb 2020
  NAV: ₹4,968,964  |  Slot: ₹248,448  |  Cash: ₹147,887
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HDFC        69     PVT BNK    01-Feb-17   297.7       555.7       473    ₹122,032      +86.7%    1097d 
  HDFCBANK    70     PVT BNK    01-Feb-17   297.7       555.7       473    ₹122,032      +86.7%    1097d 
  ICICIGI     41     FIN SVC    03-Dec-18   818.5       1,247.4     249    ₹106,794      +52.4%    427d  
  COLPAL      112    FMCG       01-Oct-19   1,276.3     1,161.5     177    ₹-20,332      -9.0%     125d  
  RELIANCE    75     OIL&GAS    02-Dec-19   706.5       617.0       335    ₹-29,984      -12.7%    63d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COROMANDEL  2      MFG        3.274    0.09   +45.3%    +35.1%    585.9       424    ₹248,438      +6.6%     
  SRF         4      MFG        3.093    -0.04  +87.1%    +29.3%    741.4       335    ₹248,372      +4.2%     
  JUBLFOOD    5      CONSUMP    2.586    0.36   +66.3%    +25.1%    385.8       643    ₹248,070      +9.9%     
  COFORGE     8      IT         2.407    0.23   +49.5%    +18.8%    349.4       711    ₹248,424      +2.6%     
  IPCALAB     9      HEALTH     2.406    0.08   +53.4%    +18.6%    583.3       425    ₹247,894      -1.9%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   13     CON DUR    01-Aug-18   878.9       1,736.3     240    ₹205,763      +97.5%      +2.9%     
  ABBOTINDIA  17     HEALTH     01-Aug-18   6,983.6     11,694.8    30     ₹141,337      +67.5%      +0.5%     
  PIIND       7      MFG        03-Jun-19   1,116.0     1,520.1     205    ₹82,833       +36.2%      +3.5%     
  BERGEPAINT  6      CON DUR    01-Oct-19   350.7       461.5       647    ₹71,648       +31.6%      +4.1%     
  LALPATHLAB  10     HEALTH     01-Oct-19   634.6       830.1       357    ₹69,795       +30.8%      +4.0%     
  GUJGASLTD   1      OIL&GAS    02-Dec-19   206.7       269.8       1148   ₹72,474       +30.5%      +2.3%     
  IGL         3      OIL&GAS    02-Dec-19   185.5       227.3       1279   ₹53,440       +22.5%      +6.7%     
  NESTLEIND   14     FMCG       01-Oct-19   636.8       761.7       356    ₹44,454       +19.6%      +6.2%     
  MANAPPURAM  19     FIN SVC    02-Dec-19   139.6       163.8       1699   ₹41,033       +17.3%      +1.0%     
  HDFCAMC     11     FIN SVC    01-Oct-19   1,208.6     1,354.2     187    ₹27,219       +12.0%      -3.1%     
  WHIRLPOOL   20     CON DUR    02-Dec-19   2,121.8     2,361.3     111    ₹26,584       +11.3%      -0.1%     
  NAM-INDIA   23     FIN SVC    02-Dec-19   290.5       285.9       816    ₹-3,728       -1.6%       -1.8%     

  AFTER: Invested ₹4,813,708 | Cash ₹153,781 | Total ₹4,967,490 | Positions 17/20 | Slot ₹248,448

========================================================================
  REBALANCE #32  —  01 Apr 2020
  NAV: ₹4,013,305  |  Slot: ₹200,665  |  Cash: ₹153,781
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, DIVISLAB

  [REGIME OFF] Nifty 200 4,275.7 < EMA200 5,815.6 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  1      HEALTH     01-Aug-18   6,983.6     14,325.7    30     ₹220,261      +105.1%     +6.4%     
  BATAINDIA   64     CON DUR    01-Aug-18   878.9       1,116.5     240    ₹57,003       +27.0% ⚠    -9.2%     
  IPCALAB     2      HEALTH     03-Feb-20   583.3       686.1       425    ₹43,715       +17.6%      +4.0%     
  NESTLEIND   3      FMCG       01-Oct-19   636.8       731.5       356    ₹33,700       +14.9%      +3.6%     
  BERGEPAINT  7      CON DUR    01-Oct-19   350.7       392.5       647    ₹27,029       +11.9%      +0.5%     
  PIIND       21     MFG        03-Jun-19   1,116.0     1,175.0     205    ₹12,098       +5.3%       -3.7%     
  LALPATHLAB  12     HEALTH     01-Oct-19   634.6       664.5       357    ₹10,682       +4.7%       -5.6%     
  GUJGASLTD   9      OIL&GAS    02-Dec-19   206.7       216.2       1148   ₹10,948       +4.6%       -6.3%     
  IGL         13     OIL&GAS    02-Dec-19   185.5       174.2       1279   ₹-14,486      -6.1%       +2.5%     
  COROMANDEL  11     MFG        03-Feb-20   585.9       497.7       424    ₹-37,424      -15.1%      -3.2%     
  WHIRLPOOL   30     CON DUR    02-Dec-19   2,121.8     1,742.9     111    ₹-42,057      -17.9%      -8.2%     
  HDFCAMC     20     FIN SVC    01-Oct-19   1,208.6     962.5       187    ₹-46,022      -20.4%      -9.0%     
  NAM-INDIA   18     FIN SVC    02-Dec-19   290.5       215.1       816    ₹-61,467      -25.9%      -6.1%     
  JUBLFOOD    27     CONSUMP    03-Feb-20   385.8       273.9       643    ₹-71,980      -29.0%      -5.2%     
  SRF         28     MFG        03-Feb-20   741.4       520.3       335    ₹-74,059      -29.8%      -14.5%    
  COFORGE     34     IT         03-Feb-20   349.4       216.9       711    ₹-94,228      -37.9%      -10.1%    
  MANAPPURAM  61     FIN SVC    02-Dec-19   139.6       83.7        1699   ₹-95,045      -40.1%      -16.9%    
  ⚠  WAZ < 0 (momentum below universe mean): BATAINDIA

  AFTER: Invested ₹3,859,523 | Cash ₹153,781 | Total ₹4,013,305 | Positions 17/20 | Slot ₹200,665

========================================================================
  REBALANCE #33  —  01 Jun 2020
  NAV: ₹4,637,462  |  Slot: ₹231,873  |  Cash: ₹153,781
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, DIVISLAB, SYNGENE, TORNTPHARM, AUROPHARMA, SUNPHARMA

  [REGIME OFF] Nifty 200 5,087.4 < EMA200 5,490.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  1      HEALTH     01-Aug-18   6,983.6     15,437.7    30     ₹253,622      +121.1%     -0.9%     
  BATAINDIA   —      CON DUR    01-Aug-18   878.9       1,302.1     240    ₹101,563      +48.1%      +5.7%     
  PIIND       16     MFG        03-Jun-19   1,116.0     1,552.3     205    ₹89,442       +39.1%      +4.1%     
  IPCALAB     7      HEALTH     03-Feb-20   583.3       749.6       425    ₹70,669       +28.5%      -2.1%     
  NESTLEIND   5      FMCG       01-Oct-19   636.8       802.9       356    ₹59,133       +26.1%      +2.0%     
  LALPATHLAB  23     HEALTH     01-Oct-19   634.6       728.6       357    ₹33,558       +14.8%      -1.4%     
  BERGEPAINT  21     CON DUR    01-Oct-19   350.7       397.9       647    ₹30,497       +13.4%      +4.6%     
  GUJGASLTD   26     OIL&GAS    02-Dec-19   206.7       233.9       1148   ₹31,310       +13.2%      +0.9%     
  IGL         12     OIL&GAS    02-Dec-19   185.5       209.8       1279   ₹31,140       +13.1%      +2.0%     
  COROMANDEL  6      MFG        03-Feb-20   585.9       605.6       424    ₹8,345        +3.4%       +4.7%     
  SRF         35     MFG        03-Feb-20   741.4       726.6       335    ₹-4,976       -2.0%       +6.3%     
  HDFCAMC     25     FIN SVC    01-Oct-19   1,208.6     1,168.9     187    ₹-7,432       -3.3%       +5.5%     
  WHIRLPOOL   24     CON DUR    02-Dec-19   2,121.8     1,998.5     111    ₹-13,690      -5.8%       +6.4%     
  JUBLFOOD    —      CONSUMP    03-Feb-20   385.8       331.8       643    ₹-34,696      -14.0%      +4.1%     
  MANAPPURAM  54     FIN SVC    02-Dec-19   139.6       117.8       1699   ₹-37,131      -15.7%      +9.3%     
  COFORGE     46     IT         03-Feb-20   349.4       272.4       711    ₹-54,731      -22.0% ⚠    +4.4%     
  NAM-INDIA   —      FIN SVC    02-Dec-19   290.5       224.5       816    ₹-53,800      -22.7%      +10.0%    
  ⚠  WAZ < 0 (momentum below universe mean): COFORGE

  AFTER: Invested ₹4,483,681 | Cash ₹153,781 | Total ₹4,637,462 | Positions 17/20 | Slot ₹231,873

========================================================================
  REBALANCE #34  —  03 Aug 2020
  NAV: ₹4,970,849  |  Slot: ₹248,542  |  Cash: ₹153,781
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, TORNTPHARM, DIVISLAB, AJANTPHARM

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BATAINDIA   —      CON DUR    01-Aug-18   878.9       1,180.7     240    ₹72,410       +34.3%    733d  
  NESTLEIND   51     FMCG       01-Oct-19   636.8       775.0       356    ₹49,194       +21.7%    307d  
  COFORGE     16     IT         03-Feb-20   349.4       362.1       711    ₹9,039        +3.6%     182d  
  MANAPPURAM  37     FIN SVC    02-Dec-19   139.6       142.6       1699   ₹5,107        +2.2%     245d  
  SRF         48     MFG        03-Feb-20   741.4       751.3       335    ₹3,313        +1.3%     182d  
  WHIRLPOOL   46     CON DUR    02-Dec-19   2,121.8     2,080.9     111    ₹-4,543       -1.9%     245d  
  IGL         88     OIL&GAS    02-Dec-19   185.5       175.4       1279   ₹-12,861      -5.4%     245d  
  HDFCAMC     77     FIN SVC    01-Oct-19   1,208.6     1,080.2     187    ₹-24,019      -10.6%    307d  
  JUBLFOOD    —      CONSUMP    03-Feb-20   385.8       344.3       643    ₹-26,713      -10.8%    182d  
  NAM-INDIA   —      FIN SVC    02-Dec-19   290.5       220.6       816    ₹-57,025      -24.1%    245d  

  ENTRIES (10)
  [52w filter blocked 1: ENDURANCE(-22.9%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JUBILANT    1      CONSUMP    3.766    0.86   +100.3%   +127.5%   618.2       402    ₹248,509      +16.9%    
  SYNGENE     3      HEALTH     2.820    0.47   +48.5%    +51.9%    473.6       524    ₹248,149      +7.9%     
  MUTHOOTFIN  4      FIN SVC    2.678    1.15   +115.6%   +64.0%    1,172.9     211    ₹247,483      +4.0%     
  MPHASIS     5      IT         2.558    0.56   +30.2%    +66.8%    1,024.9     242    ₹248,036      +9.8%     
  BALKRISIND  6      MFG        2.356    0.97   +81.9%    +47.5%    1,249.0     199    ₹248,541      +3.8%     
  WIPRO       10     IT         1.998    0.63   +6.9%     +52.9%    129.8       1915   ₹248,510      +7.9%     
  HCLTECH     11     IT         1.968    0.73   +41.1%    +36.1%    559.9       443    ₹248,023      +8.0%     
  INFY        16     IT         1.681    0.78   +24.3%    +44.0%    815.8       304    ₹247,995      +7.1%     
  BRITANNIA   18     FMCG       1.598    0.78   +40.7%    +26.8%    3,411.0     72     ₹245,594      +0.6%     
  ATGL        20     OIL&GAS    1.518    0.79   -3.8%     +60.8%    156.2       1590   ₹248,424      +3.5%     

  HOLDS (7)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  35     HEALTH     01-Aug-18   6,983.6     14,688.6    30     ₹231,148      +110.3%     +4.1%     
  PIIND       15     MFG        03-Jun-19   1,116.0     1,809.4     205    ₹142,146      +62.1%      +6.6%     
  IPCALAB     7      HEALTH     03-Feb-20   583.3       927.0       425    ₹146,067      +58.9%      +7.8%     
  GUJGASLTD   9      OIL&GAS    02-Dec-19   206.7       291.7       1148   ₹97,653       +41.2%      +3.6%     
  LALPATHLAB  14     HEALTH     01-Oct-19   634.6       892.7       357    ₹92,124       +40.7%      +0.5%     
  COROMANDEL  2      MFG        03-Feb-20   585.9       737.5       424    ₹64,262       +25.9%      +1.3%     
  BERGEPAINT  20     CON DUR    01-Oct-19   350.7       426.3       647    ₹48,909       +21.6%      +1.5%     

  AFTER: Invested ₹4,926,919 | Cash ₹40,986 | Total ₹4,967,905 | Positions 17/20 | Slot ₹248,542

========================================================================
  REBALANCE #35  —  01 Oct 2020
  NAV: ₹5,332,974  |  Slot: ₹266,649  |  Cash: ₹40,986
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, APOLLOHOSP, CIPLA, SANOFI, TORNTPHARM

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LALPATHLAB  28     HEALTH     01-Oct-19   634.6       891.4       357    ₹91,680       +40.5%    366d  
  BRITANNIA   37     FMCG       03-Aug-20   3,411.0     3,515.2     72     ₹7,498        +3.1%     59d   
  MUTHOOTFIN  32     FIN SVC    03-Aug-20   1,172.9     1,055.2     211    ₹-24,827      -10.0%    59d   
  JUBILANT    —      OTHER      03-Aug-20   618.2       527.1       402    ₹-36,618      -14.7%    59d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIENT    2      METAL      3.358    1.15   +103.2%   +87.7%    307.3       867    ₹266,400      +8.0%     
  DIVISLAB    3      HEALTH     3.201    0.60   +84.3%    +41.8%    2,973.1     89     ₹264,603      -1.9%     
  NAVINFLUOR  5      MFG        2.910    0.76   +180.4%   +24.5%    2,095.0     127    ₹266,065      +3.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  23     HEALTH     01-Aug-18   6,983.6     15,256.9    30     ₹248,197      +118.5%     +0.2%     
  IPCALAB     2      HEALTH     03-Feb-20   583.3       1,106.3     425    ₹222,289      +89.7%      +7.1%     
  PIIND       19     MFG        03-Jun-19   1,116.0     1,913.5     205    ₹163,479      +71.5%      +0.2%     
  GUJGASLTD   26     OIL&GAS    02-Dec-19   206.7       297.7       1148   ₹104,550      +44.1%      +1.9%     
  BERGEPAINT  21     CON DUR    01-Oct-19   350.7       479.7       647    ₹83,450       +36.8%      +2.9%     
  COROMANDEL  14     MFG        03-Feb-20   585.9       749.7       424    ₹69,414       +27.9%      +0.9%     
  ATGL        27     OIL&GAS    03-Aug-20   156.2       193.0       1590   ₹58,495       +23.5%      +2.2%     
  MPHASIS     8      IT         03-Aug-20   1,024.9     1,215.0     242    ₹46,006       +18.5%      +4.9%     
  SYNGENE     7      HEALTH     03-Aug-20   473.6       554.2       524    ₹42,257       +17.0%      +3.6%     
  HCLTECH     6      IT         03-Aug-20   559.9       646.5       443    ₹38,386       +15.5%      +3.5%     
  BALKRISIND  17     MFG        03-Aug-20   1,249.0     1,397.6     199    ₹29,575       +11.9%      +6.1%     
  WIPRO       10     IT         03-Aug-20   129.8       144.3       1915   ₹27,803       +11.2%      +3.1%     
  INFY        16     IT         03-Aug-20   815.8       867.6       304    ₹15,744       +6.3%       +3.0%     

  AFTER: Invested ₹5,083,183 | Cash ₹248,845 | Total ₹5,332,028 | Positions 16/20 | Slot ₹266,649

========================================================================
  REBALANCE #36  —  01 Dec 2020
  NAV: ₹6,059,372  |  Slot: ₹302,969  |  Cash: ₹248,845
========================================================================
  [SECTOR CAP≤4] dropped: MRF, TATACHEM, LALPATHLAB, SRF

  EXITS (1)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ABBOTINDIA  87     HEALTH     01-Aug-18   6,983.6     14,338.5    30     ₹220,645      +105.3%   853d  

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  APOLLOHOSP  3      HEALTH     2.631    -0.09  +71.1%    +47.7%    2,431.2     124    ₹301,470      +8.8%     
  ADANIENSOL  16     ENERGY     1.772    -0.16  +35.4%    +43.8%    379.0       799    ₹302,781      +8.4%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       360.4       1590   ₹324,538      +130.6%     +22.2%    
  PIIND       13     MFG        03-Jun-19   1,116.0     2,294.6     205    ₹241,616      +105.6%     +1.8%     
  IPCALAB     11     HEALTH     03-Feb-20   583.3       1,120.5     425    ₹228,304      +92.1%      +5.0%     
  GUJGASLTD   24     OIL&GAS    02-Dec-19   206.7       326.4       1148   ₹137,481      +58.0%      +5.8%     
  BERGEPAINT  23     CON DUR    01-Oct-19   350.7       532.5       647    ₹117,587      +51.8%      +2.4%     
  ADANIENT    4      METAL      01-Oct-20   307.3       420.7       867    ₹98,338       +36.9%      +10.8%    
  COROMANDEL  17     MFG        03-Feb-20   585.9       778.3       424    ₹81,542       +32.8%      +6.0%     
  BALKRISIND  12     MFG        03-Aug-20   1,249.0     1,584.9     199    ₹66,849       +26.9%      +5.6%     
  NAVINFLUOR  2      MFG        01-Oct-20   2,095.0     2,638.5     127    ₹69,027       +25.9%      +6.2%     
  WIPRO       9      IT         03-Aug-20   129.8       162.6       1915   ₹62,933       +25.3%      +1.5%     
  INFY        8      IT         03-Aug-20   815.8       980.5       304    ₹50,064       +20.2%      +2.3%     
  HCLTECH     18     IT         03-Aug-20   559.9       666.4       443    ₹47,193       +19.0%      +0.6%     
  SYNGENE     6      HEALTH     03-Aug-20   473.6       563.1       524    ₹46,901       +18.9%      +1.1%     
  DIVISLAB    5      HEALTH     01-Oct-20   2,973.1     3,512.3     89     ₹47,988       +18.1%      +5.7%     
  MPHASIS     22     IT         03-Aug-20   1,024.9     1,173.6     242    ₹35,983       +14.5%      -1.1%     

  AFTER: Invested ₹5,984,623 | Cash ₹74,031 | Total ₹6,058,654 | Positions 17/20 | Slot ₹302,969

========================================================================
  REBALANCE #37  —  01 Feb 2021
  NAV: ₹6,330,943  |  Slot: ₹316,547  |  Cash: ₹74,031
========================================================================
  [SECTOR CAP≤4] dropped: MRF

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PIIND       90     MFG        03-Jun-19   1,116.0     1,966.5     205    ₹174,358      +76.2%    609d  
  BERGEPAINT  44     CON DUR    01-Oct-19   350.7       585.6       647    ₹151,934      +67.0%    489d  
  GUJGASLTD   50     OIL&GAS    02-Dec-19   206.7       344.2       1148   ₹157,877      +66.5%    427d  
  IPCALAB     85     HEALTH     03-Feb-20   583.3       924.7       425    ₹145,119      +58.5%    364d  
  COROMANDEL  42     MFG        03-Feb-20   585.9       786.2       424    ₹84,930       +34.2%    364d  
  HCLTECH     32     IT         03-Aug-20   559.9       744.7       443    ₹81,872       +33.0%    182d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TATACHEM    3      MFG        2.646    0.12   +55.0%    +53.3%    453.6       697    ₹316,189      -1.5%     
  CROMPTON    4      CON DUR    2.379    -0.04  +62.7%    +40.6%    400.6       790    ₹316,478      +3.1%     
  HINDZINC    5      METAL      2.297    0.12   +75.0%    +42.0%    188.7       1677   ₹316,393      +5.7%     
  BAJAJ-AUTO  7      AUTO       2.091    0.00   +40.5%    +42.5%    3,548.3     89     ₹315,798      +8.6%     
  LTTS        9      IT         1.944    -0.13  +48.4%    +48.5%    2,321.4     136    ₹315,717      +1.8%     
  LT          10     INFRA      1.926    0.16   +12.2%    +58.9%    1,361.6     232    ₹315,886      +7.5%     
  HONAUT      13     MFG        1.749    -0.11  +48.0%    +43.6%    40,417.2    7      ₹282,920      +5.0%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       388.5       1590   ₹369,223      +148.6%     +5.3%     
  ADANIENT    2      METAL      01-Oct-20   307.3       535.7       867    ₹198,017      +74.3%      +4.7%     
  WIPRO       8      IT         03-Aug-20   129.8       194.7       1915   ₹124,363      +50.0%      -0.9%     
  MPHASIS     27     IT         03-Aug-20   1,024.9     1,370.0     242    ₹83,506       +33.7%      -3.1%     
  INFY        12     IT         03-Aug-20   815.8       1,086.5     304    ₹82,297       +33.2%      -2.6%     
  ADANIENSOL  6      ENERGY     01-Dec-20   379.0       481.5       799    ₹81,937       +27.1%      +6.6%     
  BALKRISIND  29     MFG        03-Aug-20   1,249.0     1,570.8     199    ₹64,047       +25.8%      -0.9%     
  SYNGENE     16     HEALTH     03-Aug-20   473.6       556.1       524    ₹43,243       +17.4%      -5.5%     
  DIVISLAB    11     HEALTH     01-Oct-20   2,973.1     3,359.8     89     ₹34,421       +13.0%      -3.7%     
  NAVINFLUOR  26     MFG        01-Oct-20   2,095.0     2,307.2     127    ₹26,949       +10.1%      -6.4%     
  APOLLOHOSP  18     HEALTH     01-Dec-20   2,431.2     2,629.1     124    ₹24,541       +8.1%       +3.7%     

  AFTER: Invested ₹6,202,899 | Cash ₹125,457 | Total ₹6,328,355 | Positions 18/20 | Slot ₹316,547

========================================================================
  REBALANCE #38  —  01 Apr 2021
  NAV: ₹8,671,782  |  Slot: ₹433,589  |  Cash: ₹125,457
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  WIPRO       33     IT         03-Aug-20   129.8       192.4       1915   ₹119,851      +48.2%    241d  
  NAVINFLUOR  47     MFG        01-Oct-20   2,095.0     2,731.4     127    ₹80,825       +30.4%    182d  
  BALKRISIND  48     MFG        03-Aug-20   1,249.0     1,615.7     199    ₹72,976       +29.4%    241d  
  DIVISLAB    70     HEALTH     01-Oct-20   2,973.1     3,507.8     89     ₹47,587       +18.0%    182d  
  SYNGENE     54     HEALTH     03-Aug-20   473.6       552.4       524    ₹41,324       +16.7%    241d  
  LT          38     INFRA      01-Feb-21   1,361.6     1,357.6     232    ₹-927         -0.3%     59d   
  HINDZINC    28     METAL      01-Feb-21   188.7       183.1       1677   ₹-9,314       -2.9%     59d   
  CROMPTON    44     CON DUR    01-Feb-21   400.6       378.3       790    ₹-17,631      -5.6%     59d   
  BAJAJ-AUTO  45     AUTO       01-Feb-21   3,548.3     3,227.8     89     ₹-28,524      -9.0%     59d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  DEEPAKNTR   5      MFG        2.783    0.05   +338.0%   +77.5%    1,618.8     267    ₹432,218      +7.1%     
  GRASIM      6      INFRA      2.606    0.08   +217.5%   +55.7%    1,412.7     306    ₹432,291      +5.6%     
  TATAELXSI   7      IT         2.572    0.04   +341.6%   +49.7%    2,599.1     166    ₹431,449      +2.7%     
  DIXON       8      CON DUR    2.507    0.07   +429.0%   +34.2%    3,580.2     121    ₹433,210      -5.8%     
  LAURUSLABS  9      HEALTH     1.956    0.04   +462.4%   +4.5%     359.2       1207   ₹433,572      +2.0%     
  GUJGASLTD   10     OIL&GAS    1.826    0.09   +141.0%   +43.3%    524.7       826    ₹433,375      +5.5%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       1,058.9     1590   ₹1,435,223    +577.7%     +31.6%    
  ADANIENT    2      METAL      01-Oct-20   307.3       1,104.0     867    ₹690,789      +259.3%     +16.5%    
  ADANIENSOL  3      ENERGY     01-Dec-20   379.0       999.2       799    ₹495,580      +163.7%     +21.3%    
  TATACHEM    4      MFG        01-Feb-21   453.6       715.4       697    ₹182,430      +57.7%      +5.5%     
  MPHASIS     27     IT         03-Aug-20   1,024.9     1,565.1     242    ₹130,714      +52.7%      +3.6%     
  INFY        25     IT         03-Aug-20   815.8       1,193.6     304    ₹114,857      +46.3%      +2.5%     
  APOLLOHOSP  19     HEALTH     01-Dec-20   2,431.2     2,856.7     124    ₹52,760       +17.5%      -1.0%     
  HONAUT      24     MFG        01-Feb-21   40,417.2    45,601.3    7      ₹36,289       +12.8%      +0.5%     
  LTTS        26     IT         01-Feb-21   2,321.4     2,550.6     136    ₹31,161       +9.9%       +3.8%     

  AFTER: Invested ₹8,295,849 | Cash ₹372,851 | Total ₹8,668,700 | Positions 15/20 | Slot ₹433,589

========================================================================
  REBALANCE #39  —  01 Jun 2021
  NAV: ₹10,360,192  |  Slot: ₹518,010  |  Cash: ₹372,851
========================================================================
  [SECTOR CAP≤4] dropped: POLYCAB

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIENT    —      METAL      01-Oct-20   307.3       1,412.2     867    ₹957,966      +359.6%   243d  
  TATACHEM    —      MFG        01-Feb-21   453.6       648.6       697    ₹135,903      +43.0%    120d  
  APOLLOHOSP  —      HEALTH     01-Dec-20   2,431.2     3,198.0     124    ₹95,080       +31.5%    182d  
  LTTS        —      IT         01-Feb-21   2,321.4     2,519.1     136    ₹26,877       +8.5%     120d  
  HONAUT      —      MFG        01-Feb-21   40,417.2    41,057.7    7      ₹4,484        +1.6%     120d  
  GRASIM      —      INFRA      01-Apr-21   1,412.7     1,403.1     306    ₹-2,957       -0.7%     61d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JSWSTEEL    3      METAL      3.491    0.47   +281.0%   +70.2%    657.4       787    ₹517,396      +0.4%     
  TATASTEEL   6      METAL      2.684    0.79   +282.6%   +51.3%    92.9        5578   ₹517,967      +0.6%     
  BSE         7      FIN SVC    2.506    0.46   +155.2%   +60.8%    97.3        5323   ₹517,979      +16.9%    
  SAIL        8      METAL      2.487    0.77   +308.4%   +69.7%    104.5       4955   ₹517,948      -0.8%     
  DALBHARAT   9      MFG        1.948    0.70   +227.2%   +25.2%    1,770.4     292    ₹516,959      +3.9%     
  ZYDUSLIFE   11     HEALTH     1.906    0.33   +79.8%    +42.9%    597.4       867    ₹517,988      +2.1%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       1,438.2     1590   ₹2,038,345    +820.5%     +10.6%    
  ADANIENSOL  2      ENERGY     01-Dec-20   379.0       1,499.4     799    ₹895,279      +295.7%     +13.2%    
  MPHASIS     29     IT         03-Aug-20   1,024.9     1,746.1     242    ₹174,531      +70.4%      +5.5%     
  INFY        27     IT         03-Aug-20   815.8       1,208.2     304    ₹119,302      +48.1%      +2.4%     
  LAURUSLABS  4      HEALTH     01-Apr-21   359.2       524.5       1207   ₹199,481      +46.0%      +7.1%     
  TATAELXSI   5      IT         01-Apr-21   2,599.1     3,384.3     166    ₹130,337      +30.2%      +1.8%     
  DIXON       18     CON DUR    01-Apr-21   3,580.2     4,098.5     121    ₹62,705       +14.5%      +3.4%     
  DEEPAKNTR   10     MFG        01-Apr-21   1,618.8     1,730.5     267    ₹29,816       +6.9%       -0.6%     
  GUJGASLTD   37     OIL&GAS    01-Apr-21   524.7       517.6       826    ₹-5,854       -1.4%       +3.1%     

  AFTER: Invested ₹9,961,239 | Cash ₹395,265 | Total ₹10,356,504 | Positions 15/20 | Slot ₹518,010

========================================================================
  REBALANCE #40  —  02 Aug 2021
  NAV: ₹10,300,313  |  Slot: ₹515,016  |  Cash: ₹395,265
========================================================================
  [SECTOR CAP≤4] dropped: POLYCAB, PIDILITIND

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIENSOL  53     ENERGY     01-Dec-20   379.0       908.8       799    ₹423,350      +139.8%   244d  
  DIXON       41     CON DUR    01-Apr-21   3,580.2     4,339.1     121    ₹91,826       +21.2%    123d  
  ZYDUSLIFE   79     HEALTH     01-Jun-21   597.4       574.2       867    ₹-20,169      -3.9%     62d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COFORGE     3      IT         2.715    0.59   +177.1%   +77.8%    969.2       531    ₹514,645      +10.9%    
  SRF         8      MFG        2.213    0.66   +130.0%   +37.9%    1,773.4     290    ₹514,298      +14.7%    
  ABBOTINDIA  10     HEALTH     2.087    0.02   +37.1%    +32.3%    18,684.8    27     ₹504,490      +11.5%    
  AMBUJACEM   11     INFRA      2.031    0.64   +108.0%   +34.2%    402.3       1280   ₹514,973      +7.3%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        28     OIL&GAS    03-Aug-20   156.2       887.1       1590   ₹1,161,986    +467.7%     -5.7%     
  MPHASIS     9      IT         03-Aug-20   1,024.9     2,352.2     242    ₹321,195      +129.5%     +8.5%     
  LAURUSLABS  5      HEALTH     01-Apr-21   359.2       644.6       1207   ₹344,518      +79.5%      +1.3%     
  INFY        16     IT         03-Aug-20   815.8       1,421.0     304    ₹184,000      +74.2%      +3.3%     
  TATAELXSI   4      IT         01-Apr-21   2,599.1     4,010.4     166    ₹234,274      +54.3%      +0.6%     
  GUJGASLTD   7      OIL&GAS    01-Apr-21   524.7       722.7       826    ₹163,604      +37.8%      +8.3%     
  BSE         1      FIN SVC    01-Jun-21   97.3        133.3       5323   ₹191,431      +37.0%      +10.0%    
  TATASTEEL   2      METAL      01-Jun-21   92.9        121.6       5578   ₹160,300      +30.9%      +8.9%     
  DEEPAKNTR   24     MFG        01-Apr-21   1,618.8     2,032.0     267    ₹110,330      +25.5%      +6.9%     
  DALBHARAT   6      MFG        01-Jun-21   1,770.4     2,108.0     292    ₹98,576       +19.1%      +1.3%     
  SAIL        19     METAL      01-Jun-21   104.5       120.3       4955   ₹78,205       +15.1%      +6.6%     
  JSWSTEEL    13     METAL      01-Jun-21   657.4       713.8       787    ₹44,327       +8.6%       +5.1%     

  AFTER: Invested ₹10,204,468 | Cash ₹93,413 | Total ₹10,297,881 | Positions 16/20 | Slot ₹515,016

========================================================================
  REBALANCE #41  —  01 Oct 2021
  NAV: ₹11,370,176  |  Slot: ₹568,509  |  Cash: ₹93,413
========================================================================
  [SECTOR CAP≤4] dropped: TECHM, HCLTECH

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LAURUSLABS  74     HEALTH     01-Apr-21   359.2       609.9       1207   ₹302,552      +69.8%    183d  
  TATASTEEL   12     METAL      01-Jun-21   92.9        111.9       5578   ₹106,329      +20.5%    122d  
  GUJGASLTD   83     OIL&GAS    01-Apr-21   524.7       589.5       826    ₹53,534       +12.4%    183d  
  ABBOTINDIA  —      OTHER      02-Aug-21   18,684.8    20,842.3    27     ₹58,252       +11.5%    60d   
  SAIL        48     METAL      01-Jun-21   104.5       100.6       4955   ₹-19,560      -3.8%     122d  

  ENTRIES (5)
  [52w filter blocked 1: ADANIENSOL(-20.7%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IRCTC       3      PSE        3.393    0.84   +181.5%   +86.7%    725.8       783    ₹568,283      +6.9%     
  BAJAJHLDNG  6      FIN SVC    2.420    0.53   +99.7%    +34.8%    4,433.3     128    ₹567,461      +4.4%     
  PRESTIGE    7      REALTY     2.362    0.87   +97.5%    +69.3%    477.2       1191   ₹568,353      +8.8%     
  POLYCAB     9      MFG        2.115    0.56   +184.7%   +17.1%    2,283.5     248    ₹566,308      +0.5%     
  HAVELLS     11     CON DUR    1.930    0.95   +107.5%   +37.4%    1,315.9     432    ₹568,459      -1.8%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        2      OIL&GAS    03-Aug-20   156.2       1,421.5     1590   ₹2,011,788    +809.8%     +3.2%     
  MPHASIS     9      IT         03-Aug-20   1,024.9     2,770.0     242    ₹422,311      +170.3%     -1.8%     
  TATAELXSI   4      IT         01-Apr-21   2,599.1     5,491.9     166    ₹480,207      +111.3%     +6.9%     
  INFY        40     IT         03-Aug-20   815.8       1,450.3     304    ₹192,896      +77.8%      -2.0%     
  DEEPAKNTR   11     MFG        01-Apr-21   1,618.8     2,348.6     267    ₹194,858      +45.1%      +0.2%     
  BSE         15     FIN SVC    01-Jun-21   97.3        129.4       5323   ₹170,618      +32.9%      +1.3%     
  SRF         5      MFG        02-Aug-21   1,773.4     2,186.9     290    ₹119,908      +23.3%      +3.6%     
  DALBHARAT   21     MFG        01-Jun-21   1,770.4     2,064.4     292    ₹85,852       +16.6%      -1.7%     
  COFORGE     29     IT         02-Aug-21   969.2       997.8       531    ₹15,190       +3.0%       -0.5%     
  JSWSTEEL    44     METAL      01-Jun-21   657.4       644.5       787    ₹-10,175      -2.0%       -0.3%     
  AMBUJACEM   18     INFRA      02-Aug-21   402.3       387.1       1280   ₹-19,522      -3.8%       -3.1%     

  AFTER: Invested ₹11,207,166 | Cash ₹159,640 | Total ₹11,366,805 | Positions 16/20 | Slot ₹568,509

========================================================================
  REBALANCE #42  —  01 Dec 2021
  NAV: ₹11,758,703  |  Slot: ₹587,935  |  Cash: ₹159,640
========================================================================
  [SECTOR CAP≤4] dropped: TECHM

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DEEPAKNTR   24     MFG        01-Apr-21   1,618.8     2,120.9     267    ₹134,058      +31.0%    244d  
  DALBHARAT   66     MFG        01-Jun-21   1,770.4     1,797.2     292    ₹7,809        +1.5%     183d  
  JSWSTEEL    47     METAL      01-Jun-21   657.4       609.1       787    ₹-38,047      -7.4%     183d  
  PRESTIGE    23     REALTY     01-Oct-21   477.2       439.3       1191   ₹-45,142      -7.9%     61d   
  AMBUJACEM   81     INFRA      02-Aug-21   402.3       357.4       1280   ₹-57,520      -11.2%    121d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIENSOL  2      ENERGY     3.205    1.04   +373.4%   +19.4%    1,797.2     327    ₹587,668      -4.2%     
  DIXON       8      CON DUR    2.367    0.90   +137.1%   +23.4%    5,067.9     116    ₹587,877      -2.5%     
  ESCORTS     9      MFG        2.174    0.80   +32.6%    +35.3%    1,800.8     326    ₹587,052      +7.5%     
  ONGC        10     OIL&GAS    2.170    1.16   +99.2%    +24.7%    109.7       5361   ₹587,858      -3.8%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        4      OIL&GAS    03-Aug-20   156.2       1,623.7     1590   ₹2,333,292    +939.2%     +1.9%     
  MPHASIS     15     IT         03-Aug-20   1,024.9     2,756.6     242    ₹419,066      +169.0%     -6.0%     
  TATAELXSI   1      IT         01-Apr-21   2,599.1     5,587.5     166    ₹496,075      +115.0%     -3.0%     
  INFY        35     IT         03-Aug-20   815.8       1,506.9     304    ₹210,093      +84.7%      -0.7%     
  BSE         5      FIN SVC    01-Jun-21   97.3        175.3       5323   ₹415,158      +80.1%      +9.5%     
  SRF         30     MFG        02-Aug-21   1,773.4     1,988.5     290    ₹62,365       +12.1%      -5.0%     
  BAJAJHLDNG  8      FIN SVC    01-Oct-21   4,433.3     4,853.5     128    ₹53,789       +9.5%       +4.8%     
  IRCTC       6      PSE        01-Oct-21   725.8       776.7       783    ₹39,869       +7.0%       -4.6%     
  COFORGE     26     IT         02-Aug-21   969.2       1,019.2     531    ₹26,530       +5.2%       -0.2%     
  HAVELLS     37     CON DUR    01-Oct-21   1,315.9     1,320.8     432    ₹2,126        +0.4%       +0.4%     
  POLYCAB     9      MFG        01-Oct-21   2,283.5     2,268.6     248    ₹-3,692       -0.7%       -1.5%     

  AFTER: Invested ₹11,398,462 | Cash ₹357,450 | Total ₹11,755,912 | Positions 15/20 | Slot ₹587,935

========================================================================
  REBALANCE #43  —  01 Feb 2022
  NAV: ₹12,807,439  |  Slot: ₹640,372  |  Cash: ₹357,450
========================================================================
  [SECTOR CAP≤4] dropped: SIEMENS

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  IRCTC       12     PSE        01-Oct-21   725.8       819.7       783    ₹73,537       +12.9%    123d  
  HAVELLS     84     CON DUR    01-Oct-21   1,315.9     1,152.7     432    ₹-70,486      -12.4%    123d  
  DIXON       74     CON DUR    01-Dec-21   5,067.9     4,429.2     116    ₹-74,090      -12.6%    62d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  POWERGRID   8      ENERGY     2.311    0.70   +60.2%    +17.4%    130.0       4924   ₹640,272      +1.4%     
  LT          11     INFRA      2.023    1.12   +47.9%    +10.9%    1,891.8     338    ₹639,419      +2.3%     
  TORNTPOWER  12     ENERGY     1.965    0.79   +78.1%    +8.0%     488.3       1311   ₹640,217      -1.9%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        2      OIL&GAS    03-Aug-20   156.2       1,861.9     1590   ₹2,712,032    +1091.7%    +3.3%     
  MPHASIS     29     IT         03-Aug-20   1,024.9     2,880.5     242    ₹449,048      +181.0%     +0.4%     
  TATAELXSI   3      IT         01-Apr-21   2,599.1     7,089.3     166    ₹745,382      +172.8%     +10.5%    
  BSE         1      FIN SVC    01-Jun-21   97.3        211.4       5323   ₹607,248      +117.2%     +0.2%     
  INFY        25     IT         03-Aug-20   815.8       1,557.1     304    ₹225,359      +90.9%      -1.4%     
  SRF         5      MFG        02-Aug-21   1,773.4     2,413.0     290    ₹185,464      +36.1%      -0.3%     
  ONGC        6      OIL&GAS    01-Dec-21   109.7       131.8       5361   ₹118,605      +20.2%      +5.2%     
  ADANIENSOL  4      ENERGY     01-Dec-21   1,797.2     1,986.9     327    ₹62,048       +10.6%      +1.7%     
  BAJAJHLDNG  10     FIN SVC    01-Oct-21   4,433.3     4,889.1     128    ₹58,340       +10.3%      -0.4%     
  POLYCAB     7      MFG        01-Oct-21   2,283.5     2,471.0     248    ₹46,489       +8.2%       +0.5%     
  ESCORTS     8      MFG        01-Dec-21   1,800.8     1,800.2     326    ₹-175         -0.0%       -0.5%     
  COFORGE     35     IT         02-Aug-21   969.2       907.8       531    ₹-32,602      -6.3%       -7.7%     

  AFTER: Invested ₹12,716,318 | Cash ₹88,841 | Total ₹12,805,159 | Positions 15/20 | Slot ₹640,372

========================================================================
  REBALANCE #44  —  01 Apr 2022
  NAV: ₹14,235,442  |  Slot: ₹711,772  |  Cash: ₹88,841
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BSE         1      FIN SVC    01-Jun-21   97.3        297.8       5323   ₹1,066,953    +206.0%   304d  
  SRF         4      MFG        02-Aug-21   1,773.4     2,590.2     290    ₹236,858      +46.1%    242d  
  TORNTPOWER  70     ENERGY     01-Feb-22   488.3       459.1       1311   ₹-38,321      -6.0%     59d   
  ESCORTS     73     MFG        01-Dec-21   1,800.8     1,654.2     326    ₹-47,783      -8.1%     121d  
  COFORGE     84     IT         02-Aug-21   969.2       838.9       531    ₹-69,206      -13.4%    242d  

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HAL         4      DEFENCE    2.747    0.92   +59.8%    +28.3%    723.8       983    ₹711,499      +7.9%     
  PERSISTENT  5      IT         2.707    0.96   +163.5%   -1.4%     2,292.8     310    ₹710,759      +4.7%     
  NTPC        7      ENERGY     2.261    0.85   +46.8%    +15.9%    126.0       5646   ₹711,664      +6.4%     
  BEL         11     DEFENCE    1.982    0.98   +84.2%    +4.0%     68.5        10383  ₹711,705      +3.3%     
  RELIANCE    12     OIL&GAS    1.975    1.06   +33.8%    +12.6%    1,202.9     591    ₹710,903      +5.3%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        6      OIL&GAS    03-Aug-20   156.2       2,246.2     1590   ₹3,323,048    +1337.7%    +16.1%    
  TATAELXSI   2      IT         01-Apr-21   2,599.1     8,464.4     166    ₹973,650      +225.7%     +12.8%    
  MPHASIS     18     IT         03-Aug-20   1,024.9     3,060.9     242    ₹492,708      +198.6%     +2.7%     
  INFY        28     IT         03-Aug-20   815.8       1,672.6     304    ₹260,486      +105.0%     +2.7%     
  ADANIENSOL  3      ENERGY     01-Dec-21   1,797.2     2,421.4     327    ₹204,146      +34.7%      +4.1%     
  ONGC        10     OIL&GAS    01-Dec-21   109.7       130.8       5361   ₹113,518      +19.3%      -1.3%     
  BAJAJHLDNG  14     FIN SVC    01-Oct-21   4,433.3     5,052.3     128    ₹79,231       +14.0%      +6.5%     
  POWERGRID   15     ENERGY     01-Feb-22   130.0       141.2       4924   ₹55,181       +8.6%       +6.2%     
  POLYCAB     25     MFG        01-Oct-21   2,283.5     2,369.9     248    ₹21,429       +3.8%       +2.3%     
  LT          54     INFRA      01-Feb-22   1,891.8     1,701.3     338    ₹-64,376      -10.1%      +1.3%     

  AFTER: Invested ₹13,780,441 | Cash ₹450,778 | Total ₹14,231,219 | Positions 15/20 | Slot ₹711,772

========================================================================
  REBALANCE #45  —  01 Jun 2022
  NAV: ₹13,916,110  |  Slot: ₹695,805  |  Cash: ₹450,778
========================================================================

  [REGIME OFF] Nifty 200 8,710.7 < EMA200 8,862.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        5      OIL&GAS    03-Aug-20   156.2       2,322.5     1590   ₹3,444,346    +1386.5%    -2.8%     
  TATAELXSI   2      IT         01-Apr-21   2,599.1     8,173.1     166    ₹925,283      +214.5%     +5.6%     
  MPHASIS     81     IT         03-Aug-20   1,024.9     2,315.3     242    ₹312,265      +125.9% ⚠   -1.7%     
  INFY        111    IT         03-Aug-20   815.8       1,312.9     304    ₹151,141      +60.9% ⚠    -0.8%     
  HAL         1      DEFENCE    01-Apr-22   723.8       899.0       983    ₹172,262      +24.2%      +9.7%     
  BEL         4      DEFENCE    01-Apr-22   68.5        78.4        10383  ₹102,570      +14.4%      +6.0%     
  POWERGRID   10     ENERGY     01-Feb-22   130.0       143.8       4924   ₹68,006       +10.6%      -0.3%     
  NTPC        7      ENERGY     01-Apr-22   126.0       138.3       5646   ₹68,951       +9.7%       +3.0%     
  ADANIENSOL  —      ENERGY     01-Dec-21   1,797.2     1,957.9     327    ₹52,565       +8.9%       -13.5%    
  BAJAJHLDNG  35     FIN SVC    01-Oct-21   4,433.3     4,726.9     128    ₹37,582       +6.6%       +0.4%     
  ONGC        49     OIL&GAS    01-Dec-21   109.7       116.7       5361   ₹37,512       +6.4%       -3.5%     
  POLYCAB     22     MFG        01-Oct-21   2,283.5     2,426.5     248    ₹35,461       +6.3%       +0.4%     
  RELIANCE    17     OIL&GAS    01-Apr-22   1,202.9     1,192.8     591    ₹-5,983       -0.8%       +1.5%     
  LT          —      INFRA      01-Feb-22   1,891.8     1,566.3     338    ₹-110,023     -17.2%      +1.6%     
  PERSISTENT  37     IT         01-Apr-22   2,292.8     1,815.3     310    ₹-148,029     -20.8%      -0.8%     
  ⚠  WAZ < 0 (momentum below universe mean): MPHASIS, INFY

  AFTER: Invested ₹13,465,331 | Cash ₹450,778 | Total ₹13,916,110 | Positions 15/20 | Slot ₹695,805

========================================================================
  REBALANCE #46  —  01 Aug 2022
  NAV: ₹15,903,368  |  Slot: ₹795,168  |  Cash: ₹450,778
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       3,213.2     1590   ₹4,860,632    +1956.6%  728d  
  MPHASIS     122    IT         03-Aug-20   1,024.9     2,155.2     242    ₹273,522      +110.3%   728d  
  ADANIENSOL  —      ENERGY     01-Dec-21   1,797.2     3,261.8     327    ₹478,924      +81.5%    243d  
  INFY        80     IT         03-Aug-20   815.8       1,377.3     304    ₹170,713      +68.8%    728d  
  POWERGRID   62     ENERGY     01-Feb-22   130.0       137.4       4924   ₹36,331       +5.7%     181d  
  POLYCAB     63     MFG        01-Oct-21   2,283.5     2,294.6     248    ₹2,756        +0.5%     304d  
  ONGC        90     OIL&GAS    01-Dec-21   109.7       107.8       5361   ₹-9,886       -1.7%     243d  
  RELIANCE    71     OIL&GAS    01-Apr-22   1,202.9     1,166.2     591    ₹-21,682      -3.0%     122d  
  LT          —      INFRA      01-Feb-22   1,891.8     1,746.8     338    ₹-49,008      -7.7%     181d  
  PERSISTENT  93     IT         01-Apr-22   2,292.8     1,780.3     310    ₹-158,876     -22.4%    122d  

  ENTRIES (14)
  [52w filter blocked 1: ADANIGREEN(-25.5%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TVSMOTOR    1      AUTO       3.482    0.82   +62.0%    +47.7%    911.5       872    ₹794,819      +7.4%     
  M&M         2      AUTO       3.407    1.00   +71.6%    +39.5%    1,193.6     666    ₹794,938      +8.2%     
  ITC         4      FMCG       2.747    0.76   +54.3%    +21.3%    254.0       3130   ₹794,967      +3.9%     
  VBL         5      FMCG       2.725    0.67   +83.6%    +27.4%    183.1       4342   ₹795,106      +7.8%     
  SIEMENS     6      ENERGY     2.456    0.94   +42.9%    +25.4%    1,598.0     497    ₹794,200      +3.8%     
  CUMMINSIND  7      INFRA      2.356    0.79   +48.7%    +20.6%    1,157.0     687    ₹794,828      +6.1%     
  COALINDIA   10     ENERGY     1.972    0.99   +65.3%    +13.6%    157.1       5059   ₹795,022      +7.2%     
  EICHERMOT   11     AUTO       1.968    0.97   +21.8%    +24.4%    2,962.8     268    ₹794,035      +2.7%     
  PAGEIND     12     MFG        1.938    0.91   +56.0%    +8.7%     46,806.7    16     ₹748,907      +8.7%     
  BOSCHLTD    13     AUTO       1.917    1.13   +19.1%    +26.2%    16,768.6    47     ₹788,124      +8.4%     
  SBILIFE     14     FIN SVC    1.897    0.86   +16.1%    +21.8%    1,303.6     609    ₹793,878      +11.7%    
  HINDUNILVR  15     FMCG       1.857    0.66   +11.5%    +20.5%    2,413.3     329    ₹793,988      +2.4%     
  ABBOTINDIA  16     HEALTH     1.782    0.44   +14.5%    +23.3%    19,434.7    40     ₹777,388      +4.7%     
  SBIN        18     PSU BNK    1.679    1.13   +27.4%    +13.0%    495.9       1603   ₹794,873      +5.8%     

  HOLDS (5)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   10     IT         01-Apr-21   2,599.1     8,324.1     166    ₹950,352      +220.3%     +5.1%     
  HAL         4      DEFENCE    01-Apr-22   723.8       963.6       983    ₹235,673      +33.1%      +8.3%     
  BEL         9      DEFENCE    01-Apr-22   68.5        90.6        10383  ₹228,632      +32.1%      +10.0%    
  BAJAJHLDNG  43     FIN SVC    01-Oct-21   4,433.3     4,946.0     128    ₹65,629       +11.6%      +7.1%     
  NTPC        38     ENERGY     01-Apr-22   126.0       138.0       5646   ₹67,457       +9.5%       +4.9%     

  AFTER: Invested ₹15,736,595 | Cash ₹153,646 | Total ₹15,890,241 | Positions 19/20 | Slot ₹795,168

========================================================================
  REBALANCE #47  —  03 Oct 2022
  NAV: ₹16,320,306  |  Slot: ₹816,015  |  Cash: ₹153,646
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TATAELXSI   51     IT         01-Apr-21   2,599.1     7,934.6     166    ₹885,688      +205.3%   550d  
  HINDUNILVR  45     FMCG       01-Aug-22   2,413.3     2,440.8     329    ₹9,033        +1.1%     63d   
  SBILIFE     52     FIN SVC    01-Aug-22   1,303.6     1,228.1     609    ₹-45,971      -5.8%     63d   
  ABBOTINDIA  107    HEALTH     01-Aug-22   19,434.7    18,058.7    40     ₹-55,041      -7.1%     63d   
  BOSCHLTD    98     AUTO       01-Aug-22   16,768.6    14,839.5    47     ₹-90,669      -11.5%    63d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TIINDIA     1      AUTO       3.352    0.88   +97.9%    +53.1%    2,688.0     303    ₹814,461      +3.6%     
  TITAN       8      CON DUR    2.283    1.18   +21.3%    +32.7%    2,549.8     320    ₹815,950      -1.3%     
  AMBUJACEM   11     INFRA      2.165    0.91   +20.8%    +32.9%    480.2       1699   ₹815,882      -1.7%     
  ICICIBANK   13     PVT BNK    2.028    1.00   +19.0%    +21.2%    827.9       985    ₹815,436      -3.0%     
  CIPLA       14     HEALTH     1.977    0.27   +17.0%    +19.7%    1,090.2     748    ₹815,499      +5.1%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         2      DEFENCE    01-Apr-22   723.8       1,092.9     983    ₹362,794      +51.0%      -3.3%     
  BAJAJHLDNG  4      FIN SVC    01-Oct-21   4,433.3     6,221.9     128    ₹228,942      +40.3%      +1.1%     
  BEL         8      DEFENCE    01-Apr-22   68.5        94.6        10383  ₹270,862      +38.1%      -5.2%     
  VBL         3      FMCG       01-Aug-22   183.1       212.1       4342   ₹126,013      +15.8%      +0.2%     
  NTPC        24     ENERGY     01-Apr-22   126.0       144.1       5646   ₹101,937      +14.3%      -1.8%     
  EICHERMOT   18     AUTO       01-Aug-22   2,962.8     3,344.6     268    ₹102,327      +12.9%      -2.5%     
  TVSMOTOR    6      AUTO       01-Aug-22   911.5       979.2       872    ₹59,055       +7.4%       -2.8%     
  ITC         11     FMCG       01-Aug-22   254.0       267.9       3130   ₹43,691       +5.5%       -1.9%     
  COALINDIA   27     ENERGY     01-Aug-22   157.1       161.0       5059   ₹19,334       +2.4%       -3.1%     
  M&M         13     AUTO       01-Aug-22   1,193.6     1,206.7     666    ₹8,734        +1.1%       -1.5%     
  PAGEIND     9      MFG        01-Aug-22   46,806.7    47,181.7    16     ₹6,001        +0.8%       -0.9%     
  SIEMENS     31     ENERGY     01-Aug-22   1,598.0     1,568.0     497    ₹-14,926      -1.9%       -4.7%     
  CUMMINSIND  29     INFRA      01-Aug-22   1,157.0     1,129.1     687    ₹-19,144      -2.4%       -1.9%     
  SBIN        39     PSU BNK    01-Aug-22   495.9       482.9       1603   ₹-20,849      -2.6%       -4.0%     

  AFTER: Invested ₹15,956,021 | Cash ₹359,444 | Total ₹16,315,465 | Positions 19/20 | Slot ₹816,015

========================================================================
  REBALANCE #48  —  01 Dec 2022
  NAV: ₹17,590,186  |  Slot: ₹879,509  |  Cash: ₹359,444
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TITAN       60     CON DUR    03-Oct-22   2,549.8     2,613.7     320    ₹20,445       +2.5%     59d   
  SIEMENS     50     ENERGY     01-Aug-22   1,598.0     1,607.3     497    ₹4,635        +0.6%     122d  
  PAGEIND     79     MFG        01-Aug-22   46,806.7    45,354.9    16     ₹-23,229      -3.1%     122d  

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SUNPHARMA   5      HEALTH     2.632    0.63   +37.9%    +17.2%    1,009.8     870    ₹878,544      +2.3%     
  PFC         6      FIN SVC    2.611    0.92   +29.9%    +20.9%    94.9        9266   ₹879,446      +11.0%    
  AXISBANK    9      PVT BNK    2.359    1.02   +36.8%    +20.3%    901.5       975    ₹878,925      +3.3%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         2      DEFENCE    01-Apr-22   723.8       1,323.9     983    ₹589,924      +82.9%      +4.1%     
  BEL         24     DEFENCE    01-Apr-22   68.5        100.1       10383  ₹327,750      +46.1%      -2.3%     
  VBL         7      FMCG       01-Aug-22   183.1       250.6       4342   ₹292,989      +36.8%      +9.5%     
  BAJAJHLDNG  26     FIN SVC    01-Oct-21   4,433.3     6,038.0     128    ₹205,407      +36.2%      -2.6%     
  NTPC        25     ENERGY     01-Apr-22   126.0       154.8       5646   ₹162,062      +22.8%      +1.2%     
  AMBUJACEM   5      INFRA      03-Oct-22   480.2       570.9       1699   ₹154,110      +18.9%      +3.4%     
  CUMMINSIND  8      INFRA      01-Aug-22   1,157.0     1,372.2     687    ₹147,840      +18.6%      +5.9%     
  COALINDIA   23     ENERGY     01-Aug-22   157.1       180.3       5059   ₹116,912      +14.7%      -1.4%     
  SBIN        19     PSU BNK    01-Aug-22   495.9       564.9       1603   ₹110,724      +13.9%      +1.6%     
  TVSMOTOR    20     AUTO       01-Aug-22   911.5       1,032.8     872    ₹105,801      +13.3%      -2.1%     
  EICHERMOT   31     AUTO       01-Aug-22   2,962.8     3,319.6     268    ₹95,623       +12.0%      -1.4%     
  ICICIBANK   27     PVT BNK    03-Oct-22   827.9       917.5       985    ₹88,287       +10.8%      +1.6%     
  ITC         11     FMCG       01-Aug-22   254.0       280.5       3130   ₹82,858       +10.4%      -0.9%     
  M&M         29     AUTO       01-Aug-22   1,193.6     1,247.3     666    ₹35,773       +4.5%       +1.6%     
  TIINDIA     14     AUTO       03-Oct-22   2,688.0     2,806.1     303    ₹35,798       +4.4%       +5.2%     
  CIPLA       33     HEALTH     03-Oct-22   1,090.2     1,084.6     748    ₹-4,219       -0.5%       +0.5%     

  AFTER: Invested ₹17,506,750 | Cash ₹80,305 | Total ₹17,587,055 | Positions 19/20 | Slot ₹879,509

========================================================================
  REBALANCE #49  —  01 Feb 2023
  NAV: ₹16,320,917  |  Slot: ₹816,046  |  Cash: ₹80,305
========================================================================

  [REGIME OFF] Nifty 200 9,172.1 < EMA200 9,287.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         14     DEFENCE    01-Apr-22   723.8       1,136.0     983    ₹405,225      +57.0%      -5.1%     
  BAJAJHLDNG  67     FIN SVC    01-Oct-21   4,433.3     5,815.5     128    ₹176,921      +31.2%      +3.8%     
  BEL         64     DEFENCE    01-Apr-22   68.5        87.4        10383  ₹196,009      +27.5%      -7.3%     
  VBL         7      FMCG       01-Aug-22   183.1       232.5       4342   ₹214,364      +27.0%      -4.7%     
  NTPC        41     ENERGY     01-Apr-22   126.0       152.8       5646   ₹150,900      +21.2%      +1.1%     
  CUMMINSIND  6      INFRA      01-Aug-22   1,157.0     1,361.3     687    ₹140,399      +17.7%      -0.0%     
  ITC         1      FMCG       01-Aug-22   254.0       298.5       3130   ₹139,346      +17.5%      +6.2%     
  COALINDIA   24     ENERGY     01-Aug-22   157.1       175.9       5059   ₹94,930       +11.9%      -0.7%     
  TVSMOTOR    22     AUTO       01-Aug-22   911.5       1,001.7     872    ₹78,654       +9.9%       -0.2%     
  M&M         12     AUTO       01-Aug-22   1,193.6     1,303.8     666    ₹73,408       +9.2%       +2.7%     
  EICHERMOT   66     AUTO       01-Aug-22   2,962.8     3,189.9     268    ₹60,852       +7.7% ⚠     +2.7%     
  ICICIBANK   76     PVT BNK    03-Oct-22   827.9       827.5       985    ₹-384         -0.0% ⚠     -1.5%     
  SBIN        84     PSU BNK    01-Aug-22   495.9       489.9       1603   ₹-9,531       -1.2% ⚠     -8.6%     
  PFC         8      FIN SVC    01-Dec-22   94.9        93.4        9266   ₹-13,652      -1.6%       -6.0%     
  SUNPHARMA   38     HEALTH     01-Dec-22   1,009.8     979.4       870    ₹-26,488      -3.0%       -1.5%     
  TIINDIA     36     AUTO       03-Oct-22   2,688.0     2,605.8     303    ₹-24,906      -3.1%       -1.5%     
  AXISBANK    59     PVT BNK    01-Dec-22   901.5       855.0       975    ₹-45,315      -5.2%       -5.5%     
  CIPLA       83     HEALTH     03-Oct-22   1,090.2     995.1       748    ₹-71,183      -8.7% ⚠     -2.3%     
  AMBUJACEM   131    INFRA      03-Oct-22   480.2       328.3       1699   ₹-258,047     -31.6% ⚠    -28.6%    
  ⚠  WAZ < 0 (momentum below universe mean): EICHERMOT, ICICIBANK, CIPLA, SBIN, AMBUJACEM

  AFTER: Invested ₹16,240,613 | Cash ₹80,305 | Total ₹16,320,917 | Positions 19/20 | Slot ₹816,046

========================================================================
  REBALANCE #50  —  03 Apr 2023
  NAV: ₹16,985,631  |  Slot: ₹849,282  |  Cash: ₹80,305
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
  HAL         5      DEFENCE    01-Apr-22   723.8       1,312.6     983    ₹578,835      +81.4%      +1.9%     
  VBL         3      FMCG       01-Aug-22   183.1       280.4       4342   ₹422,309      +53.1%      +5.0%     
  BEL         30     DEFENCE    01-Apr-22   68.5        94.1        10383  ₹264,836      +37.2%      +3.4%     
  CUMMINSIND  4      INFRA      01-Aug-22   1,157.0     1,543.4     687    ₹265,472      +33.4%      -1.1%     
  NTPC        11     ENERGY     01-Apr-22   126.0       164.0       5646   ₹214,049      +30.1%      +1.8%     
  BAJAJHLDNG  49     FIN SVC    01-Oct-21   4,433.3     5,564.3     128    ₹144,770      +25.5%      -3.3%     
  ITC         1      FMCG       01-Aug-22   254.0       318.1       3130   ₹200,641      +25.2%      -0.3%     
  TVSMOTOR    10     AUTO       01-Aug-22   911.5       1,072.3     872    ₹140,199      +17.6%      +2.1%     
  PFC         8      FIN SVC    01-Dec-22   94.9        108.8       9266   ₹129,010      +14.7%      +1.5%     
  COALINDIA   34     ENERGY     01-Aug-22   157.1       179.8       5059   ₹114,652      +14.4%      +2.0%     
  ICICIBANK   35     PVT BNK    03-Oct-22   827.9       862.4       985    ₹34,075       +4.2%       +3.4%     
  SBIN        102    PSU BNK    01-Aug-22   495.9       489.4       1603   ₹-10,425      -1.3% ⚠     +0.4%     
  EICHERMOT   59     AUTO       01-Aug-22   2,962.8     2,901.3     268    ₹-16,482      -2.1% ⚠     +0.2%     
  AXISBANK    67     PVT BNK    01-Dec-22   901.5       862.3       975    ₹-38,168      -4.3% ⚠     +2.0%     
  TIINDIA     43     AUTO       03-Oct-22   2,688.0     2,553.0     303    ₹-40,892      -5.0%       -0.7%     
  M&M         28     AUTO       01-Aug-22   1,193.6     1,128.2     666    ₹-43,576      -5.5%       -1.4%     
  SUNPHARMA   57     HEALTH     01-Dec-22   1,009.8     951.9       870    ₹-50,414      -5.7% ⚠     +0.5%     
  CIPLA       133    HEALTH     03-Oct-22   1,090.2     859.5       748    ₹-172,584     -21.2% ⚠    -0.3%     
  AMBUJACEM   97     INFRA      03-Oct-22   480.2       368.3       1699   ₹-190,091     -23.3%      +1.3%     
  ⚠  WAZ < 0 (momentum below universe mean): SUNPHARMA, EICHERMOT, AXISBANK, SBIN, CIPLA

  AFTER: Invested ₹16,905,326 | Cash ₹80,305 | Total ₹16,985,631 | Positions 19/20 | Slot ₹849,282

========================================================================
  REBALANCE #51  —  01 Jun 2023
  NAV: ₹19,090,297  |  Slot: ₹954,515  |  Cash: ₹80,305
========================================================================

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PFC         5      FIN SVC    01-Dec-22   94.9        128.6       9266   ₹312,100      +35.5%    182d  
  NTPC        95     ENERGY     01-Apr-22   126.0       160.8       5646   ₹196,347      +27.6%    426d  
  SBIN        40     PSU BNK    01-Aug-22   495.9       551.9       1603   ₹89,836       +11.3%    304d  
  M&M         63     AUTO       01-Aug-22   1,193.6     1,271.9     666    ₹52,118       +6.6%     304d  
  AXISBANK    —      PVT BNK    01-Dec-22   901.5       917.3       975    ₹15,413       +1.8%     182d  
  SUNPHARMA   101    HEALTH     01-Dec-22   1,009.8     960.3       870    ₹-43,097      -4.9%     182d  
  AMBUJACEM   —      INFRA      03-Oct-22   480.2       421.8       1699   ₹-99,178      -12.2%    241d  
  CIPLA       111    HEALTH     03-Oct-22   1,090.2     930.2       748    ₹-119,720     -14.7%    241d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RECLTD      2      FIN SVC    2.958    1.14   +77.0%    +24.4%    120.3       7931   ₹954,403      +5.1%     
  CHOLAFIN    4      FIN SVC    2.829    1.17   +63.8%    +36.8%    1,039.4     918    ₹954,134      +3.0%     
  SYNGENE     5      HEALTH     2.475    0.48   +37.7%    +28.3%    720.6       1324   ₹954,129      +3.7%     
  TORNTPHARM  7      HEALTH     2.240    0.42   +24.5%    +19.9%    1,716.8     555    ₹952,841      +4.8%     
  NESTLEIND   9      FMCG       2.156    0.49   +25.3%    +17.7%    1,062.6     898    ₹954,174      +1.6%     
  BAJAJ-AUTO  11     AUTO       1.990    0.61   +24.9%    +20.6%    4,298.5     222    ₹954,260      +2.5%     
  MAXHEALTH   13     HEALTH     1.846    0.32   +45.8%    +23.5%    530.5       1799   ₹954,432      +2.0%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         13     DEFENCE    01-Apr-22   723.8       1,490.9     983    ₹754,065      +106.0%     +1.9%     
  VBL         4      FMCG       01-Aug-22   183.1       335.0       4342   ₹659,625      +83.0%      +5.7%     
  BEL         18     DEFENCE    01-Apr-22   68.5        109.9       10383  ₹429,615      +60.4%      +3.8%     
  BAJAJHLDNG  41     FIN SVC    01-Oct-21   4,433.3     6,686.4     128    ₹288,403      +50.8%      +2.4%     
  ITC         1      FMCG       01-Aug-22   254.0       377.4       3130   ₹386,352      +48.6%      +3.5%     
  CUMMINSIND  16     INFRA      01-Aug-22   1,157.0     1,683.2     687    ₹361,545      +45.5%      +4.5%     
  TVSMOTOR    10     AUTO       01-Aug-22   911.5       1,257.5     872    ₹301,688      +38.0%      +1.9%     
  EICHERMOT   35     AUTO       01-Aug-22   2,962.8     3,588.7     268    ₹167,729      +21.1%      +4.1%     
  COALINDIA   55     ENERGY     01-Aug-22   157.1       188.1       5059   ₹156,799      +19.7%      -3.3%     
  ICICIBANK   27     PVT BNK    03-Oct-22   827.9       913.5       985    ₹84,346       +10.3%      -0.3%     
  TIINDIA     33     AUTO       03-Oct-22   2,688.0     2,875.8     303    ₹56,897       +7.0%       +5.1%     

  AFTER: Invested ₹18,714,776 | Cash ₹367,592 | Total ₹19,082,367 | Positions 18/20 | Slot ₹954,515

========================================================================
  REBALANCE #52  —  01 Aug 2023
  NAV: ₹20,895,867  |  Slot: ₹1,044,793  |  Cash: ₹367,592
========================================================================
  [SECTOR CAP≤4] dropped: SUNPHARMA

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJAJHLDNG  61     FIN SVC    01-Oct-21   4,433.3     7,119.1     128    ₹343,789      +60.6%    669d  
  RECLTD      2      FIN SVC    01-Jun-23   120.3       175.8       7931   ₹439,904      +46.1%    61d   
  COALINDIA   86     ENERGY     01-Aug-22   157.1       196.3       5059   ₹198,119      +24.9%    365d  
  ICICIBANK   57     PVT BNK    03-Oct-22   827.9       970.5       985    ₹140,528      +17.2%    302d  
  TIINDIA     52     AUTO       03-Oct-22   2,688.0     3,050.7     303    ₹109,895      +13.5%    302d  
  EICHERMOT   106    AUTO       01-Aug-22   2,962.8     3,298.8     268    ₹90,046       +11.3%    365d  
  NESTLEIND   85     FMCG       01-Jun-23   1,062.6     1,097.5     898    ₹31,381       +3.3%     61d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  POLYCAB     1      MFG        3.428    0.91   +109.5%   +41.5%    4,561.7     229    ₹1,044,627    +7.3%     
  TATACOMM    2      CONSUMP    2.709    1.02   +69.3%    +44.3%    1,705.7     612    ₹1,043,871    +6.9%     
  NTPC        3      ENERGY     2.695    0.74   +56.9%    +27.7%    207.6       5032   ₹1,044,759    +12.7%    
  APOLLOTYRE  5      AUTO       2.354    1.13   +102.0%   +21.5%    415.0       2517   ₹1,044,646    +2.1%     
  COLPAL      6      FMCG       2.296    0.42   +31.3%    +28.3%    1,880.1     555    ₹1,043,437    +7.0%     
  CIPLA       11     HEALTH     1.921    0.36   +24.4%    +29.0%    1,145.3     912    ₹1,044,557    +9.6%     
  LT          13     INFRA      1.846    0.85   +49.6%    +13.2%    2,566.8     407    ₹1,044,704    +4.5%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         8      DEFENCE    01-Apr-22   723.8       1,870.9     983    ₹1,127,574    +158.5%     +0.8%     
  BEL         23     DEFENCE    01-Apr-22   68.5        127.0       10383  ₹607,030      +85.3%      +3.1%     
  VBL         49     FMCG       01-Aug-22   183.1       317.7       4342   ₹584,175      +73.5%      -1.2%     
  CUMMINSIND  12     INFRA      01-Aug-22   1,157.0     1,865.6     687    ₹486,825      +61.2%      +1.2%     
  ITC         15     FMCG       01-Aug-22   254.0       399.0       3130   ₹453,787      +57.1%      -0.7%     
  TVSMOTOR    24     AUTO       01-Aug-22   911.5       1,353.9     872    ₹385,816      +48.5%      +1.7%     
  TORNTPHARM  14     HEALTH     01-Jun-23   1,716.8     1,927.0     555    ₹116,638      +12.2%      +2.0%     
  SYNGENE     25     HEALTH     01-Jun-23   720.6       800.9       1324   ₹106,278      +11.1%      +3.4%     
  BAJAJ-AUTO  46     AUTO       01-Jun-23   4,298.5     4,697.4     222    ₹88,567       +9.3%       +1.9%     
  CHOLAFIN    13     FIN SVC    01-Jun-23   1,039.4     1,126.2     918    ₹79,739       +8.4%       -0.7%     
  MAXHEALTH   35     HEALTH     01-Jun-23   530.5       569.7       1799   ₹70,470       +7.4%       -4.7%     

  AFTER: Invested ₹20,790,224 | Cash ₹96,963 | Total ₹20,887,186 | Positions 18/20 | Slot ₹1,044,793

========================================================================
  REBALANCE #53  —  03 Oct 2023
  NAV: ₹21,626,904  |  Slot: ₹1,081,345  |  Cash: ₹96,963
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BEL         46     DEFENCE    01-Apr-22   68.5        136.2       10383  ₹701,945      +98.6%    550d  
  CUMMINSIND  91     INFRA      01-Aug-22   1,157.0     1,624.9     687    ₹321,502      +40.4%    428d  
  MAXHEALTH   79     HEALTH     01-Jun-23   530.5       590.5       1799   ₹107,870      +11.3%    124d  
  TATACOMM    21     CONSUMP    01-Aug-23   1,705.7     1,840.2     612    ₹82,313       +7.9%     63d   
  TORNTPHARM  76     HEALTH     01-Jun-23   1,716.8     1,826.7     555    ₹60,967       +6.4%     124d  
  APOLLOTYRE  85     AUTO       01-Aug-23   415.0       361.3       2517   ₹-135,333     -13.0%    63d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COALINDIA   4      ENERGY     2.714    0.87   +50.3%    +28.2%    242.6       4457   ₹1,081,240    +5.4%     
  SUNTV       5      MEDIA      2.593    0.94   +31.4%    +43.0%    586.4       1843   ₹1,080,776    +4.2%     
  OIL         6      OIL&GAS    2.463    0.46   +78.2%    +22.4%    180.7       5983   ₹1,081,267    +3.8%     
  PGHH        7      FMCG       2.123    0.31   +23.2%    +24.1%    16,905.8    63     ₹1,065,064    +2.8%     
  TORNTPOWER  9      ENERGY     1.909    1.00   +58.9%    +23.4%    718.0       1506   ₹1,081,328    +4.3%     
  TRENT       11     CONSUMP    1.884    1.09   +46.3%    +17.6%    2,053.9     526    ₹1,080,370    -0.3%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         48     DEFENCE    01-Apr-22   723.8       1,901.3     983    ₹1,157,458    +162.7%     +0.3%     
  VBL         40     FMCG       01-Aug-22   183.1       368.8       4342   ₹806,335      +101.4%     +0.5%     
  TVSMOTOR    25     AUTO       01-Aug-22   911.5       1,512.4     872    ₹523,989      +65.9%      +2.7%     
  ITC         75     FMCG       01-Aug-22   254.0       377.5       3130   ₹386,486      +48.6%      -1.4%     
  CHOLAFIN    30     FIN SVC    01-Jun-23   1,039.4     1,249.1     918    ₹192,552      +20.2%      +6.0%     
  LT          7      INFRA      01-Aug-23   2,566.8     2,991.9     407    ₹172,991      +16.6%      +5.9%     
  POLYCAB     5      MFG        01-Aug-23   4,561.7     5,293.7     229    ₹167,641      +16.0%      +3.5%     
  SYNGENE     50     HEALTH     01-Jun-23   720.6       805.4       1324   ₹112,198      +11.8%      +1.7%     
  BAJAJ-AUTO  31     AUTO       01-Jun-23   4,298.5     4,785.7     222    ₹108,158      +11.3%      +1.4%     
  NTPC        10     ENERGY     01-Aug-23   207.6       225.5       5032   ₹90,137       +8.6%       +2.0%     
  CIPLA       34     HEALTH     01-Aug-23   1,145.3     1,149.7     912    ₹3,945        +0.4%       -1.8%     
  COLPAL      22     FMCG       01-Aug-23   1,880.1     1,853.9     555    ₹-14,536      -1.4%       -0.8%     

  AFTER: Invested ₹21,358,400 | Cash ₹260,822 | Total ₹21,619,221 | Positions 18/20 | Slot ₹1,081,345

========================================================================
  REBALANCE #54  —  01 Dec 2023
  NAV: ₹24,202,152  |  Slot: ₹1,210,108  |  Cash: ₹260,822
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HAL         13     DEFENCE    01-Apr-22   723.8       2,392.1     983    ₹1,639,918    +230.5%   609d  
  SYNGENE     88     HEALTH     01-Jun-23   720.6       741.8       1324   ₹28,078       +2.9%     183d  
  CIPLA       98     HEALTH     01-Aug-23   1,145.3     1,171.9     912    ₹24,245       +2.3%     122d  

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HEROMOTOCO  9      AUTO       2.117    0.97   +46.0%    +25.8%    3,443.0     351    ₹1,208,483    +9.6%     
  PERSISTENT  10     IT         2.068    1.16   +67.2%    +26.0%    3,167.3     382    ₹1,209,891    +1.8%     
  BOSCHLTD    11     AUTO       2.058    0.75   +36.2%    +18.5%    21,526.2    56     ₹1,205,467    +6.6%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VBL         22     FMCG       01-Aug-22   183.1       432.7       4342   ₹1,083,650    +136.3%     +5.8%     
  TVSMOTOR    4      AUTO       01-Aug-22   911.5       1,887.8     872    ₹851,364      +107.1%     +9.7%     
  ITC         56     FMCG       01-Aug-22   254.0       386.1       3130   ₹413,487      +52.0%      +2.6%     
  TRENT       5      CONSUMP    03-Oct-23   2,053.9     2,800.7     526    ₹392,776      +36.4%      +10.1%    
  BAJAJ-AUTO  7      AUTO       01-Jun-23   4,298.5     5,767.9     222    ₹326,213      +34.2%      +5.8%     
  TORNTPOWER  10     ENERGY     03-Oct-23   718.0       914.4       1506   ₹295,742      +27.3%      +14.4%    
  COALINDIA   3      ENERGY     03-Oct-23   242.6       301.3       4457   ₹261,694      +24.2%      +6.1%     
  NTPC        8      ENERGY     01-Aug-23   207.6       253.9       5032   ₹232,927      +22.3%      +7.3%     
  LT          12     INFRA      01-Aug-23   2,566.8     3,106.2     407    ₹219,508      +21.0%      +4.3%     
  COLPAL      16     FMCG       01-Aug-23   1,880.1     2,158.3     555    ₹154,446      +14.8%      +5.5%     
  POLYCAB     23     MFG        01-Aug-23   4,561.7     5,157.1     229    ₹136,348      +13.1%      +0.5%     
  SUNTV       32     MEDIA      03-Oct-23   586.4       639.8       1843   ₹98,305       +9.1%       +2.0%     
  CHOLAFIN    61     FIN SVC    01-Jun-23   1,039.4     1,124.0     918    ₹77,682       +8.1%       -0.4%     
  OIL         28     OIL&GAS    03-Oct-23   180.7       192.7       5983   ₹71,403       +6.6%       +1.5%     
  PGHH        49     FMCG       03-Oct-23   16,905.8    16,628.0    63     ₹-17,502      -1.6%       -1.2%     

  AFTER: Invested ₹23,162,744 | Cash ₹1,035,105 | Total ₹24,197,849 | Positions 18/20 | Slot ₹1,210,108

========================================================================
  REBALANCE #55  —  01 Feb 2024
  NAV: ₹26,834,206  |  Slot: ₹1,341,710  |  Cash: ₹1,035,105
========================================================================
  [SECTOR CAP≤4] dropped: APOLLOTYRE

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ITC         77     FMCG       01-Aug-22   254.0       380.2       3130   ₹394,949      +49.7%    549d  
  CHOLAFIN    80     FIN SVC    01-Jun-23   1,039.4     1,140.8     918    ₹93,092       +9.8%     245d  
  SUNTV       85     MEDIA      03-Oct-23   586.4       621.7       1843   ₹65,055       +6.0%     121d  
  PGHH        90     FMCG       03-Oct-23   16,905.8    16,275.4    63     ₹-39,714      -3.7%     121d  
  POLYCAB     106    MFG        01-Aug-23   4,561.7     4,200.9     229    ₹-82,630      -7.9%     184d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  NHPC        3      ENERGY     3.080    1.07   +124.2%   +79.2%    85.8        15639  ₹1,341,673    +17.7%    
  MRF         7      MFG        2.281    0.53   +59.4%    +30.7%    142,044.1   9      ₹1,278,397    +4.6%     
  BHARTIARTL  15     CONSUMP    1.985    0.62   +51.3%    +24.3%    1,135.3     1181   ₹1,340,823    +3.5%     
  TORNTPHARM  16     HEALTH     1.974    0.33   +61.0%    +30.2%    2,440.9     549    ₹1,340,064    +3.4%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VBL         19     FMCG       01-Aug-22   183.1       510.3       4342   ₹1,420,828    +178.7%     +2.4%     
  TVSMOTOR    16     AUTO       01-Aug-22   911.5       1,972.3     872    ₹925,019      +116.4%     +0.1%     
  BAJAJ-AUTO  5      AUTO       01-Jun-23   4,298.5     7,303.5     222    ₹667,126      +69.9%      +5.9%     
  TRENT       4      CONSUMP    03-Oct-23   2,053.9     3,095.0     526    ₹547,609      +50.7%      -0.6%     
  OIL         21     OIL&GAS    03-Oct-23   180.7       271.3       5983   ₹541,732      +50.1%      +9.7%     
  NTPC        11     ENERGY     01-Aug-23   207.6       304.0       5032   ₹484,949      +46.4%      +3.3%     
  COALINDIA   15     ENERGY     03-Oct-23   242.6       353.5       4457   ₹494,330      +45.7%      +4.9%     
  TORNTPOWER  17     ENERGY     03-Oct-23   718.0       1,009.7     1506   ₹439,240      +40.6%      +5.3%     
  PERSISTENT  20     IT         01-Dec-23   3,167.3     4,099.4     382    ₹356,067      +29.4%      +5.1%     
  LT          51     INFRA      01-Aug-23   2,566.8     3,308.0     407    ₹301,665      +28.9%      -3.9%     
  COLPAL      24     FMCG       01-Aug-23   1,880.1     2,370.6     555    ₹272,251      +26.1%      +0.8%     
  HEROMOTOCO  13     AUTO       01-Dec-23   3,443.0     4,200.0     351    ₹265,705      +22.0%      +5.2%     
  BOSCHLTD    38     AUTO       01-Dec-23   21,526.2    23,082.1    56     ₹87,128       +7.2%       +3.1%     

  AFTER: Invested ₹25,729,739 | Cash ₹1,098,173 | Total ₹26,827,911 | Positions 17/20 | Slot ₹1,341,710

========================================================================
  REBALANCE #56  —  01 Apr 2024
  NAV: ₹30,080,990  |  Slot: ₹1,504,049  |  Cash: ₹1,098,173
========================================================================
  [SECTOR CAP≤4] dropped: MARUTI, MOTHERSON

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COALINDIA   25     ENERGY     03-Oct-23   242.6       388.7       4457   ₹651,004      +60.2%    181d  
  NTPC        34     ENERGY     01-Aug-23   207.6       326.4       5032   ₹597,505      +57.2%    244d  
  PERSISTENT  55     IT         01-Dec-23   3,167.3     3,949.9     382    ₹298,975      +24.7%    122d  
  NHPC        28     ENERGY     01-Feb-24   85.8        86.2        15639  ₹7,171        +0.5%     60d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SUNPHARMA   3      HEALTH     2.795    0.47   +71.1%    +30.8%    1,598.7     940    ₹1,502,768    +3.3%     
  CUMMINSIND  6      INFRA      2.700    0.94   +84.2%    +52.5%    2,927.3     513    ₹1,501,721    +6.4%     
  KALYANKJIL  8      CON DUR    2.253    0.72   +260.5%   +20.9%    423.7       3549   ₹1,503,738    +8.3%     
  DIXON       9      CON DUR    2.139    0.98   +160.8%   +17.5%    7,586.0     198    ₹1,502,020    +7.3%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VBL         39     FMCG       01-Aug-22   183.1       555.1       4342   ₹1,615,250    +203.1%     -0.7%     
  TVSMOTOR    37     AUTO       01-Aug-22   911.5       2,122.8     872    ₹1,056,303    +132.9%     +1.1%     
  OIL         10     OIL&GAS    03-Oct-23   180.7       373.1       5983   ₹1,151,100    +106.5%     +2.3%     
  BAJAJ-AUTO  1      AUTO       01-Jun-23   4,298.5     8,626.2     222    ₹960,748      +100.7%     +4.3%     
  TORNTPOWER  6      ENERGY     03-Oct-23   718.0       1,379.4     1506   ₹995,977      +92.1%      +13.6%    
  TRENT       7      CONSUMP    03-Oct-23   2,053.9     3,877.1     526    ₹958,959      +88.8%      -0.9%     
  LT          44     INFRA      01-Aug-23   2,566.8     3,736.4     407    ₹476,004      +45.6%      +6.2%     
  BOSCHLTD    5      AUTO       01-Dec-23   21,526.2    29,732.3    56     ₹459,542      +38.1%      +2.7%     
  COLPAL      36     FMCG       01-Aug-23   1,880.1     2,572.1     555    ₹384,090      +36.8%      +2.5%     
  HEROMOTOCO  27     AUTO       01-Dec-23   3,443.0     4,380.0     351    ₹328,901      +27.2%      +1.7%     
  TORNTPHARM  30     HEALTH     01-Feb-24   2,440.9     2,620.8     549    ₹98,770       +7.4%       +2.8%     
  BHARTIARTL  20     CONSUMP    01-Feb-24   1,135.3     1,200.6     1181   ₹77,046       +5.7%       +1.4%     
  MRF         42     MFG        01-Feb-24   142,044.1   135,309.0   9      ₹-60,616      -4.7%       -1.2%     

  AFTER: Invested ₹28,760,847 | Cash ₹1,313,006 | Total ₹30,073,853 | Positions 17/20 | Slot ₹1,504,049

========================================================================
  REBALANCE #57  —  03 Jun 2024
  NAV: ₹32,005,039  |  Slot: ₹1,600,252  |  Cash: ₹1,313,006
========================================================================
  [SECTOR CAP≤4] dropped: M&M, ASHOKLEY, MOTHERSON

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VBL         85     FMCG       01-Aug-22   183.1       582.2       4342   ₹1,732,717    +217.9%   672d  
  OIL         —      OIL&GAS    03-Oct-23   180.7       422.6       5983   ₹1,447,049    +133.8%   244d  
  TORNTPOWER  —      ENERGY     03-Oct-23   718.0       1,469.3     1506   ₹1,131,422    +104.6%   244d  
  LT          46     INFRA      01-Aug-23   2,566.8     3,794.0     407    ₹499,440      +47.8%    307d  
  BHARTIARTL  —      CONSUMP    01-Feb-24   1,135.3     1,371.9     1181   ₹279,416      +20.8%    123d  
  SUNPHARMA   98     HEALTH     01-Apr-24   1,598.7     1,425.8     940    ₹-162,521     -10.8%    63d   
  MRF         137    MFG        01-Feb-24   142,044.1   126,586.5   9      ₹-139,119     -10.9%    123d  

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SIEMENS     1      ENERGY     3.477    1.08   +114.1%   +59.3%    4,254.8     376    ₹1,599,823    +6.5%     
  BSE         3      FIN SVC    2.933    1.03   +415.1%   +18.8%    894.9       1788   ₹1,600,015    -0.2%     
  PRESTIGE    6      REALTY     2.652    1.16   +250.6%   +41.9%    1,731.5     924    ₹1,599,947    +12.4%    
  CGPOWER     8      ENERGY     2.419    0.55   +92.6%    +60.2%    683.4       2341   ₹1,599,790    +10.2%    
  ESCORTS     10     MFG        2.304    0.89   +89.4%    +34.6%    3,820.6     418    ₹1,596,996    +5.0%     
  HAVELLS     12     CON DUR    2.083    0.70   +51.7%    +32.4%    1,852.4     863    ₹1,598,642    +4.5%     
  HINDALCO    13     METAL      2.076    1.14   +70.7%    +37.8%    686.9       2329   ₹1,599,896    +4.2%     
  POLYCAB     16     MFG        1.985    0.98   +106.5%   +44.7%    6,826.6     234    ₹1,597,430    +7.3%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TVSMOTOR    58     AUTO       01-Aug-22   911.5       2,232.9     872    ₹1,152,273    +145.0%     +3.9%     
  TRENT       11     CONSUMP    03-Oct-23   2,053.9     4,652.6     526    ₹1,366,915    +126.5%     +2.2%     
  BAJAJ-AUTO  31     AUTO       01-Jun-23   4,298.5     8,906.0     222    ₹1,022,865    +107.2%     +3.9%     
  HEROMOTOCO  38     AUTO       01-Dec-23   3,443.0     4,829.0     351    ₹486,507      +40.3%      +3.5%     
  COLPAL      55     FMCG       01-Aug-23   1,880.1     2,578.9     555    ₹387,825      +37.2%      -0.3%     
  BOSCHLTD    60     AUTO       01-Dec-23   21,526.2    29,444.7    56     ₹443,435      +36.8%      -2.0%     
  DIXON       6      CON DUR    01-Apr-24   7,586.0     9,878.1     198    ₹453,838      +30.2%      +10.6%    
  CUMMINSIND  13     INFRA      01-Apr-24   2,927.3     3,618.2     513    ₹354,430      +23.6%      +2.9%     
  TORNTPHARM  77     HEALTH     01-Feb-24   2,440.9     2,623.4     549    ₹100,168      +7.5%       +0.7%     
  KALYANKJIL  32     CON DUR    01-Apr-24   423.7       388.9       3549   ₹-123,399     -8.2%       -2.3%     

  AFTER: Invested ₹30,571,774 | Cash ₹1,418,075 | Total ₹31,989,849 | Positions 18/20 | Slot ₹1,600,252

========================================================================
  REBALANCE #58  —  01 Aug 2024
  NAV: ₹35,007,684  |  Slot: ₹1,750,384  |  Cash: ₹1,418,075
========================================================================
  [SECTOR CAP≤4] dropped: MOTHERSON, M&M

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DIXON       6      CON DUR    01-Apr-24   7,586.0     11,661.2    198    ₹806,895      +53.7%    122d  
  CUMMINSIND  43     INFRA      01-Apr-24   2,927.3     3,738.1     513    ₹415,907      +27.7%    122d  
  PRESTIGE    22     REALTY     03-Jun-24   1,731.5     1,751.7     924    ₹18,579       +1.2%     59d   
  POLYCAB     92     MFG        03-Jun-24   6,826.6     6,705.5     234    ₹-28,335      -1.8%     59d   
  HAVELLS     87     CON DUR    03-Jun-24   1,852.4     1,811.8     863    ₹-35,090      -2.2%     59d   
  HINDALCO    105    METAL      03-Jun-24   686.9       664.8       2329   ₹-51,610      -3.2%     59d   
  SIEMENS     40     ENERGY     03-Jun-24   4,254.8     4,111.4     376    ₹-53,929      -3.4%     59d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  OFSS        4      IT         2.578    0.98   +186.5%   +48.1%    10,120.6    172    ₹1,740,751    +1.7%     
  ZYDUSLIFE   5      HEALTH     2.423    0.71   +103.4%   +30.5%    1,227.3     1426   ₹1,750,187    +5.2%     
  PERSISTENT  6      IT         2.296    0.79   +91.3%    +42.7%    4,750.7     368    ₹1,748,263    +2.8%     
  LUPIN       8      HEALTH     2.064    0.47   +107.4%   +19.2%    1,941.3     901    ₹1,749,090    +7.9%     
  SUNTV       10     MEDIA      1.975    0.59   +78.1%    +35.8%    852.7       2052   ₹1,749,741    +8.2%     
  INFY        13     IT         1.862    0.60   +32.1%    +33.0%    1,741.4     1005   ₹1,750,117    +4.6%     
  SUNPHARMA   17     HEALTH     1.716    0.46   +58.2%    +14.5%    1,688.4     1036   ₹1,749,145    +5.2%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TVSMOTOR    11     AUTO       01-Aug-22   911.5       2,564.6     872    ₹1,441,481    +181.4%     +5.0%     
  TRENT       2      CONSUMP    03-Oct-23   2,053.9     5,759.0     526    ₹1,948,842    +180.4%     +5.3%     
  BAJAJ-AUTO  25     AUTO       01-Jun-23   4,298.5     9,358.3     222    ₹1,123,279    +117.7%     +2.2%     
  COLPAL      15     FMCG       01-Aug-23   1,880.1     3,238.2     555    ₹753,769      +72.2%      +7.2%     
  BOSCHLTD    19     AUTO       01-Dec-23   21,526.2    33,910.4    56     ₹693,518      +57.5%      -0.3%     
  HEROMOTOCO  30     AUTO       01-Dec-23   3,443.0     5,063.6     351    ₹568,845      +47.1%      -1.2%     
  KALYANKJIL  7      CON DUR    01-Apr-24   423.7       561.8       3549   ₹490,066      +32.6%      +5.2%     
  TORNTPHARM  24     HEALTH     01-Feb-24   2,440.9     3,146.3     549    ₹387,247      +28.9%      +5.1%     
  ESCORTS     26     MFG        03-Jun-24   3,820.6     4,093.5     418    ₹114,088      +7.1%       +1.4%     
  CGPOWER     50     ENERGY     03-Jun-24   683.4       725.6       2341   ₹98,727       +6.2%       +1.4%     
  BSE         48     FIN SVC    03-Jun-24   894.9       878.3       1788   ₹-29,591      -1.8%       +8.3%     

  AFTER: Invested ₹33,755,005 | Cash ₹1,238,149 | Total ₹34,993,154 | Positions 18/20 | Slot ₹1,750,384

========================================================================
  REBALANCE #59  —  01 Oct 2024
  NAV: ₹39,316,043  |  Slot: ₹1,965,802  |  Cash: ₹1,238,149
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  CGPOWER     78     ENERGY     03-Jun-24   683.4       748.2       2341   ₹151,645      +9.5%     120d  
  ESCORTS     99     MFG        03-Jun-24   3,820.6     4,151.9     418    ₹138,497      +8.7%     120d  
  SUNTV       —      OTHER      01-Aug-24   852.7       819.0       2052   ₹-69,195      -4.0%     61d   
  ZYDUSLIFE   62     HEALTH     01-Aug-24   1,227.3     1,068.1     1426   ₹-227,056     -13.0%    61d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  VOLTAS      8      CON DUR    2.141    0.87   +113.5%   +27.9%    1,838.5     1069   ₹1,965,335    +0.4%     
  UNITDSPR    11     FMCG       1.934    0.56   +56.3%    +26.8%    1,589.7     1236   ₹1,964,845    +3.2%     
  ALKEM       12     HEALTH     1.828    0.62   +68.2%    +25.1%    6,044.3     325    ₹1,964,390    +0.5%     
  HCLTECH     14     IT         1.747    0.62   +45.9%    +23.6%    1,693.6     1160   ₹1,964,621    +2.4%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       1      CONSUMP    03-Oct-23   2,053.9     7,597.1     526    ₹2,915,694    +269.9%     +2.9%     
  TVSMOTOR    11     AUTO       01-Aug-22   911.5       2,817.1     872    ₹1,661,675    +209.1%     +0.7%     
  BAJAJ-AUTO  3      AUTO       01-Jun-23   4,298.5     11,692.4    222    ₹1,641,453    +172.0%     +2.8%     
  COLPAL      4      FMCG       01-Aug-23   1,880.1     3,666.2     555    ₹991,297      +95.0%      +3.9%     
  KALYANKJIL  2      CON DUR    01-Apr-24   423.7       748.0       3549   ₹1,150,776    +76.5%      +6.8%     
  BOSCHLTD    15     AUTO       01-Dec-23   21,526.2    37,334.5    56     ₹885,264      +73.4%      +6.5%     
  HEROMOTOCO  29     AUTO       01-Dec-23   3,443.0     5,420.2     351    ₹693,992      +57.4%      -1.5%     
  BSE         7      FIN SVC    03-Jun-24   894.9       1,282.3     1788   ₹692,659      +43.3%      +11.1%    
  TORNTPHARM  10     HEALTH     01-Feb-24   2,440.9     3,308.2     549    ₹476,119      +35.5%      -1.3%     
  PERSISTENT  16     IT         01-Aug-24   4,750.7     5,435.4     368    ₹251,973      +14.4%      +3.4%     
  LUPIN       6      HEALTH     01-Aug-24   1,941.3     2,180.8     901    ₹215,843      +12.3%      -0.1%     
  SUNPHARMA   5      HEALTH     01-Aug-24   1,688.4     1,889.9     1036   ₹208,802      +11.9%      +2.9%     
  OFSS        20     IT         01-Aug-24   10,120.6    10,613.9    172    ₹84,837       +4.9%       +0.6%     
  INFY        37     IT         01-Aug-24   1,741.4     1,790.1     1005   ₹48,887       +2.8%       +0.1%     

  AFTER: Invested ₹39,246,481 | Cash ₹60,231 | Total ₹39,306,711 | Positions 18/20 | Slot ₹1,965,802

========================================================================
  REBALANCE #60  —  02 Dec 2024
  NAV: ₹37,017,795  |  Slot: ₹1,850,890  |  Cash: ₹60,231
========================================================================
  [SECTOR CAP≤4] dropped: COFORGE, WIPRO

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TVSMOTOR    79     AUTO       01-Aug-22   911.5       2,474.5     872    ₹1,362,905    +171.5%   854d  
  COLPAL      134    FMCG       01-Aug-23   1,880.1     2,792.9     555    ₹506,645      +48.6%    489d  
  HEROMOTOCO  89     AUTO       01-Dec-23   3,443.0     4,476.0     351    ₹362,587      +30.0%    367d  
  ALKEM       95     HEALTH     01-Oct-24   6,044.3     5,593.8     325    ₹-146,421     -7.5%     62d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  INDHOTEL    3      CONSUMP    2.898    1.18   +90.9%    +23.7%    795.2       2327   ₹1,850,337    +6.1%     
  M&M         8      AUTO       2.353    1.18   +94.8%    +7.5%     2,960.8     625    ₹1,850,506    +2.6%     
  CGPOWER     11     ENERGY     2.016    0.90   +93.6%    +8.5%     752.0       2461   ₹1,850,771    +2.9%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       8      CONSUMP    03-Oct-23   2,053.9     6,791.3     526    ₹2,491,873    +230.6%     +0.3%     
  BAJAJ-AUTO  74     AUTO       01-Jun-23   4,298.5     8,781.1     222    ₹995,143      +104.3%     -4.4%     
  KALYANKJIL  6      CON DUR    01-Apr-24   423.7       720.0       3549   ₹1,051,517    +69.9%      +3.3%     
  BSE         2      FIN SVC    03-Jun-24   894.9       1,516.5     1788   ₹1,111,488    +69.5%      +0.2%     
  BOSCHLTD    10     AUTO       01-Dec-23   21,526.2    34,460.4    56     ₹724,317      +60.1%      +0.1%     
  TORNTPHARM  31     HEALTH     01-Feb-24   2,440.9     3,278.0     549    ₹459,547      +34.3%      +3.5%     
  PERSISTENT  9      IT         01-Aug-24   4,750.7     5,875.8     368    ₹414,044      +23.7%      +3.1%     
  OFSS        3      IT         01-Aug-24   10,120.6    11,378.1    172    ₹216,276      +12.4%      +5.8%     
  LUPIN       35     HEALTH     01-Aug-24   1,941.3     2,056.8     901    ₹104,049      +5.9%       -0.3%     
  SUNPHARMA   19     HEALTH     01-Aug-24   1,688.4     1,780.3     1036   ₹95,198       +5.4%       +0.8%     
  HCLTECH     15     IT         01-Oct-24   1,693.6     1,756.4     1160   ₹72,746       +3.7%       +1.0%     
  INFY        55     IT         01-Aug-24   1,741.4     1,787.1     1005   ₹45,915       +2.6%       +1.0%     
  UNITDSPR    22     FMCG       01-Oct-24   1,589.7     1,512.0     1236   ₹-96,050      -4.9%       +2.7%     
  VOLTAS      12     CON DUR    01-Oct-24   1,838.5     1,706.2     1069   ₹-141,422     -7.2%       +1.4%     

  AFTER: Invested ₹35,412,335 | Cash ₹1,598,868 | Total ₹37,011,203 | Positions 17/20 | Slot ₹1,850,890

========================================================================
  REBALANCE #61  —  01 Feb 2025
  NAV: ₹34,436,503  |  Slot: ₹1,721,825  |  Cash: ₹1,598,868
========================================================================

  [REGIME OFF] Nifty 200 13,064.5 < EMA200 13,264.1 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       20     CONSUMP    03-Oct-23   2,053.9     6,176.8     526    ₹2,168,627    +200.7%     +2.9%     
  BAJAJ-AUTO  60     AUTO       01-Jun-23   4,298.5     8,805.4     222    ₹1,000,534    +104.8%     +5.4%     
  BSE         3      FIN SVC    03-Jun-24   894.9       1,795.7     1788   ₹1,610,673    +100.7%     -1.5%     
  BOSCHLTD    105    AUTO       01-Dec-23   21,526.2    28,305.1    56     ₹379,618      +31.5% ⚠    -6.3%     
  TORNTPHARM  21     HEALTH     01-Feb-24   2,440.9     3,173.5     549    ₹402,191      +30.0%      -1.6%     
  PERSISTENT  15     IT         01-Aug-24   4,750.7     5,896.4     368    ₹421,628      +24.1%      -2.5%     
  KALYANKJIL  87     CON DUR    01-Apr-24   423.7       504.0       3549   ₹284,870      +18.9% ⚠    -5.2%     
  LUPIN       30     HEALTH     01-Aug-24   1,941.3     2,043.5     901    ₹92,140       +5.3%       -3.1%     
  M&M         4      AUTO       02-Dec-24   2,960.8     3,020.0     625    ₹37,024       +2.0%       +4.0%     
  SUNPHARMA   39     HEALTH     01-Aug-24   1,688.4     1,714.9     1036   ₹27,534       +1.6%       -1.9%     
  INFY        25     IT         01-Aug-24   1,741.4     1,760.0     1005   ₹18,733       +1.1%       -1.3%     
  INDHOTEL    5      CONSUMP    02-Dec-24   795.2       795.6       2327   ₹924          +0.0%       +1.3%     
  HCLTECH     50     IT         01-Oct-24   1,693.6     1,605.9     1160   ₹-101,776     -5.2%       -5.1%     
  UNITDSPR    17     FMCG       01-Oct-24   1,589.7     1,478.3     1236   ₹-137,641     -7.0%       +1.9%     
  OFSS        76     IT         01-Aug-24   10,120.6    8,249.1     172    ₹-321,901     -18.5% ⚠    -11.8%    
  CGPOWER     74     ENERGY     02-Dec-24   752.0       609.4       2461   ₹-350,941     -19.0% ⚠    -4.9%     
  VOLTAS      88     CON DUR    01-Oct-24   1,838.5     1,312.5     1069   ₹-562,296     -28.6% ⚠    -11.6%    
  ⚠  WAZ < 0 (momentum below universe mean): CGPOWER, OFSS, KALYANKJIL, VOLTAS, BOSCHLTD

  AFTER: Invested ₹32,837,635 | Cash ₹1,598,868 | Total ₹34,436,503 | Positions 17/20 | Slot ₹1,721,825

========================================================================
  REBALANCE #62  —  01 Apr 2025
  NAV: ₹32,259,374  |  Slot: ₹1,612,969  |  Cash: ₹1,598,868
========================================================================
  [SECTOR CAP≤4] dropped: CHOLAFIN, SHRIRAMFIN, HDFCLIFE, BAJAJHLDNG

  [REGIME OFF] Nifty 200 12,809.3 < EMA200 13,066.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       54     CONSUMP    03-Oct-23   2,053.9     5,565.3     526    ₹1,846,982    +171.0%     +6.8%     
  BSE         6      FIN SVC    03-Jun-24   894.9       1,816.3     1788   ₹1,647,479    +103.0%     +15.8%    
  BAJAJ-AUTO  105    AUTO       01-Jun-23   4,298.5     7,687.3     222    ₹752,320      +78.8% ⚠    +1.0%     
  TORNTPHARM  31     HEALTH     01-Feb-24   2,440.9     3,150.0     549    ₹389,264      +29.0%      +0.8%     
  BOSCHLTD    —      AUTO       01-Dec-23   21,526.2    27,510.7    56     ₹335,133      +27.8%      +1.1%     
  PERSISTENT  60     IT         01-Aug-24   4,750.7     5,179.0     368    ₹157,593      +9.0% ⚠     -3.6%     
  KALYANKJIL  114    CON DUR    01-Apr-24   423.7       456.7       3549   ₹117,138      +7.8% ⚠     -1.1%     
  INDHOTEL    29     CONSUMP    02-Dec-24   795.2       799.8       2327   ₹10,857       +0.6%       +2.7%     
  LUPIN       72     HEALTH     01-Aug-24   1,941.3     1,943.3     901    ₹1,836        +0.1% ⚠     -3.4%     
  SUNPHARMA   86     HEALTH     01-Aug-24   1,688.4     1,681.9     1036   ₹-6,745       -0.4% ⚠     -0.7%     
  M&M         39     AUTO       02-Dec-24   2,960.8     2,589.3     625    ₹-232,203     -12.5%      -3.5%     
  UNITDSPR    56     FMCG       01-Oct-24   1,589.7     1,387.6     1236   ₹-249,791     -12.7% ⚠    +2.6%     
  HCLTECH     125    IT         01-Oct-24   1,693.6     1,450.8     1160   ₹-281,738     -14.3% ⚠    -3.9%     
  INFY        118    IT         01-Aug-24   1,741.4     1,451.2     1005   ₹-291,641     -16.7% ⚠    -6.5%     
  CGPOWER     —      ENERGY     02-Dec-24   752.0       613.6       2461   ₹-340,589     -18.4%      -1.2%     
  VOLTAS      82     CON DUR    01-Oct-24   1,838.5     1,340.3     1069   ₹-532,518     -27.1% ⚠    -4.3%     
  OFSS        131    IT         01-Aug-24   10,120.6    7,036.0     172    ₹-530,566     -30.5% ⚠    -3.4%     
  ⚠  WAZ < 0 (momentum below universe mean): UNITDSPR, PERSISTENT, LUPIN, VOLTAS, SUNPHARMA, BAJAJ-AUTO, KALYANKJIL, INFY, HCLTECH, OFSS

  AFTER: Invested ₹30,660,506 | Cash ₹1,598,868 | Total ₹32,259,374 | Positions 17/20 | Slot ₹1,612,969

========================================================================
  REBALANCE #63  —  02 Jun 2025
  NAV: ₹35,321,486  |  Slot: ₹1,766,074  |  Cash: ₹1,598,868
========================================================================
  [SECTOR CAP≤4] dropped: LTF, MOTILALOFS

  EXITS (14)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TRENT       65     CONSUMP    03-Oct-23   2,053.9     5,610.0     526    ₹1,870,472    +173.1%   608d  
  BAJAJ-AUTO  —      AUTO       01-Jun-23   4,298.5     8,187.8     222    ₹863,440      +90.5%    732d  
  BOSCHLTD    —      AUTO       01-Dec-23   21,526.2    30,811.8    56     ₹519,994      +43.1%    549d  
  TORNTPHARM  59     HEALTH     01-Feb-24   2,440.9     3,094.5     549    ₹358,794      +26.8%    487d  
  PERSISTENT  56     IT         01-Aug-24   4,750.7     5,484.5     368    ₹270,028      +15.4%    305d  
  LUPIN       69     HEALTH     01-Aug-24   1,941.3     1,949.0     901    ₹6,940        +0.4%     305d  
  M&M         45     AUTO       02-Dec-24   2,960.8     2,970.1     625    ₹5,828        +0.3%     182d  
  SUNPHARMA   70     HEALTH     01-Aug-24   1,688.4     1,658.3     1036   ₹-31,111      -1.8%     305d  
  INDHOTEL    42     CONSUMP    02-Dec-24   795.2       777.8       2327   ₹-40,308      -2.2%     182d  
  HCLTECH     67     IT         01-Oct-24   1,693.6     1,564.5     1160   ₹-149,807     -7.6%     244d  
  CGPOWER     —      ENERGY     02-Dec-24   752.0       677.6       2461   ₹-183,187     -9.9%     182d  
  INFY        124    IT         01-Aug-24   1,741.4     1,498.0     1005   ₹-244,670     -14.0%    305d  
  OFSS        82     IT         01-Aug-24   10,120.6    8,038.0     172    ₹-358,210     -20.6%    305d  
  VOLTAS      125    CON DUR    01-Oct-24   1,838.5     1,235.2     1069   ₹-644,903     -32.8%    244d  

  ENTRIES (15)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SOLARINDS   1      DEFENCE    4.524    0.31   +68.3%    +83.8%    16,284.3    108    ₹1,758,699    +10.5%    
  BHARTIHEXA  3      IT         3.050    0.05   +80.6%    +46.6%    1,840.5     959    ₹1,765,001    +7.6%     
  MAZDOCK     4      DEFENCE    2.804    0.42   +118.1%   +57.5%    3,363.6     525    ₹1,765,873    +1.6%     
  HDFCLIFE    5      FIN SVC    2.709    0.20   +36.3%    +24.3%    761.9       2318   ₹1,766,017    +1.4%     
  BEL         6      DEFENCE    2.652    0.31   +32.7%    +52.0%    385.0       4587   ₹1,765,831    +6.9%     
  ICICIBANK   7      PVT BNK    2.552    0.06   +29.5%    +19.1%    1,439.4     1226   ₹1,764,696    +0.9%     
  DIVISLAB    8      HEALTH     2.343    0.28   +54.6%    +14.7%    6,509.4     271    ₹1,764,037    +2.0%     
  BHARTIARTL  9      CONSUMP    2.253    0.13   +34.7%    +15.8%    1,838.7     960    ₹1,765,179    +0.7%     
  SBILIFE     10     FIN SVC    2.241    0.25   +28.1%    +21.5%    1,800.0     981    ₹1,765,798    +2.0%     
  HDFCBANK    11     PVT BNK    2.196    0.02   +26.5%    +15.2%    937.7       1883   ₹1,765,610    +0.5%     
  SUZLON      13     ENERGY     2.090    0.64   +57.5%    +31.1%    71.2        24807  ₹1,766,010    +13.2%    
  HAL         14     DEFENCE    1.881    0.45   -1.5%     +49.7%    4,959.1     356    ₹1,765,424    +3.7%     
  HDFCAMC     15     FIN SVC    1.870    0.39   +25.2%    +27.4%    2,315.5     762    ₹1,764,448    +3.0%     
  PAGEIND     18     MFG        1.725    0.30   +28.4%    +12.0%    45,270.3    39     ₹1,765,541    -1.0%     
  BRITANNIA   19     FMCG       1.707    0.14   +8.9%     +16.7%    5,532.5     319    ₹1,764,866    +2.5%     

  HOLDS (3)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BSE         2      FIN SVC    03-Jun-24   894.9       2,693.3     1788   ₹3,215,606    +201.0%     +12.3%    
  KALYANKJIL  40     CON DUR    01-Apr-24   423.7       555.0       3549   ₹466,049      +31.0%      +1.4%     
  UNITDSPR    12     FMCG       01-Oct-24   1,589.7     1,533.0     1236   ₹-70,032      -3.6%       +0.6%     

  AFTER: Invested ₹35,153,253 | Cash ₹136,798 | Total ₹35,290,052 | Positions 18/20 | Slot ₹1,766,074

========================================================================
  REBALANCE #64  —  01 Aug 2025
  NAV: ₹34,113,796  |  Slot: ₹1,705,690  |  Cash: ₹136,798
========================================================================
  [SECTOR CAP≤4] dropped: JIOFIN, LTF

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PAGEIND     36     MFG        02-Jun-25   45,270.3    46,152.7    39     ₹34,415       +1.9%     60d   
  SBILIFE     42     FIN SVC    02-Jun-25   1,800.0     1,793.7     981    ₹-6,172       -0.3%     60d   
  HDFCLIFE    43     FIN SVC    02-Jun-25   761.9       739.1       2318   ₹-52,863      -3.0%     60d   
  HAL         85     DEFENCE    02-Jun-25   4,959.1     4,387.3     356    ₹-203,528     -11.5%    60d   
  UNITDSPR    128    FMCG       01-Oct-24   1,589.7     1,316.4     1236   ₹-337,716     -17.2%    304d  
  MAZDOCK     84     DEFENCE    02-Jun-25   3,363.6     2,705.0     525    ₹-345,766     -19.6%    60d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MUTHOOTFIN  3      FIN SVC    2.820    0.45   +45.1%    +15.3%    2,571.0     663    ₹1,704,547    -1.5%     
  MOTILALOFS  4      FIN SVC    2.749    1.00   +45.4%    +38.7%    914.3       1865   ₹1,705,207    +1.0%     
  ETERNAL     5      CONSUMP    2.647    0.74   +34.2%    +31.0%    304.8       5597   ₹1,705,686    +5.8%     
  TORNTPHARM  10     HEALTH     2.403    0.30   +19.4%    +12.7%    3,645.9     467    ₹1,702,638    +3.8%     
  CUMMINSIND  12     INFRA      2.269    0.75   -4.8%     +29.1%    3,552.5     480    ₹1,705,222    +1.6%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BSE         1      FIN SVC    03-Jun-24   894.9       2,411.3     1788   ₹2,711,390    +169.5%     -4.0%     
  KALYANKJIL  31     CON DUR    01-Apr-24   423.7       581.1       3549   ₹558,761      +37.2%      -1.1%     
  HDFCAMC     2      FIN SVC    02-Jun-25   2,315.5     2,747.4     762    ₹329,077      +18.7%      +3.0%     
  HDFCBANK    9      PVT BNK    02-Jun-25   937.7       989.7       1883   ₹98,053       +5.6%       +0.7%     
  BRITANNIA   28     FMCG       02-Jun-25   5,532.5     5,723.0     319    ₹60,771       +3.4%       +1.3%     
  BHARTIARTL  16     CONSUMP    02-Jun-25   1,838.7     1,884.4     960    ₹43,845       +2.5%       -2.0%     
  ICICIBANK   13     PVT BNK    02-Jun-25   1,439.4     1,460.3     1226   ₹25,670       +1.5%       +0.7%     
  BHARTIHEXA  11     IT         02-Jun-25   1,840.5     1,844.6     959    ₹3,970        +0.2%       +2.0%     
  DIVISLAB    15     HEALTH     02-Jun-25   6,509.4     6,361.5     271    ₹-40,070      -2.3%       -4.2%     
  BEL         6      DEFENCE    02-Jun-25   385.0       374.7       4587   ₹-46,937      -2.7%       -4.9%     
  SUZLON      25     ENERGY     02-Jun-25   71.2        65.9        24807  ₹-129,989     -7.4%       +2.4%     
  SOLARINDS   22     DEFENCE    02-Jun-25   16,284.3    13,807.0    108    ₹-267,543     -15.2%      -8.1%     

  AFTER: Invested ₹32,618,429 | Cash ₹1,485,247 | Total ₹34,103,676 | Positions 17/20 | Slot ₹1,705,690

========================================================================
  REBALANCE #65  —  01 Oct 2025
  NAV: ₹32,970,175  |  Slot: ₹1,648,509  |  Cash: ₹1,485,247
========================================================================
  [SECTOR CAP≤4] dropped: ASHOKLEY

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HDFCAMC     12     FIN SVC    02-Jun-25   2,315.5     2,718.3     762    ₹306,893      +17.4%    121d  
  KALYANKJIL  119    CON DUR    01-Apr-24   423.7       465.3       3549   ₹147,611      +9.8%     548d  
  ETERNAL     —      CONSUMP    01-Aug-25   304.8       329.0       5597   ₹135,727      +8.0%     61d   
  BHARTIARTL  66     CONSUMP    02-Jun-25   1,838.7     1,867.6     960    ₹27,717       +1.6%     121d  
  MOTILALOFS  —      FIN SVC    01-Aug-25   914.3       892.9       1865   ₹-39,961      -2.3%     61d   
  ICICIBANK   55     PVT BNK    02-Jun-25   1,439.4     1,372.0     1226   ₹-82,624      -4.7%     121d  
  BHARTIHEXA  63     IT         02-Jun-25   1,840.5     1,660.8     959    ₹-172,294     -9.8%     121d  
  DIVISLAB    97     HEALTH     02-Jun-25   6,509.4     5,710.0     271    ₹-216,627     -12.3%    121d  
  SOLARINDS   89     DEFENCE    02-Jun-25   16,284.3    13,374.0    108    ₹-314,307     -17.9%    121d  
  SUZLON      116    ENERGY     02-Jun-25   71.2        55.2        24807  ₹-397,160     -22.5%    121d  

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  EICHERMOT   1      AUTO       3.798    0.63   +42.4%    +24.3%    7,021.5     234    ₹1,643,031    +3.0%     
  FORTIS      2      HEALTH     3.709    0.61   +62.4%    +25.0%    989.7       1665   ₹1,647,767    +3.4%     
  LTF         4      FIN SVC    3.198    1.08   +40.6%    +25.4%    255.9       6440   ₹1,648,285    +7.6%     
  INDIANB     5      PSU BNK    2.745    0.57   +41.9%    +13.3%    721.7       2284   ₹1,648,309    +4.8%     
  TVSMOTOR    6      AUTO       2.646    0.80   +20.1%    +19.3%    3,445.8     478    ₹1,647,071    +0.7%     
  HEROMOTOCO  7      AUTO       2.569    0.69   -6.6%     +29.9%    5,328.6     309    ₹1,646,549    +2.3%     
  BOSCHLTD    8      AUTO       2.432    0.56   +4.6%     +19.7%    38,320.0    43     ₹1,647,760    -2.3%     
  NYKAA       9      CONSUMP    2.403    0.77   +20.0%    +14.0%    241.3       6832   ₹1,648,288    +2.1%     
  BAJFINANCE  10     FIN SVC    2.305    0.53   +27.9%    +7.0%     981.7       1679   ₹1,648,216    +0.7%     
  GODFRYPHLP  12     FMCG       2.003    0.75   +43.6%    +15.1%    3,357.6     490    ₹1,645,205    -1.7%     
  COROMANDEL  14     MFG        1.934    0.67   +38.7%    -0.6%     2,242.2     735    ₹1,647,989    -0.5%     

  HOLDS (7)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BSE         31     FIN SVC    03-Jun-24   894.9       2,081.4     1788   ₹2,121,528    +132.6%     -3.5%     
  MUTHOOTFIN  3      FIN SVC    01-Aug-25   2,571.0     3,118.1     663    ₹362,776      +21.3%      +5.7%     
  CUMMINSIND  14     INFRA      01-Aug-25   3,552.5     3,831.6     480    ₹133,935      +7.9%       -2.8%     
  BRITANNIA   39     FMCG       02-Jun-25   5,532.5     5,966.5     319    ₹138,447      +7.8%       -0.3%     
  BEL         18     DEFENCE    02-Jun-25   385.0       404.8       4587   ₹90,880       +5.1%       +2.4%     
  HDFCBANK    37     PVT BNK    02-Jun-25   937.7       949.5       1883   ₹22,384       +1.3%       +0.4%     
  TORNTPHARM  25     HEALTH     01-Aug-25   3,645.9     3,535.8     467    ₹-51,428      -3.0%       -0.6%     

  AFTER: Invested ₹32,945,722 | Cash ₹2,938 | Total ₹32,948,661 | Positions 18/20 | Slot ₹1,648,509

========================================================================
  REBALANCE #66  —  01 Dec 2025
  NAV: ₹36,187,012  |  Slot: ₹1,809,351  |  Cash: ₹2,938
========================================================================
  [SECTOR CAP≤4] dropped: ASHOKLEY, PNB

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BSE         13     FIN SVC    03-Jun-24   894.9       2,886.6     1788   ₹3,561,226    +222.6%   546d  
  LTF         1      FIN SVC    01-Oct-25   255.9       306.0       6440   ₹322,352      +19.6%    61d   
  COROMANDEL  53     MFG        01-Oct-25   2,242.2     2,382.9     735    ₹103,448      +6.3%     61d   
  BRITANNIA   62     FMCG       02-Jun-25   5,532.5     5,813.5     319    ₹89,640       +5.1%     182d  
  TORNTPHARM  54     HEALTH     01-Aug-25   3,645.9     3,704.1     467    ₹27,172       +1.6%     122d  
  BOSCHLTD    115    AUTO       01-Oct-25   38,320.0    36,335.0    43     ₹-85,355      -5.2%     61d   
  FORTIS      58     HEALTH     01-Oct-25   989.7       904.8       1665   ₹-141,192     -8.6%     61d   
  GODFRYPHLP  89     FMCG       01-Oct-25   3,357.6     2,837.9     490    ₹-254,634     -15.5%    61d   

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  CANBK       2      PSU BNK    3.274    0.94   +53.2%    +44.9%    145.7       12418  ₹1,809,227    +3.7%     
  M&MFIN      4      FIN SVC    2.957    0.89   +39.5%    +44.9%    367.9       4918   ₹1,809,332    +9.0%     
  BANKINDIA   6      PSU BNK    2.394    0.93   +41.6%    +33.5%    142.6       12689  ₹1,809,228    +1.8%     
  SBIN        7      PSU BNK    2.336    0.65   +18.3%    +21.3%    955.9       1892   ₹1,808,507    +1.1%     
  MARUTI      10     AUTO       2.303    0.55   +48.7%    +8.8%     16,097.0    112    ₹1,802,864    +1.2%     
  ADANIENSOL  13     ENERGY     2.215    1.15   +66.3%    +30.7%    999.1       1810   ₹1,808,371    +1.5%     
  RELIANCE    16     OIL&GAS    1.925    0.82   +21.4%    +15.4%    1,558.9     1160   ₹1,808,315    +2.7%     
  BHARTIARTL  17     CONSUMP    1.901    0.59   +33.6%    +10.6%    2,089.7     865    ₹1,807,590    -0.6%     
  ADANIPORTS  18     INFRA      1.899    0.91   +36.2%    +16.6%    1,524.1     1187   ₹1,809,079    +2.9%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  2      FIN SVC    01-Aug-25   2,571.0     3,779.1     663    ₹801,026      +47.0%      +6.6%     
  CUMMINSIND  25     INFRA      01-Aug-25   3,552.5     4,523.6     480    ₹466,121      +27.3%      +4.5%     
  INDIANB     4      PSU BNK    01-Oct-25   721.7       868.8       2284   ₹336,125      +20.4%      +2.6%     
  HEROMOTOCO  11     AUTO       01-Oct-25   5,328.6     6,175.1     309    ₹261,569      +15.9%      +7.1%     
  NYKAA       16     CONSUMP    01-Oct-25   241.3       264.9       6832   ₹161,508      +9.8%       +0.6%     
  BEL         30     DEFENCE    02-Jun-25   385.0       415.5       4587   ₹139,981      +7.9%       +0.3%     
  TVSMOTOR    17     AUTO       01-Oct-25   3,445.8     3,649.0     478    ₹97,172       +5.9%       +4.5%     
  HDFCBANK    42     PVT BNK    02-Jun-25   937.7       985.8       1883   ₹90,643       +5.1%       +0.5%     
  BAJFINANCE  10     FIN SVC    01-Oct-25   981.7       1,014.9     1679   ₹55,736       +3.4%       -0.3%     
  EICHERMOT   7      AUTO       01-Oct-25   7,021.5     7,125.5     234    ₹24,336       +1.5%       +1.6%     

  AFTER: Invested ₹35,529,406 | Cash ₹638,285 | Total ₹36,167,690 | Positions 19/20 | Slot ₹1,809,351

========================================================================
  REBALANCE #67  —  02 Feb 2026
  NAV: ₹34,167,923  |  Slot: ₹1,708,396  |  Cash: ₹638,285
========================================================================
  [SECTOR CAP≤4] dropped: UNIONBANK, HINDZINC

  [REGIME OFF] Nifty 200 13,949.8 < EMA200 14,017.7 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  11     FIN SVC    01-Aug-25   2,571.0     3,505.5     663    ₹619,573      +36.3%      -8.4%     
  CUMMINSIND  33     INFRA      01-Aug-25   3,552.5     4,073.8     480    ₹250,190      +14.7%      -0.1%     
  BEL         15     DEFENCE    02-Jun-25   385.0       437.2       4587   ₹239,782      +13.6%      +4.1%     
  INDIANB     21     PSU BNK    01-Oct-25   721.7       817.4       2284   ₹218,604      +13.3%      -3.0%     
  SBIN        6      PSU BNK    01-Dec-25   955.9       1,010.5     1892   ₹103,333      +5.7%       -0.3%     
  TVSMOTOR    10     AUTO       01-Oct-25   3,445.8     3,633.1     478    ₹89,551       +5.4%       -0.6%     
  HEROMOTOCO  26     AUTO       01-Oct-25   5,328.6     5,515.5     309    ₹57,739       +3.5%       -0.1%     
  BANKINDIA   20     PSU BNK    01-Dec-25   142.6       146.9       12689  ₹54,434       +3.0%       -3.1%     
  EICHERMOT   28     AUTO       01-Oct-25   7,021.5     6,985.5     234    ₹-8,424       -0.5%       -2.8%     
  NYKAA       36     CONSUMP    01-Oct-25   241.3       237.6       6832   ₹-24,868      -1.5%       -3.3%     
  HDFCBANK    66     PVT BNK    02-Jun-25   937.7       913.0       1883   ₹-46,432      -2.6% ⚠     -1.2%     
  CANBK       9      PSU BNK    01-Dec-25   145.7       141.8       12418  ₹-48,446      -2.7%       -3.6%     
  M&MFIN      23     FIN SVC    01-Dec-25   367.9       353.6       4918   ₹-70,327      -3.9%       -2.9%     
  BHARTIARTL  54     CONSUMP    01-Dec-25   2,089.7     1,965.4     865    ₹-107,519     -5.9%       -2.2%     
  ADANIPORTS  43     INFRA      01-Dec-25   1,524.1     1,397.2     1187   ₹-150,589     -8.3%       -0.8%     
  BAJFINANCE  82     FIN SVC    01-Oct-25   981.7       898.2       1679   ₹-140,174     -8.5% ⚠     -4.4%     
  MARUTI      81     AUTO       01-Dec-25   16,097.0    14,384.0    112    ₹-191,856     -10.6% ⚠    -7.8%     
  RELIANCE    72     OIL&GAS    01-Dec-25   1,558.9     1,384.0     1160   ₹-202,874     -11.2% ⚠    -3.1%     
  ADANIENSOL  78     ENERGY     01-Dec-25   999.1       884.6       1810   ₹-207,245     -11.5% ⚠    -3.7%     
  ⚠  WAZ < 0 (momentum below universe mean): HDFCBANK, RELIANCE, ADANIENSOL, MARUTI, BAJFINANCE

  AFTER: Invested ₹33,529,639 | Cash ₹638,285 | Total ₹34,167,923 | Positions 19/20 | Slot ₹1,708,396

========================================================================
  REBALANCE #68  —  01 Apr 2026
  NAV: ₹32,573,535  |  Slot: ₹1,628,677  |  Cash: ₹638,285
========================================================================

  [REGIME OFF] Nifty 200 12,720.3 < EMA200 13,909.6 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  CUMMINSIND  7      INFRA      01-Aug-25   3,552.5     4,609.1     480    ₹507,146      +29.7%      -0.3%     
  MUTHOOTFIN  45     FIN SVC    01-Aug-25   2,571.0     3,228.7     663    ₹436,081      +25.6%      -1.7%     
  INDIANB     5      PSU BNK    01-Oct-25   721.7       869.5       2284   ₹337,579      +20.5%      -0.2%     
  BEL         14     DEFENCE    02-Jun-25   385.0       418.7       4587   ₹154,746      +8.8%       -2.0%     
  SBIN        11     PSU BNK    01-Dec-25   955.9       999.8       1892   ₹83,075       +4.6%       -4.7%     
  NYKAA       33     CONSUMP    01-Oct-25   241.3       240.0       6832   ₹-8,813       -0.5%       -2.1%     
  TVSMOTOR    23     AUTO       01-Oct-25   3,445.8     3,425.8     478    ₹-9,539       -0.6%       -2.8%     
  EICHERMOT   37     AUTO       01-Oct-25   7,021.5     6,825.5     234    ₹-45,864      -2.8%       -3.2%     
  BANKINDIA   27     PSU BNK    01-Dec-25   142.6       137.2       12689  ₹-68,073      -3.8%       -6.2%     
  HEROMOTOCO  26     AUTO       01-Oct-25   5,328.6     5,122.0     309    ₹-63,851      -3.9%       -3.5%     
  ADANIENSOL  48     ENERGY     01-Dec-25   999.1       956.6       1810   ₹-76,925      -4.3%       -2.5%     
  ADANIPORTS  47     INFRA      01-Dec-25   1,524.1     1,379.6     1187   ₹-171,511     -9.5%       -0.4%     
  RELIANCE    70     OIL&GAS    01-Dec-25   1,558.9     1,362.9     1160   ₹-227,353     -12.6% ⚠    -1.6%     
  BHARTIARTL  84     CONSUMP    01-Dec-25   2,089.7     1,781.9     865    ₹-266,247     -14.7% ⚠    -3.3%     
  CANBK       44     PSU BNK    01-Dec-25   145.7       123.2       12418  ₹-278,897     -15.4%      -6.8%     
  BAJFINANCE  99     FIN SVC    01-Oct-25   981.7       812.3       1679   ₹-284,354     -17.3% ⚠    -6.7%     
  M&MFIN      —      FIN SVC    01-Dec-25   367.9       289.7       4918   ₹-384,588     -21.3%      -10.3%    
  HDFCBANK    138    PVT BNK    02-Jun-25   937.7       730.2       1883   ₹-390,693     -22.1% ⚠    -8.1%     
  MARUTI      109    AUTO       01-Dec-25   16,097.0    12,509.0    112    ₹-401,856     -22.3% ⚠    -4.6%     
  ⚠  WAZ < 0 (momentum below universe mean): RELIANCE, BHARTIARTL, BAJFINANCE, MARUTI, HDFCBANK

  AFTER: Invested ₹31,935,250 | Cash ₹638,285 | Total ₹32,573,535 | Positions 19/20 | Slot ₹1,628,677

========================================================================
  REBALANCE #69  —  01 Jun 2026
  NAV: ₹34,513,820  |  Slot: ₹1,725,691  |  Cash: ₹638,285
========================================================================
  [SECTOR CAP≤4] dropped: ADANIGREEN, PREMIERENE, CGPOWER

  [REGIME OFF] Nifty 200 13,528.3 < EMA200 13,828.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  CUMMINSIND  6      INFRA      01-Aug-25   3,552.5     5,680.5     480    ₹1,021,418    +59.9%      +3.4%     
  ADANIENSOL  7      ENERGY     01-Dec-25   999.1       1,496.5     1810   ₹900,294      +49.8%      +6.7%     
  MUTHOOTFIN  39     FIN SVC    01-Aug-25   2,571.0     3,246.4     663    ₹447,816      +26.3%      -3.3%     
  ADANIPORTS  20     INFRA      01-Dec-25   1,524.1     1,776.0     1187   ₹299,051      +16.5%      +1.4%     
  NYKAA       45     CONSUMP    01-Oct-25   241.3       266.7       6832   ₹173,806      +10.5%      -0.3%     
  INDIANB     72     PSU BNK    01-Oct-25   721.7       792.9       2284   ₹162,583      +9.9% ⚠     -3.3%     
  BEL         85     DEFENCE    02-Jun-25   385.0       407.2       4587   ₹101,995      +5.8% ⚠     -3.6%     
  EICHERMOT   62     AUTO       01-Oct-25   7,021.5     7,100.5     234    ₹18,486       +1.1%       -0.9%     
  SBIN        99     PSU BNK    01-Dec-25   955.9       954.1       1892   ₹-3,350       -0.2% ⚠     -2.5%     
  TVSMOTOR    86     AUTO       01-Oct-25   3,445.8     3,344.4     478    ₹-48,448      -2.9% ⚠     -2.8%     
  BANKINDIA   89     PSU BNK    01-Dec-25   142.6       136.7       12689  ₹-74,134      -4.1% ⚠     -1.3%     
  HEROMOTOCO  98     AUTO       01-Oct-25   5,328.6     4,819.9     309    ₹-157,200     -9.5% ⚠     -4.1%     
  BAJFINANCE  110    FIN SVC    01-Oct-25   981.7       883.6       1679   ₹-164,621     -10.0% ⚠    -3.4%     
  BHARTIARTL  —      CONSUMP    01-Dec-25   2,089.7     1,810.6     865    ₹-241,421     -13.4%      -2.2%     
  CANBK       96     PSU BNK    01-Dec-25   145.7       123.9       12418  ₹-270,963     -15.0% ⚠    -2.8%     
  RELIANCE    106    OIL&GAS    01-Dec-25   1,558.9     1,313.9     1160   ₹-284,162     -15.7% ⚠    -2.8%     
  MARUTI      109    AUTO       01-Dec-25   16,097.0    12,946.0    112    ₹-352,912     -19.6% ⚠    -1.8%     
  M&MFIN      103    FIN SVC    01-Dec-25   367.9       295.1       4918   ₹-358,030     -19.8% ⚠    -4.5%     
  HDFCBANK    149    PVT BNK    02-Jun-25   937.7       730.6       1883   ₹-389,860     -22.1% ⚠    -3.0%     
  ⚠  WAZ < 0 (momentum below universe mean): INDIANB, BEL, TVSMOTOR, BANKINDIA, CANBK, HEROMOTOCO, SBIN, M&MFIN, RELIANCE, MARUTI, BAJFINANCE, HDFCBANK

  AFTER: Invested ₹33,875,535 | Cash ₹638,285 | Total ₹34,513,820 | Positions 19/20 | Slot ₹1,725,691

========================================================================
  FINAL SUMMARY
========================================================================
  Period        : 2015-01-01 → 2026-07-01  (11.5 years)
  Starting Cap  : ₹2,000,000
  Final Value   : ₹35,812,484
  Total Return  : +1690.6%  (on total invested)
  CAGR          : +28.5%

  Closed Trades : 299  |  Open: 19
  Win Rate      : 59.9%  (179W / 120L)
  Profit Factor : 5.20
  Avg hold      : 230 days
  Total charges : ₹672,793
  Closed net P&L: ₹31,557,080
  Open unreal   : ₹2,079,011

  YEAR-BY-YEAR:
  2015  +  6.8%  ██████
  2016  + 17.3%  █████████████████
  2017  + 52.0%  ████████████████████████████████████████
  2018  +  7.0%  ███████
  2019  + 16.4%  ████████████████
  2020  + 27.7%  ███████████████████████████
  2021  + 94.1%  ████████████████████████████████████████
  2022  + 49.6%  ████████████████████████████████████████
  2023  + 37.6%  █████████████████████████████████████
  2024  + 53.0%  ████████████████████████████████████████
  2025  -  2.2%  ░░
  2026  -  4.6%  ░░░░

  Rebalance NAV exported → mom20_rebal.csv (69 rows)
