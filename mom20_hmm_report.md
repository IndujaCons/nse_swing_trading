=== Mom20 — 2-Monthly Rebalance, β≤1.2 | Regime ON [HMM] | Sector≤4 ===
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

===========================================================================================
  MOM20 PIT BACKTEST  |  NAV/20 slot  |  Monthly Rebalance  |  Beta≤1.2  |  Regime ON [HMM]
===========================================================================================

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
  BRITANNIA   2      FMCG       01-Apr-15   946.0       1,370.3     112    ₹47,522       +44.9%      +7.7%     Model is not converging.  Current: 1709.7788027446038 is not greater than 1709.7852711722128. Delta is -0.006468427608979255
Model is not converging.  Current: 1708.7094487716008 is not greater than 1708.7905148747955. Delta is -0.0810661031946438
Model is not converging.  Current: 1709.676965144217 is not greater than 1709.7934771597697. Delta is -0.11651201555264379

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

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SRF         66     MFG        02-Feb-15   181.9       215.0       549    ₹18,179       +18.2%    241d  
  EMAMILTD    35     FMCG       01-Apr-15   420.8       496.1       252    ₹18,982       +17.9%    183d  
  BBTC        49     FMCG       02-Feb-15   438.1       468.5       228    ₹6,939        +6.9%     241d  
  EICHERMOT   65     AUTO       01-Jun-15   1,740.7     1,695.1     59     ₹-2,688       -2.6%     122d  
  WHIRLPOOL   55     CON DUR    02-Feb-15   693.1       664.0       144    ₹-4,190       -4.2%     241d  
  MARICO      69     FMCG       01-Jun-15   190.3       170.9       546    ₹-10,596      -10.2%    122d  
  INDUSTOWER  102    INFRA      01-Jun-15   327.7       280.3       317    ₹-15,024      -14.5%    122d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HONAUT      7      MFG        2.744    0.62   +75.8%    +21.7%    9,045.6     11     ₹99,501       +2.6%     
  J&KBANK     8      PVT BNK    2.570    0.00   +1.3%     +1.3%     149.2       704    ₹105,053      +0.0%     
  CUMMINSIND  9      INFRA      2.488    0.60   +60.3%    +19.3%    904.6       116    ₹104,929      -0.6%     
  DIVISLAB    10     HEALTH     2.432    0.53   +32.7%    +22.8%    1,041.5     100    ₹104,148      +2.0%     
  AMARAJABAT  14     AUTO       2.307    -0.04  +69.6%    +17.1%    931.1       112    ₹104,285      +4.0%     
  REPCOHOME   15     FIN SVC    2.277    0.64   +69.8%    +12.6%    664.2       158    ₹104,937      +4.7%     
  INFY        17     IT         2.217    0.38   +29.7%    +18.7%    434.2       242    ₹105,085      +5.1%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BRITANNIA   3      FMCG       01-Apr-15   946.0       1,364.1     112    ₹46,835       +44.2%      +4.4%     
  ASHOKLEY    4      AUTO       02-Feb-15   26.9        38.7        3716   ₹43,684       +43.7%      +5.4%     
  ABBOTINDIA  1      HEALTH     03-Aug-15   4,000.7     5,121.9     27     ₹30,274       +28.0%      +5.2%     
  AJANTPHARM  23     HEALTH     01-Apr-15   737.1       913.9       144    ₹25,462       +24.0%      +3.4%     
  BAJFINANCE  25     FIN SVC    01-Jun-15   42.0        49.1        2479   ₹17,713       +17.0%      +1.3%     
  IBULHSGFIN  5      FIN SVC    03-Aug-15   498.5       524.9       221    ₹5,833        +5.3%       +7.0%     
  MARUTI      6      AUTO       03-Aug-15   4,018.0     4,182.0     27     ₹4,429        +4.1%       +3.1%     
  RAJESHEXPO  2      CON DUR    03-Aug-15   561.3       548.4       196    ₹-2,516       -2.3%       +12.6%    
  APLLTD      24     HEALTH     03-Aug-15   657.5       626.0       167    ₹-5,262       -4.8%       +2.5%     
  BAJAJFINSV  19     FIN SVC    03-Aug-15   189.4       175.2       581    ₹-8,229       -7.5%       -1.1%     
  HINDPETRO   31     OIL&GAS    03-Aug-15   81.6        71.8        1350   ₹-13,162      -11.9%      -1.0%     

  AFTER: Invested ₹2,055,872 | Cash ₹45,084 | Total ₹2,100,956 | Positions 18/20 | Slot ₹105,091

========================================================================
  REBALANCE #06  —  01 Dec 2015
  NAV: ₹2,083,700  |  Slot: ₹104,185  |  Cash: ₹45,084
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  AJANTPHARM  83     HEALTH     01-Apr-15   737.1       814.9       144    ₹11,207       +10.6%    244d  
  J&KBANK     63     PVT BNK    01-Oct-15   149.2       149.2       704    ₹0            +0.0%     61d   
  REPCOHOME   52     FIN SVC    01-Oct-15   664.2       647.2       158    ₹-2,673       -2.5%     61d   
  CUMMINSIND  118    INFRA      01-Oct-15   904.6       853.5       116    ₹-5,928       -5.6%     61d   
  INFY        103    IT         01-Oct-15   434.2       403.1       242    ₹-7,536       -7.2%     61d   
  AMARAJABAT  124    AUTO       01-Oct-15   931.1       787.6       112    ₹-16,069      -15.4%    61d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  KAJARIACER  2      CON DUR    3.442    0.64   +57.9%    +44.8%    438.3       237    ₹103,868      +5.6%     
  PETRONET    3      OIL&GAS    2.759    0.35   +27.4%    +35.3%    81.2        1282   ₹104,132      +10.0%    
  COFORGE     7      IT         2.362    0.51   +56.3%    +33.6%    101.0       1031   ₹104,136      +1.3%     
  RELCAPITAL  8      FIN SVC    2.333    1.16   -10.5%    +59.3%    359.8       289    ₹103,977      +4.8%     
  NATCOPHARM  9      HEALTH     2.279    1.05   +95.4%    +22.7%    483.2       215    ₹103,889      +2.2%     
  TVSMOTOR    10     AUTO       2.242    0.51   +31.8%    +35.9%    285.6       364    ₹103,955      +5.2%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 Model is not converging.  Current: 1842.7413202030812 is not greater than 1842.8656003284593. Delta is -0.12428012537816358
Model is not converging.  Current: 1841.796324165001 is not greater than 1841.8530239077281. Delta is -0.05669974272723266
Model is not converging.  Current: 1842.7610451780545 is not greater than 1842.8726047392245. Delta is -0.11155956116999732
Model is not converging.  Current: 1959.229904776181 is not greater than 1959.2811286589897. Delta is -0.05122388280869927
Model is not converging.  Current: 1933.89926783318 is not greater than 1933.9389530508154. Delta is -0.03968521763545141
Model is not converging.  Current: 1959.1726363321552 is not greater than 1959.2359155862007. Delta is -0.06327925404548296

  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ASHOKLEY    17     AUTO       02-Feb-15   26.9        38.3        3716   ₹42,388       +42.4%      +0.3%     
  BRITANNIA   14     FMCG       01-Apr-15   946.0       1,298.0     112    ₹39,430       +37.2%      -0.8%     
  BAJFINANCE  7      FIN SVC    01-Jun-15   42.0        53.9        2479   ₹29,548       +28.4%      +3.7%     
  RAJESHEXPO  1      CON DUR    03-Aug-15   561.3       698.5       196    ₹26,888       +24.4%      +4.8%     
  ABBOTINDIA  8      HEALTH     03-Aug-15   4,000.7     4,604.4     27     ₹16,302       +15.1%      -1.7%     
  BAJAJFINSV  5      FIN SVC    03-Aug-15   189.4       198.0       581    ₹4,974        +4.5%       +2.7%     
  MARUTI      16     AUTO       03-Aug-15   4,018.0     4,159.9     27     ₹3,834        +3.5%       -0.5%     
  DIVISLAB    35     HEALTH     01-Oct-15   1,041.5     1,050.6     100    ₹916          +0.9%       -0.1%     
  HONAUT      39     MFG        01-Oct-15   9,045.6     8,830.7     11     ₹-2,364       -2.4%       -2.0%     
  IBULHSGFIN  25     FIN SVC    03-Aug-15   498.5       480.1       221    ₹-4,081       -3.7%       +4.9%     
  HINDPETRO   22     OIL&GAS    03-Aug-15   81.6        78.4        1350   ₹-4,318       -3.9%       +6.0%     
  APLLTD      27     HEALTH     03-Aug-15   657.5       629.3       167    ₹-4,711       -4.3%       +3.8%     

  AFTER: Invested ₹2,053,140 | Cash ₹29,819 | Total ₹2,082,959 | Positions 18/20 | Slot ₹104,185

========================================================================
  REBALANCE #07  —  01 Feb 2016
  NAV: ₹2,022,841  |  Slot: ₹101,142  |  Cash: ₹29,819
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BRITANNIA   36     FMCG       01-Apr-15   946.0       1,223.3     112    ₹31,066       +29.3%    306d  
  ABBOTINDIA  40     HEALTH     03-Aug-15   4,000.7     4,529.7     27     ₹14,283       +13.2%    182d  
  HONAUT      55     MFG        01-Oct-15   9,045.6     8,443.1     11     ₹-6,627       -6.7%     123d  
  MARUTI      97     AUTO       03-Aug-15   4,018.0     3,604.9     27     ₹-11,153      -10.3%    182d  
  APLLTD      38     HEALTH     03-Aug-15   657.5       560.0       167    ₹-16,286      -14.8%    182d  
  RELCAPITAL  101    FIN SVC    01-Dec-15   359.8       306.6       289    ₹-15,363      -14.8%    62d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  VAKRANGEE   4      IT         2.974    0.93   +49.8%    +44.8%    84.9        1190   ₹101,059      +8.9%     
  BERGEPAINT  5      CON DUR    2.699    0.19   +18.6%    +22.6%    152.0       665    ₹101,062      +4.8%     
  MARICO      6      FMCG       2.585    0.29   +29.2%    +15.3%    194.1       521    ₹101,107      +2.2%     
  NHPC        7      ENERGY     2.559    0.25   +13.0%    +16.2%    12.2        8319   ₹101,136      +3.9%     
  IGL         8      OIL&GAS    2.471    0.23   +19.5%    +14.6%    48.8        2072   ₹101,102      +1.5%     
  HAVELLS     9      CON DUR    2.444    0.12   +12.9%    +18.9%    279.4       362    ₹101,128      +2.9%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ASHOKLEY    18     AUTO       02-Feb-15   26.9        37.5        3716   ₹39,492       +39.5%      +3.6%     
  BAJFINANCE  3      FIN SVC    01-Jun-15   42.0        57.8        2479   ₹39,072       +37.5%      +2.0%     
  RAJESHEXPO  1      CON DUR    03-Aug-15   561.3       720.9       196    ₹31,278       +28.4%      +2.5%     
  PETRONET    2      OIL&GAS    01-Dec-15   81.2        87.1        1282   ₹7,513        +7.2%       +2.3%     
  KAJARIACER  11     CON DUR    01-Dec-15   438.3       448.1       237    ₹2,320        +2.2%       +1.7%     
  DIVISLAB    12     HEALTH     01-Oct-15   1,041.5     1,056.3     100    ₹1,480        +1.4%       +2.6%     
  NATCOPHARM  10     HEALTH     01-Dec-15   483.2       480.6       215    ₹-563         -0.5%       -3.3%     
  BAJAJFINSV  24     FIN SVC    03-Aug-15   189.4       185.3       581    ₹-2,397       -2.2%       -1.9%     
  TVSMOTOR    35     AUTO       01-Dec-15   285.6       277.4       364    ₹-2,974       -2.9%       +3.2%     
  IBULHSGFIN  26     FIN SVC    03-Aug-15   498.5       480.1       221    ₹-4,082       -3.7%       +1.6%     
  HINDPETRO   13     OIL&GAS    03-Aug-15   81.6        74.6        1350   ₹-9,402       -8.5%       -2.4%     
  COFORGE     31     IT         01-Dec-15   101.0       92.3        1031   ₹-8,971       -8.6%       +0.3%     

  AFTER: Invested ₹1,967,957 | Cash ₹54,163 | Total ₹2,022,120 | Positions 18/20 | Slot ₹101,142

========================================================================
  REBALANCE #08  —  01 Apr 2016
  NAV: ₹2,033,298  |  Slot: ₹101,665  |  Cash: ₹54,163
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  KAJARIACER  —      OTHER      01-Dec-15   438.3       440.5       237    ₹536          +0.5%     122d  
  BAJAJFINSV  52     FIN SVC    03-Aug-15   189.4       173.9       581    ₹-9,038       -8.2%     242d  
  BERGEPAINT  68     CON DUR    01-Feb-16   152.0       138.1       665    ₹-9,204       -9.1%     60d   
  DIVISLAB    99     HEALTH     01-Oct-15   1,041.5     942.5       100    ₹-9,893       -9.5%     183d  
  IBULHSGFIN  65     FIN SVC    03-Aug-15   498.5       443.6       221    ₹-12,136      -11.0%    242d  
  COFORGE     51     IT         01-Dec-15   101.0       86.1        1031   ₹-15,400      -14.8%    122d  
  NATCOPHARM  142    HEALTH     01-Dec-15   483.2       384.9       215    ₹-21,126      -20.3%    122d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  WELSPUNIND  1      CONSUMP    3.815    0.42   +208.9%   +7.7%     91.2        1114   ₹101,595      +3.5%     
  GLAXO       6      HEALTH     2.996    0.31   +19.3%    +14.6%    1,501.2     67     ₹100,579      +8.6%     
  TORNTPOWER  7      ENERGY     2.907    0.36   +45.9%    +34.5%    182.9       555    ₹101,520      +0.6%     Model is not converging.  Current: 2092.393463958577 is not greater than 2092.4704737130724. Delta is -0.07700975449552061
Model is not converging.  Current: 2066.9047890305146 is not greater than 2066.9432016858354. Delta is -0.03841265532082616
Model is not converging.  Current: 2092.3743167924417 is not greater than 2092.411842349055. Delta is -0.037525556613218214
Model is not converging.  Current: 2240.22277384058 is not greater than 2240.2276045733497. Delta is -0.004830732769733004
Model is not converging.  Current: 2239.4651824035063 is not greater than 2239.5374863542866. Delta is -0.07230395078022411
Model is not converging.  Current: 2240.1130814754524 is not greater than 2240.2035051759494. Delta is -0.09042370049701276

  JSWSTEEL    8      METAL      2.858    0.32   +41.7%    +22.9%    113.4       896    ₹101,594      +4.5%     
  JUBILANT    9      CONSUMP    2.740    0.87   +163.3%   -0.7%     281.1       361    ₹101,487      +5.6%     
  HINDZINC    10     METAL      2.692    0.35   +20.6%    +26.2%    72.0        1411   ₹101,613      +7.5%     
  RAMCOCEM    11     MFG        2.369    0.19   +40.6%    +6.9%     387.3       262    ₹101,486      +1.7%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ASHOKLEY    5      AUTO       02-Feb-15   26.9        45.2        3716   ₹68,074       +68.1%      +9.3%     
  BAJFINANCE  3      FIN SVC    01-Jun-15   42.0        66.3        2479   ₹60,381       +58.0%      +4.6%     
  NHPC        2      ENERGY     01-Feb-16   12.2        14.7        8319   ₹21,132       +20.9%      +9.5%     
  RAJESHEXPO  4      CON DUR    03-Aug-15   561.3       615.3       196    ₹10,585       +9.6%       -1.8%     
  MARICO      16     FMCG       01-Feb-16   194.1       210.1       521    ₹8,340        +8.2%       +0.4%     
  TVSMOTOR    17     AUTO       01-Dec-15   285.6       307.4       364    ₹7,951        +7.6%       +7.5%     
  VAKRANGEE   12     IT         01-Feb-16   84.9        88.6        1190   ₹4,342        +4.3%       -2.9%     
  HAVELLS     36     CON DUR    01-Feb-16   279.4       290.8       362    ₹4,135        +4.1%       +4.8%     
  PETRONET    20     OIL&GAS    01-Dec-15   81.2        84.1        1282   ₹3,735        +3.6%       -0.5%     
  IGL         15     OIL&GAS    01-Feb-16   48.8        48.8        2072   ₹-54          -0.1%       +4.3%     
  HINDPETRO   34     OIL&GAS    03-Aug-15   81.6        75.4        1350   ₹-8,415       -7.6%       +5.7%     

  AFTER: Invested ₹2,027,944 | Cash ₹4,511 | Total ₹2,032,455 | Positions 18/20 | Slot ₹101,665

========================================================================
  REBALANCE #09  —  01 Jun 2016
  NAV: ₹2,057,279  |  Slot: ₹102,864  |  Cash: ₹4,511
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  NHPC        —      ENERGY     01-Feb-16   12.2        14.7        8319   ₹21,132       +20.9%    121d  
  MARICO      57     FMCG       01-Feb-16   194.1       218.4       521    ₹12,659       +12.5%    121d  
  JSWSTEEL    —      METAL      01-Apr-16   113.4       123.5       896    ₹9,024        +8.9%     61d   
  HINDZINC    46     METAL      01-Apr-16   72.0        75.5        1411   ₹4,921        +4.8%     61d   
  TVSMOTOR    50     AUTO       01-Dec-15   285.6       273.7       364    ₹-4,342       -4.2%     183d  
  VAKRANGEE   64     IT         01-Feb-16   84.9        81.4        1190   ₹-4,210       -4.2%     121d  
  GLAXO       —      HEALTH     01-Apr-16   1,501.2     1,391.4     67     ₹-7,358       -7.3%     61d   
  JUBILANT    —      CONSUMP    01-Apr-16   281.1       248.8       361    ₹-11,665      -11.5%    61d   
  TORNTPOWER  94     ENERGY     01-Apr-16   182.9       143.8       555    ₹-21,697      -21.4%    61d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BIOCON      1      HEALTH     3.831    0.19   +60.9%    +57.9%    117.0       878    ₹102,765      +10.2%    
  VGUARD      2      CON DUR    3.607    0.26   +46.9%    +66.5%    93.1        1104   ₹102,803      +13.7%    
  INDUSINDBK  5      PVT BNK    2.645    0.41   +29.8%    +36.6%    1,042.1     98     ₹102,122      +2.9%     
  HDFC        7      PVT BNK    2.391    0.28   +15.7%    +24.7%    265.8       387    ₹102,850      +1.9%     
  HDFCBANK    8      PVT BNK    2.391    0.28   +15.7%    +24.7%    265.8       387    ₹102,850      +1.9%     
  SHRIRAMFIN  9      FIN SVC    2.237    0.68   +41.9%    +46.0%    197.6       520    ₹102,760      +3.9%     
  MUTHOOTFIN  10     FIN SVC    2.142    0.24   +35.5%    +40.6%    205.5       500    ₹102,735      +12.8%    
  PIDILITIND  12     MFG        2.056    0.41   +31.7%    +20.3%    342.9       300    ₹102,860      +9.8%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  4      FIN SVC    01-Jun-15   42.0        73.7        2479   ₹78,674       +75.6%      +1.7%     
  ASHOKLEY    15     AUTO       02-Feb-15   26.9        42.8        3716   ₹59,233       +59.2%      +0.5%     
  RAMCOCEM    3      MFG        01-Apr-16   387.3       473.9       262    ₹22,665       +22.3%      +0.7%     
  HAVELLS     13     CON DUR    01-Feb-16   279.4       331.1       362    ₹18,719       +18.5%      -0.7%     
  PETRONET    18     OIL&GAS    01-Dec-15   81.2        92.8        1282   ₹14,851       +14.3%      +0.7%     
  WELSPUNIND  11     CONSUMP    01-Apr-16   91.2        97.9        1114   ₹7,415        +7.3%       +6.1%     
  HINDPETRO   6      OIL&GAS    03-Aug-15   81.6        86.0        1350   ₹5,932        +5.4%       +5.4%     
  IGL         23     OIL&GAS    01-Feb-16   48.8        48.6        2072   ₹-385         -0.4%       -1.2%     
  RAJESHEXPO  19     CON DUR    03-Aug-15   561.3       558.5       196    ₹-538         -0.5%       -0.4%     

  AFTER: Invested ₹1,961,998 | Cash ₹94,306 | Total ₹2,056,304 | Positions 17/20 | Slot ₹102,864

========================================================================
  REBALANCE #10  —  01 Aug 2016
  NAV: ₹2,341,171  |  Slot: ₹117,059  |  Cash: ₹94,306
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ASHOKLEY    94     AUTO       02-Feb-15   26.9        38.0        3716   ₹41,076       +41.1%    546d  
  WELSPUNIND  43     CONSUMP    01-Apr-16   91.2        98.3        1114   ₹7,930        +7.8%     122d  
  HDFCBANK    27     PVT BNK    01-Jun-16   265.8       283.3       387    ₹6,769        +6.6%     61d   
  RAJESHEXPO  108    CON DUR    03-Aug-15   561.3       436.5       196    ₹-24,454      -22.2%    364d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────Model is not converging.  Current: 2381.190106853356 is not greater than 2381.2348164251885. Delta is -0.044709571832299844
Model is not converging.  Current: 2354.178464684504 is not greater than 2354.216495934294. Delta is -0.03803124979003769
Model is not converging.  Current: 2381.054464528568 is not greater than 2381.1239472873717. Delta is -0.06948275880358779
Model is not converging.  Current: 2502.7756623746645 is not greater than 2502.8310200823307. Delta is -0.055357707666189526
Model is not converging.  Current: 2502.347854190436 is not greater than 2502.364089981018. Delta is -0.01623579058195901
Model is not converging.  Current: 2502.1974515932484 is not greater than 2502.266800077959. Delta is -0.0693484847106447

  BAJAJFINSV  4      FIN SVC    3.084    0.50   +68.9%    +45.2%    281.5       415    ₹116,843      +12.3%    
  ASIANPAINT  7      CONSUMP    2.368    0.39   +34.0%    +29.2%    1,030.0     113    ₹116,390      +6.5%     
  SHREECEM    10     INFRA      2.284    0.74   +51.9%    +31.1%    15,652.0    7      ₹109,564      +3.3%     
  BERGEPAINT  11     CON DUR    2.124    0.44   +53.4%    +24.9%    186.1       629    ₹117,056      +3.1%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  1      FIN SVC    01-Jun-15   42.0        108.4       2479   ₹164,601      +158.1%     +22.0%    
  HINDPETRO   6      OIL&GAS    03-Aug-15   81.6        121.9       1350   ₹54,469       +49.4%      +12.3%    
  MUTHOOTFIN  3      FIN SVC    01-Jun-16   205.5       298.8       500    ₹46,681       +45.4%      +20.6%    
  RAMCOCEM    8      MFG        01-Apr-16   387.3       544.0       262    ₹41,036       +40.4%      +0.2%     
  HAVELLS     18     CON DUR    01-Feb-16   279.4       378.3       362    ₹35,813       +35.4%      +8.5%     
  PETRONET    13     OIL&GAS    01-Dec-15   81.2        106.3       1282   ₹32,170       +30.9%      +5.9%     
  VGUARD      2      CON DUR    01-Jun-16   93.1        113.9       1104   ₹22,892       +22.3%      +11.9%    
  BIOCON      5      HEALTH     01-Jun-16   117.0       135.3       878    ₹16,043       +15.6%      +7.7%     
  IGL         20     OIL&GAS    01-Feb-16   48.8        56.2        2072   ₹15,383       +15.2%      +4.5%     
  SHRIRAMFIN  19     FIN SVC    01-Jun-16   197.6       221.8       520    ₹12,580       +12.2%      +5.8%     
  INDUSINDBK  21     PVT BNK    01-Jun-16   1,042.1     1,138.1     98     ₹9,416        +9.2%       +5.1%     
  HDFC        26     PVT BNK    01-Jun-16   265.8       283.3       387    ₹6,769        +6.6%       +1.9%     
  PIDILITIND  9      MFG        01-Jun-16   342.9       350.4       300    ₹2,257        +2.2%       +1.2%     

  AFTER: Invested ₹2,260,954 | Cash ₹79,671 | Total ₹2,340,625 | Positions 17/20 | Slot ₹117,059

========================================================================
  REBALANCE #11  —  03 Oct 2016
  NAV: ₹2,484,216  |  Slot: ₹124,211  |  Cash: ₹79,671
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  INDUSINDBK  41     PVT BNK    01-Jun-16   1,042.1     1,168.8     98     ₹12,422       +12.2%    124d  
  SHRIRAMFIN  69     FIN SVC    01-Jun-16   197.6       208.2       520    ₹5,528        +5.4%     124d  
  PIDILITIND  61     MFG        01-Jun-16   342.9       346.3       300    ₹1,020        +1.0%     124d  

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HINDZINC    2      METAL      2.887    0.36   +120.3%   +40.9%    113.8       1091   ₹124,133      +11.7%    
  MRF         5      MFG        2.569    0.19   +25.7%    +57.3%    51,226.4    2      ₹102,453      +16.3%    
  IOC         6      OIL&GAS    2.485    0.08   +60.9%    +38.9%    53.5        2320   ₹124,170      +4.8%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  3      FIN SVC    01-Jun-15   42.0        104.9       2479   ₹156,002      +149.9%     +0.1%     
  RAMCOCEM    14     MFG        01-Apr-16   387.3       596.8       262    ₹54,885       +54.1%      +4.3%     
  HINDPETRO   15     OIL&GAS    03-Aug-15   81.6        124.8       1350   ₹58,299       +52.9%      +4.3%     
  PETRONET    10     OIL&GAS    01-Dec-15   81.2        121.2       1282   ₹51,189       +49.2%      +3.5%     
  HAVELLS     21     CON DUR    01-Feb-16   279.4       403.7       362    ₹44,996       +44.5%      +4.1%     
  MUTHOOTFIN  16     FIN SVC    01-Jun-16   205.5       295.2       500    ₹44,852       +43.7%      -1.6%     
  IGL         12     OIL&GAS    01-Feb-16   48.8        69.0        2072   ₹41,811       +41.4%      +3.8%     
  VGUARD      4      CON DUR    01-Jun-16   93.1        127.4       1104   ₹37,804       +36.8%      +0.4%     
  BIOCON      1      HEALTH     01-Jun-16   117.0       157.9       878    ₹35,882       +34.9%      +3.1%     
  BAJAJFINSV  7      FIN SVC    01-Aug-16   281.5       318.0       415    ₹15,114       +12.9%      +3.5%     
  BERGEPAINT  13     CON DUR    01-Aug-16   186.1       209.6       629    ₹14,766       +12.6%      +1.5%     
  HDFC        37     PVT BNK    01-Jun-16   265.8       293.5       387    ₹10,731       +10.4%      +0.2%     
  SHREECEM    32     INFRA      01-Aug-16   15,652.0    17,211.1    7      ₹10,914       +10.0%      +5.0%     
  ASIANPAINT  24     CONSUMP    01-Aug-16   1,030.0     1,096.3     113    ₹7,487        +6.4%       +1.9%     

  AFTER: Invested ₹2,428,589 | Cash ₹55,211 | Total ₹2,483,799 | Positions 17/20 | Slot ₹124,211

========================================================================
  REBALANCE #12  —  01 Dec 2016
  NAV: ₹2,318,525  |  Slot: ₹115,926  |  Cash: ₹55,211
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HAVELLS     83     CON DUR    01-Feb-16   279.4       313.5       362    ₹12,345       +12.2%    304d  
  HDFC        50     PVT BNK    01-Jun-16   265.8       273.0       387    ₹2,794        +2.7%     183d  
  SHREECEM    44     INFRA      01-Aug-16   15,652.0    14,419.1    7      ₹-8,630       -7.9%     122d  
  ASIANPAINT  84     CONSUMP    01-Aug-16   1,030.0     865.1       113    ₹-18,630      -16.0%    122d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  VAKRANGEE   2      IT         3.620    0.06   +83.6%    +42.4%    123.4       939    ₹115,896      +4.8%     
  SYNGENE     3      HEALTH     3.076    0.20   +62.3%    +30.9%    297.8       389    ₹115,838      +9.6%     
  TVSMOTOR    9      AUTO       2.254    0.46   +26.8%    +21.5%    349.2       331    ₹115,588      +0.3%     
  AIAENG      11     MFG        2.188    0.49   +44.9%    +12.2%    1,251.1     92     ₹115,104      +4.0%     Model is not converging.  Current: 2648.7479450666146 is not greater than 2648.7946421619504. Delta is -0.04669709533573041
Model is not converging.  Current: 2621.369644384961 is not greater than 2621.391171660358. Delta is -0.021527275397147605
Model is not converging.  Current: 2648.180812777552 is not greater than 2648.1838145366864. Delta is -0.003001759134349413
Model is not converging.  Current: 2797.9713689800974 is not greater than 2798.0014806170516. Delta is -0.030111636954188725
Model is not converging.  Current: 2768.454625104869 is not greater than 2768.47440994231. Delta is -0.01978483744096593
Model is not converging.  Current: 2797.219891107874 is not greater than 2797.2279415983867. Delta is -0.008050490512687247


  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  30     FIN SVC    01-Jun-15   42.0        88.3        2479   ₹114,715      +110.2%     -0.3%     
  PETRONET    7      OIL&GAS    01-Dec-15   81.2        130.9       1282   ₹63,706       +61.2%      +2.5%     
  HINDPETRO   8      OIL&GAS    03-Aug-15   81.6        129.0       1350   ₹63,946       +58.1%      -1.9%     
  IGL         6      OIL&GAS    01-Feb-16   48.8        73.2        2072   ₹50,549       +50.0%      +0.2%     
  RAMCOCEM    10     MFG        01-Apr-16   387.3       581.0       262    ₹50,730       +50.0%      +2.6%     
  BIOCON      4      HEALTH     01-Jun-16   117.0       151.6       878    ₹30,302       +29.5%      +3.4%     
  VGUARD      12     CON DUR    01-Jun-16   93.1        116.3       1104   ₹25,567       +24.9%      -6.0%     
  MUTHOOTFIN  24     FIN SVC    01-Jun-16   205.5       253.8       500    ₹24,170       +23.5%      -3.8%     
  HINDZINC    1      METAL      03-Oct-16   113.8       123.8       1091   ₹10,986       +8.9%       +4.4%     
  BAJAJFINSV  18     FIN SVC    01-Aug-16   281.5       292.6       415    ₹4,566        +3.9%       -2.1%     
  BERGEPAINT  32     CON DUR    01-Aug-16   186.1       186.0       629    ₹-75          -0.1%       +4.3%     
  IOC         15     OIL&GAS    03-Oct-16   53.5        52.8        2320   ₹-1,645       -1.3%       -1.5%     
  MRF         5      MFG        03-Oct-16   51,226.4    48,256.4    2      ₹-5,940       -5.8%       -0.3%     

  AFTER: Invested ₹2,307,929 | Cash ₹10,047 | Total ₹2,317,976 | Positions 17/20 | Slot ₹115,926

========================================================================
  REBALANCE #13  —  01 Feb 2017
  NAV: ₹2,566,063  |  Slot: ₹128,303  |  Cash: ₹10,047
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MUTHOOTFIN  35     FIN SVC    01-Jun-16   205.5       272.1       500    ₹33,334       +32.4%    245d  
  TVSMOTOR    40     AUTO       01-Dec-16   349.2       374.6       331    ₹8,414        +7.3%     62d   
  BERGEPAINT  97     CON DUR    01-Aug-16   186.1       169.1       629    ₹-10,719      -9.2%     184d  
  SYNGENE     54     HEALTH     01-Dec-16   297.8       265.9       389    ₹-12,413      -10.7%    62d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  POWERGRID   5      ENERGY     2.888    0.34   +56.8%    +17.5%    73.8        1739   ₹128,278      +3.7%     
  BEL         9      DEFENCE    2.355    0.63   +33.2%    +20.5%    40.0        3208   ₹128,295      +4.1%     
  RECLTD      10     FIN SVC    2.354    0.84   +77.4%    +15.8%    54.0        2377   ₹128,272      +6.0%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  21     FIN SVC    01-Jun-15   42.0        102.9       2479   ₹151,084      +145.1%     +12.1%    
  HINDPETRO   2      OIL&GAS    03-Aug-15   81.6        155.5       1350   ₹99,721       +90.5%      +7.6%     
  RAMCOCEM    7      MFG        01-Apr-16   387.3       693.0       262    ₹80,091       +78.9%      +13.1%    
  IGL         8      OIL&GAS    01-Feb-16   48.8        84.2        2072   ₹73,297       +72.5%      +3.1%     
  PETRONET    23     OIL&GAS    01-Dec-15   81.2        132.7       1282   ₹66,015       +63.4%      +3.6%     
  VGUARD      11     CON DUR    01-Jun-16   93.1        141.1       1104   ₹53,004       +51.6%      +12.8%    
  BIOCON      6      HEALTH     01-Jun-16   117.0       166.6       878    ₹43,526       +42.4%      +2.4%     
  IOC         3      OIL&GAS    03-Oct-16   53.5        66.6        2320   ₹30,276       +24.4%      +5.1%     
  HINDZINC    1      METAL      03-Oct-16   113.8       138.1       1091   ₹26,562       +21.4%      +6.3%     
  BAJAJFINSV  24     FIN SVC    01-Aug-16   281.5       331.3       415    ₹20,638       +17.7%      +7.6%     
  VAKRANGEE   4      IT         01-Dec-16   123.4       137.7       939    ₹13,365       +11.5%      +4.5%     
  AIAENG      17     MFG        01-Dec-16   1,251.1     1,274.0     92     ₹2,101        +1.8%       -0.2%     
  MRF         25     MFG        03-Oct-16   51,226.4    51,910.7    2      ₹1,369        +1.3%       +0.7%     

  AFTER: Invested ₹2,471,028 | Cash ₹94,578 | Total ₹2,565,606 | Positions 16/20 | Slot ₹128,303

========================================================================
  REBALANCE #14  —  03 Apr 2017
  NAV: ₹2,782,608  |  Slot: ₹139,130  |  Cash: ₹94,578
========================================================================
  [SECTOR CAP≤4] dropped: DHFL

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJFINANCE  30     FIN SVC    01-Jun-15   42.0        114.1       2479   ₹178,752      +171.7%   672d  
  PETRONET    44     OIL&GAS    01-Dec-15   81.2        142.1       1282   ₹77,982       +74.9%    489d  
  RAMCOCEM    39     MFG        01-Apr-16   387.3       654.3       262    ₹69,940       +68.9%    367d  
  AIAENG      34     MFG        01-Dec-16   1,251.1     1,509.8     92     ₹23,794       +20.7%    123d  
  BEL         45     DEFENCE    01-Feb-17   40.0        41.1        3208   ₹3,503        +2.7%     61d   
  POWERGRID   46     ENERGY     01-Feb-17   73.8        70.7        1739   ₹-5,374       -4.2%     61d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  EDELWEISS   4      FIN SVC    2.685    1.18   +176.5%   +67.6%    73.6        1889   ₹139,058      +12.3%    
  IBULHSGFIN  5      FIN SVC    2.523    0.30   +63.1%    +54.9%    707.9       196    ₹138,742      +6.5%     
  DALMIABHA   6      MFG        2.502    0.29   +147.5%   +45.8%    1,969.1     70     ₹137,835      +2.4%     
  BBTC        7      FMCG       2.372    1.19   +122.7%   +65.1%    828.5       167    ₹138,363      +6.8%     
  SUNTV       8      MEDIA      2.335    1.02   +117.6%   +63.7%    626.4       222    ₹139,067      +5.2%     
  GUJGASLTD   10     OIL&GAS    2.305    0.42   +44.0%    +44.8%    142.3       977    ₹139,072      +4.2%     Model is not converging.  Current: 2942.3766319960077 is not greater than 2942.43512994896. Delta is -0.05849795295216609
Model is not converging.  Current: 2942.1342583187097 is not greater than 2942.173640492679. Delta is -0.0393821739694431
Model is not converging.  Current: 2941.5970431442797 is not greater than 2941.6180450603233. Delta is -0.0210019160435877
Model is not converging.  Current: 3099.0307967664894 is not greater than 3099.038099538482. Delta is -0.007302771992726775
Model is not converging.  Current: 3098.1978276687146 is not greater than 3098.2421838434107. Delta is -0.04435617469607678

  NATCOPHARM  12     HEALTH     2.250    0.28   +85.9%    +50.4%    803.3       173    ₹138,966      +8.1%     
  KARURVYSYA  13     PVT BNK    2.245    0.36   +36.7%    +42.4%    73.3        1897   ₹139,111      +10.0%    

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDPETRO   15     OIL&GAS    03-Aug-15   81.6        158.1       1350   ₹103,272      +93.8%      +0.7%     
  IGL         28     OIL&GAS    01-Feb-16   48.8        88.9        2072   ₹83,033       +82.1%      -0.4%     
  VGUARD      1      CON DUR    01-Jun-16   93.1        168.4       1104   ₹83,140       +80.9%      +2.5%     
  BIOCON      9      HEALTH     01-Jun-16   117.0       185.3       878    ₹59,921       +58.3%      +0.9%     
  BAJAJFINSV  3      FIN SVC    01-Aug-16   281.5       407.9       415    ₹52,437       +44.9%      +2.7%     
  IOC         19     OIL&GAS    03-Oct-16   53.5        70.8        2320   ₹40,120       +32.3%      +2.1%     
  RECLTD      2      FIN SVC    01-Feb-17   54.0        68.6        2377   ₹34,768       +27.1%      +6.9%     
  HINDZINC    21     METAL      03-Oct-16   113.8       143.4       1091   ₹32,342       +26.1%      +2.0%     
  VAKRANGEE   27     IT         01-Dec-16   123.4       147.3       939    ₹22,444       +19.4%      +2.1%     
  MRF         32     MFG        03-Oct-16   51,226.4    60,214.9    2      ₹17,977       +17.5%      +6.5%     

  AFTER: Invested ₹2,768,260 | Cash ₹13,030 | Total ₹2,781,289 | Positions 18/20 | Slot ₹139,130

========================================================================
  REBALANCE #15  —  01 Jun 2017
  NAV: ₹2,912,553  |  Slot: ₹145,628  |  Cash: ₹13,030
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BIOCON      102    HEALTH     01-Jun-16   117.0       156.7       878    ₹34,847       +33.9%    365d  
  RECLTD      3      FIN SVC    01-Feb-17   54.0        72.3        2377   ₹43,474       +33.9%    120d  
  MRF         —      MFG        03-Oct-16   51,226.4    67,346.4    2      ₹32,240       +31.5%    241d  
  EDELWEISS   20     FIN SVC    03-Apr-17   73.6        80.6        1889   ₹13,258       +9.5%     59d   
  BBTC        12     FMCG       03-Apr-17   828.5       871.8       167    ₹7,228        +5.2%     59d   
  HINDZINC    97     METAL      03-Oct-16   113.8       117.4       1091   ₹3,955        +3.2%     241d  
  SUNTV       36     MEDIA      03-Apr-17   626.4       633.1       222    ₹1,478        +1.1%     59d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIENSOL  1      ENERGY     4.624    1.04   +324.2%   +105.6%   127.1       1146   ₹145,599      +22.7%    
  HDFC        4      PVT BNK    2.393    0.56   +39.9%    +17.2%    371.3       392    ₹145,562      +2.9%     
  HDFCBANK    5      PVT BNK    2.393    0.56   +39.9%    +17.2%    371.3       392    ₹145,562      +2.9%     
  TVSMOTOR    6      AUTO       2.315    0.80   +81.3%    +24.9%    508.7       286    ₹145,482      +2.1%     
  MARUTI      7      AUTO       2.289    1.00   +77.2%    +20.7%    6,570.4     22     ₹144,549      +3.7%     
  HINDUNILVR  8      FMCG       2.255    0.50   +32.9%    +26.5%    940.4       154    ₹144,829      +7.3%     
  DHFL        9      FIN SVC    2.186    0.63   +114.3%   +28.8%    408.8       356    ₹145,527      +0.4%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDPETRO   35     OIL&GAS    03-Aug-15   81.6        165.0       1350   ₹112,573      +102.2%     +1.0%     
  IGL         26     OIL&GAS    01-Feb-16   48.8        93.2        2072   ₹91,971       +91.0%      +3.7%     
  VGUARD      23     CON DUR    01-Jun-16   93.1        176.8       1104   ₹92,343       +89.8%      -3.8%     
  BAJAJFINSV  21     FIN SVC    01-Aug-16   281.5       418.6       415    ₹56,871       +48.7%      -0.1%     
  IOC         16     OIL&GAS    03-Oct-16   53.5        76.9        2320   ₹54,215       +43.7%      -3.3%     
  VAKRANGEE   2      IT         01-Dec-16   123.4       174.1       939    ₹47,598       +41.1%      +9.6%     
  DALMIABHA   5      MFG        03-Apr-17   1,969.1     2,419.4     70     ₹31,520       +22.9%      +1.8%     
  IBULHSGFIN  14     FIN SVC    03-Apr-17   707.9       817.2       196    ₹21,439       +15.5%      +6.4%     
  NATCOPHARM  17     HEALTH     03-Apr-17   803.3       877.3       173    ₹12,799       +9.2%       +3.1%     
  KARURVYSYA  44     PVT BNK    03-Apr-17   73.3        74.6        1897   ₹2,449        +1.8%       +0.9%     
  GUJGASLTD   25     OIL&GAS    03-Apr-17   142.3       142.8       977    ₹459          +0.3%       -1.6%     

  AFTER: Invested ₹2,906,044 | Cash ₹5,301 | Total ₹2,911,345 | Positions 18/20 | Slot ₹145,628

========================================================================
  REBALANCE #16  —  01 Aug 2017
  NAV: ₹3,138,315  |  Slot: ₹156,916  |  Cash: ₹5,301
========================================================================
  [SECTOR CAP≤4] dropped: KOTAKBANK

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HINDPETRO   40     OIL&GAS    03-Aug-15   81.6        177.2       1350   ₹129,106      +117.2%   729d  
  VGUARD      49     CON DUR    01-Jun-16   93.1        174.9       1104   ₹90,321       +87.9%    426d  
  BAJAJFINSV  19     FIN SVC    01-Aug-16   281.5       504.6       415    ₹92,550       +79.2%    365d  
  IOC         91     OIL&GAS    03-Oct-16   53.5        68.8        2320   ₹35,465       +28.6%    302d  
  GUJGASLTD   84     OIL&GAS    03-Apr-17   142.3       142.6       977    ₹285          +0.2%     120d  
  ADANIENSOL  2      ENERGY     01-Jun-17   127.1       125.8       1146   ₹-1,432       -1.0%     61d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RAJESHEXPO  6      CON DUR    2.610    0.79   +64.8%    +16.7%    714.7       219    ₹156,514      +4.4%     
  RELIANCE    9      OIL&GAS    2.205    0.71   +58.0%    +18.0%    352.5       445    ₹156,885      +3.9%     Model is not converging.  Current: 3243.016705196974 is not greater than 3243.0215423805116. Delta is -0.004837183537802048
Model is not converging.  Current: 3242.306088246072 is not greater than 3242.320389761312. Delta is -0.014301515240276785
Model is not converging.  Current: 3396.180268260439 is not greater than 3396.2048932464404. Delta is -0.02462498600152685
Model is not converging.  Current: 3396.148929068179 is not greater than 3396.184150708574. Delta is -0.03522164039532072
Model is not converging.  Current: 3395.3382714480144 is not greater than 3395.3921009021515. Delta is -0.05382945413703055

  INDUSINDBK  10     PVT BNK    2.022    1.14   +44.6%    +16.2%    1,585.1     98     ₹155,343      +5.7%     
  PGHH        11     FMCG       1.963    0.36   +33.7%    +15.5%    7,228.0     21     ₹151,788      +1.0%     
  GODREJIND   13     CONSUMP    1.848    0.95   +49.3%    +18.0%    649.5       241    ₹156,524      -1.3%     
  UPL         18     MFG        1.635    1.13   +55.5%    +10.9%    548.9       285    ₹156,449      +3.3%     
  HONAUT      19     MFG        1.633    0.51   +24.8%    +12.5%    12,237.3    12     ₹146,848      +2.6%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  IGL         10     OIL&GAS    01-Feb-16   48.8        104.2       2072   ₹114,846      +113.6%     +4.8%     
  VAKRANGEE   1      IT         01-Dec-16   123.4       199.9       939    ₹71,786       +61.9%      +1.3%     
  DALMIABHA   28     MFG        03-Apr-17   1,969.1     2,588.4     70     ₹43,351       +31.5%      -1.1%     
  IBULHSGFIN  22     FIN SVC    03-Apr-17   707.9       876.7       196    ₹33,083       +23.8%      +6.3%     
  KARURVYSYA  26     PVT BNK    03-Apr-17   73.3        90.0        1897   ₹31,679       +22.8%      +1.0%     
  DHFL        25     FIN SVC    01-Jun-17   408.8       459.5       356    ₹18,067       +12.4%      +2.8%     
  NATCOPHARM  33     HEALTH     03-Apr-17   803.3       899.3       173    ₹16,615       +12.0%      -1.8%     
  TVSMOTOR    5      AUTO       01-Jun-17   508.7       568.4       286    ₹17,076       +11.7%      +4.4%     
  HDFC        3      PVT BNK    01-Jun-17   371.3       412.5       392    ₹16,137       +11.1%      +4.2%     
  HDFCBANK    4      PVT BNK    01-Jun-17   371.3       412.5       392    ₹16,137       +11.1%      +4.2%     
  MARUTI      9      AUTO       01-Jun-17   6,570.4     7,222.7     22     ₹14,351       +9.9%       +4.1%     
  HINDUNILVR  7      FMCG       01-Jun-17   940.4       1,017.0     154    ₹11,787       +8.1%       +2.8%     

  AFTER: Invested ₹3,128,430 | Cash ₹8,603 | Total ₹3,137,033 | Positions 19/20 | Slot ₹156,916

========================================================================
  REBALANCE #17  —  03 Oct 2017
  NAV: ₹3,285,440  |  Slot: ₹164,272  |  Cash: ₹8,603
========================================================================
  [SECTOR CAP≤4] dropped: GUJGASLTD

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MARUTI      29     AUTO       01-Jun-17   6,570.4     7,324.9     22     ₹16,598       +11.5%    124d  
  NATCOPHARM  105    HEALTH     03-Apr-17   803.3       738.5       173    ₹-11,209      -8.1%     183d  
  GODREJIND   74     CONSUMP    01-Aug-17   649.5       588.7       241    ₹-14,648      -9.4%     63d   
  UPL         91     MFG        01-Aug-17   548.9       488.7       285    ₹-17,165      -11.0%    63d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  GAIL        9      OIL&GAS    2.200    0.84   +58.0%    +19.5%    76.7        2141   ₹164,241      +8.2%     
  MGL         10     OIL&GAS    2.044    0.75   +77.3%    +10.1%    877.9       187    ₹164,176      -0.5%     
  BRITANNIA   12     FMCG       1.983    0.88   +27.7%    +17.8%    1,920.1     85     ₹163,206      +0.9%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  IGL         1      OIL&GAS    01-Feb-16   48.8        129.7       2072   ₹167,609      +165.8%     +4.3%     
  VAKRANGEE   6      IT         01-Dec-16   123.4       222.4       939    ₹92,981       +80.2%      +0.6%     
  DALMIABHA   42     MFG        03-Apr-17   1,969.1     2,715.0     70     ₹52,215       +37.9%      +1.4%     
  DHFL        15     FIN SVC    01-Jun-17   408.8       543.3       356    ₹47,891       +32.9%      +0.8%     
  HONAUT      5      MFG        01-Aug-17   12,237.3    15,425.8    12     ₹38,261       +26.1%      +3.6%     
  IBULHSGFIN  25     FIN SVC    03-Apr-17   707.9       890.5       196    ₹35,798       +25.8%      -0.5%     
  KARURVYSYA  46     PVT BNK    03-Apr-17   73.3        91.7        1897   ₹34,904       +25.1%      -3.0%     
  TVSMOTOR    9      AUTO       01-Jun-17   508.7       622.8       286    ₹32,640       +22.4%      +2.2%     
  RAJESHEXPO  3      CON DUR    01-Aug-17   714.7       815.3       219    ₹22,041       +14.1%      +6.7%     
  HDFC        10     PVT BNK    01-Jun-17   371.3       415.2       392    ₹17,185       +11.8%      +0.2%     
  HDFCBANK    11     PVT BNK    01-Jun-17   371.3       415.2       392    ₹17,185       +11.8%      +0.2%     
  HINDUNILVR  27     FMCG       01-Jun-17   940.4       1,027.7     154    ₹13,441       +9.3%       -2.7%     
  PGHH        37     FMCG       01-Aug-17   7,228.0     7,500.9     21     ₹5,731        +3.8%       +0.8%     
  INDUSINDBK  22     PVT BNK    01-Aug-17   1,585.1     1,611.1     98     ₹2,541        +1.6%       -0.2%     
  RELIANCE    18     OIL&GAS    01-Aug-17   352.5       351.0       445    ₹-675         -0.4%       -1.7%     

  AFTER: Invested ₹3,198,396 | Cash ₹86,460 | Total ₹3,284,857 | Positions 18/20 | Slot ₹164,272

========================================================================
  REBALANCE #18  —  01 Dec 2017
  NAV: ₹3,515,060  |  Slot: ₹175,753  |  Cash: ₹86,460
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  IBULHSGFIN  72     FIN SVC    03-Apr-17   707.9       831.7       196    ₹24,263       +17.5%    242d  
  KARURVYSYA  96     PVT BNK    03-Apr-17   73.3        80.2        1897   ₹13,121       +9.4%     242d  
  MGL         51     OIL&GAS    03-Oct-17   877.9       888.1       187    ₹1,896        +1.2%     59d   
  INDUSINDBK  48     PVT BNK    01-Aug-17   1,585.1     1,580.8     98     ₹-429         -0.3%     122d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BBTC        2      FMCG       3.353    0.99   +197.5%   +48.2%    1,478.3     118    ₹174,444      -0.9%     
  FRETAIL     3      CONSUMP    2.538    1.11   +348.9%   +1.6%     544.8       322    ₹175,410      +2.1%     Model is not converging.  Current: 3549.871050233935 is not greater than 3549.882079049188. Delta is -0.011028815253212088
Model is not converging.  Current: 3511.7490826240755 is not greater than 3511.765585708406. Delta is -0.01650308433045211
Model is not converging.  Current: 3549.1363756543847 is not greater than 3549.1831694160896. Delta is -0.04679376170497562
Model is not converging.  Current: 3671.832035087352 is not greater than 3671.834172083805. Delta is -0.002136996452918538
Model is not converging.  Current: 3672.0287554128586 is not greater than 3672.0306702874086. Delta is -0.0019148745500388031
Model is not converging.  Current: 3671.106142028238 is not greater than 3671.1494601481795. Delta is -0.04331811994143209

  BALKRISIND  7      MFG        2.244    0.53   +130.1%   +28.2%    978.3       179    ₹175,121      +4.4%     
  MARUTI      10     AUTO       2.111    1.09   +70.9%    +10.2%    7,994.1     21     ₹167,876      +2.5%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  IGL         6      OIL&GAS    01-Feb-16   48.8        139.5       2072   ₹187,996      +185.9%     +1.1%     
  VAKRANGEE   1      IT         01-Dec-16   123.4       320.5       939    ₹185,039      +159.7%     +7.4%     
  DALMIABHA   26     MFG        03-Apr-17   1,969.1     3,120.1     70     ₹80,575       +58.5%      +2.4%     
  DHFL        10     FIN SVC    01-Jun-17   408.8       595.4       356    ₹66,434       +45.7%      -2.9%     
  TVSMOTOR    4      AUTO       01-Jun-17   508.7       692.6       286    ₹52,592       +36.2%      +0.9%     
  HONAUT      5      MFG        01-Aug-17   12,237.3    16,639.1    12     ₹52,821       +36.0%      +1.7%     
  PGHH        9      FMCG       01-Aug-17   7,228.0     8,492.5     21     ₹26,554       +17.5%      +4.5%     
  HINDUNILVR  37     FMCG       01-Jun-17   940.4       1,090.3     154    ₹23,085       +15.9%      -1.0%     
  HDFC        15     PVT BNK    01-Jun-17   371.3       424.2       392    ₹20,725       +14.2%      +0.4%     
  HDFCBANK    16     PVT BNK    01-Jun-17   371.3       424.2       392    ₹20,725       +14.2%      +0.4%     
  RELIANCE    20     OIL&GAS    01-Aug-17   352.5       400.2       445    ₹21,191       +13.5%      -1.3%     
  BRITANNIA   13     FMCG       03-Oct-17   1,920.1     2,126.2     85     ₹17,520       +10.7%      +1.0%     
  RAJESHEXPO  32     CON DUR    01-Aug-17   714.7       751.9       219    ₹8,148        +5.2%       -1.5%     
  GAIL        28     OIL&GAS    03-Oct-17   76.7        80.3        2141   ₹7,695        +4.7%       -0.8%     

  AFTER: Invested ₹3,485,228 | Cash ₹29,010 | Total ₹3,514,238 | Positions 18/20 | Slot ₹175,753

========================================================================
  REBALANCE #19  —  01 Feb 2018
  NAV: ₹3,496,404  |  Slot: ₹174,820  |  Cash: ₹29,010
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  IGL         56     OIL&GAS    01-Feb-16   48.8        133.1       2072   ₹174,702      +172.8%   731d  
  VAKRANGEE   41     IT         01-Dec-16   123.4       262.4       939    ₹130,492      +112.6%   427d  
  DALMIABHA   44     MFG        03-Apr-17   1,969.1     2,972.0     70     ₹70,205       +50.9%    304d  
  DHFL        57     FIN SVC    01-Jun-17   408.8       564.2       356    ₹55,328       +38.0%    245d  
  TVSMOTOR    40     AUTO       01-Jun-17   508.7       641.5       286    ₹37,982       +26.1%    245d  
  GAIL        54     OIL&GAS    03-Oct-17   76.7        87.1        2141   ₹22,149       +13.5%    121d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BIOCON      1      HEALTH     2.835    0.41   +82.6%    +62.8%    305.0       573    ₹174,741      +6.0%     
  PCJEWELLER  2      CON DUR    2.775    0.86   +149.8%   +39.3%    48.1        3632   ₹174,815      -8.4%     
  LT          7      INFRA      2.543    0.15   +55.4%    +20.2%    1,278.9     136    ₹173,926      +6.1%     
  MPHASIS     8      IT         2.476    -0.04  +61.5%    +26.3%    722.3       242    ₹174,795      +9.6%     
  TITAN       9      CON DUR    2.415    0.28   +124.0%   +26.8%    807.9       216    ₹174,510      -5.6%     
  IBULHSGFIN  12     FIN SVC    2.130    0.42   +87.9%    +12.5%    1,010.8     172    ₹173,854      +5.3%     
  ABBOTINDIA  14     HEALTH     2.048    0.04   +26.5%    +28.4%    5,020.8     34     ₹170,707      +1.3%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HONAUT      29     MFG        01-Aug-17   12,237.3    16,698.3    12     ₹53,532       +36.5%      -4.5%     
  HINDUNILVR  10     FMCG       01-Jun-17   940.4       1,195.6     154    ₹39,299       +27.1%      +0.3%     
  HDFC        4      PVT BNK    01-Jun-17   371.3       457.0       392    ₹33,587       +23.1%      +2.7%     
  HDFCBANK    5      PVT BNK    01-Jun-17   371.3       457.0       392    ₹33,587       +23.1%      +2.7%     
  RELIANCE    27     OIL&GAS    01-Aug-17   352.5       415.0       445    ₹27,794       +17.7%      -0.3%     
  PGHH        26     FMCG       01-Aug-17   7,228.0     8,267.8     21     ₹21,837       +14.4%      -0.6%     
  RAJESHEXPO  13     CON DUR    01-Aug-17   714.7       814.1       219    ₹21,770       +13.9%      -0.1%     
  MARUTI      6      AUTO       01-Dec-17   7,994.1     8,730.3     21     ₹15,461       +9.2%       -0.1%     
  BRITANNIA   33     FMCG       03-Oct-17   1,920.1     2,097.2     85     ₹15,059       +9.2%       +0.8%     
  BALKRISIND  3      MFG        01-Dec-17   978.3       1,047.3     179    ₹12,338       +7.0%       -1.5%     
  FRETAIL     11     CONSUMP    01-Dec-17   544.8       542.5       322    ₹-708         -0.4%       -1.3%     
  BBTC        37     FMCG       01-Dec-17   1,478.3     1,383.9     118    ₹-11,146      -6.4%       -8.5%     

  AFTER: Invested ₹3,383,800 | Cash ₹111,158 | Total ₹3,494,958 | Positions 19/20 | Slot ₹174,820

========================================================================
  REBALANCE #20  —  02 Apr 2018
  NAV: ₹3,354,681  |  Slot: ₹167,734  |  Cash: ₹111,158
========================================================================
  [SECTOR CAP≤4] dropped: GODREJCP

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HONAUT      46     MFG        01-Aug-17   12,237.3    16,897.3    12     ₹55,920       +38.1%    244d  
  RELIANCE    36     OIL&GAS    01-Aug-17   352.5       392.6       445    ₹17,835       +11.4%    244d  
  RAJESHEXPO  63     CON DUR    01-Aug-17   714.7       733.3       219    ₹4,074        +2.6%     244d  
  BALKRISIND  39     MFG        01-Dec-17   978.3       1,002.1     179    ₹4,254        +2.4%     122d  
  ABBOTINDIA  48     HEALTH     01-Feb-18   5,020.8     4,963.2     34     ₹-1,958       -1.1%     60d   
  BBTC        88     FMCG       01-Dec-17   1,478.3     1,206.8     118    ₹-32,041      -18.4%    122d  
  PCJEWELLER  89     CON DUR    01-Feb-18   48.1        31.1        3632   ₹-61,761      -35.3%    60d   Model is not converging.  Current: 3826.880133969503 is not greater than 3826.8916805195677. Delta is -0.011546550064849725
Model is not converging.  Current: 3788.07689430141 is not greater than 3788.0909913789014. Delta is -0.014097077491442178
Model is not converging.  Current: 3826.200091438823 is not greater than 3826.2372246327723. Delta is -0.03713319394910286
Model is not converging.  Current: 3981.731778219965 is not greater than 3981.7333565966437. Delta is -0.0015783766789354559
Model is not converging.  Current: 3982.017699423732 is not greater than 3982.074837470925. Delta is -0.05713804719289328
Model is not converging.  Current: 3981.056538655673 is not greater than 3981.089511770992. Delta is -0.03297311531878222


  ENTRIES (6)
  [52w filter blocked 2: STRTECH(-21.2%), ADANIENSOL(-23.8%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  DBL         1      INFRA      3.529    1.01   +219.8%   +10.8%    1,099.5     152    ₹167,128      +12.7%    
  ASHOKLEY    2      AUTO       3.217    0.44   +72.1%    +24.9%    62.3        2692   ₹167,711      +3.0%     
  CHOLAFIN    7      FIN SVC    2.479    0.43   +49.0%    +14.9%    287.4       583    ₹167,547      +2.6%     
  VBL         10     FMCG       2.353    0.51   +70.4%    +8.5%     37.0        4528   ₹167,734      +2.3%     
  INDUSINDBK  11     PVT BNK    2.348    0.31   +30.4%    +9.3%     1,717.3     97     ₹166,580      +3.7%     
  TCS         16     IT         2.140    0.23   +22.2%    +11.3%    1,178.8     142    ₹167,387      +0.4%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  14     FMCG       01-Jun-17   940.4       1,178.2     154    ₹36,620       +25.3%      +2.2%     
  HDFC        8      PVT BNK    01-Jun-17   371.3       443.3       392    ₹28,193       +19.4%      +2.9%     
  HDFCBANK    9      PVT BNK    01-Jun-17   371.3       443.3       392    ₹28,193       +19.4%      +2.9%     
  BRITANNIA   3      FMCG       03-Oct-17   1,920.1     2,259.4     85     ₹28,843       +17.7%      +4.5%     
  PGHH        18     FMCG       01-Aug-17   7,228.0     8,386.1     21     ₹24,321       +16.0%      +0.5%     
  TITAN       4      CON DUR    01-Feb-18   807.9       917.9       216    ₹23,751       +13.6%      +6.9%     
  MARUTI      30     AUTO       01-Dec-17   7,994.1     8,364.8     21     ₹7,784        +4.6%       +2.1%     
  FRETAIL     13     CONSUMP    01-Dec-17   544.8       549.3       322    ₹1,465        +0.8%       +2.6%     
  BIOCON      23     HEALTH     01-Feb-18   305.0       294.8       573    ₹-5,814       -3.3%       +0.9%     
  MPHASIS     6      IT         01-Feb-18   722.3       695.3       242    ₹-6,528       -3.7%       +0.0%     
  LT          19     INFRA      01-Feb-18   1,278.9     1,173.7     136    ₹-14,297      -8.2%       +2.5%     
  IBULHSGFIN  28     FIN SVC    01-Feb-18   1,010.8     913.5       172    ₹-16,724      -9.6%       +1.1%     

  AFTER: Invested ₹3,105,952 | Cash ₹247,537 | Total ₹3,353,489 | Positions 18/20 | Slot ₹167,734

========================================================================
  REBALANCE #21  —  01 Jun 2018
  NAV: ₹3,561,059  |  Slot: ₹178,053  |  Cash: ₹247,537
========================================================================
  [SECTOR CAP≤4] dropped: DABUR, COLPAL

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PGHH        44     FMCG       01-Aug-17   7,228.0     8,342.6     21     ₹23,408       +15.4%    304d  
  CHOLAFIN    28     FIN SVC    02-Apr-18   287.4       308.5       583    ₹12,292       +7.3%     60d   
  BIOCON      —      HEALTH     01-Feb-18   305.0       319.7       573    ₹8,439        +4.8%     120d  
  MARUTI      49     AUTO       01-Dec-17   7,994.1     8,179.7     21     ₹3,897        +2.3%     182d  
  ASHOKLEY    27     AUTO       02-Apr-18   62.3        63.5        2692   ₹3,114        +1.9%     60d   
  LT          48     INFRA      01-Feb-18   1,278.9     1,205.8     136    ₹-9,941       -5.7%     120d  
  IBULHSGFIN  62     FIN SVC    01-Feb-18   1,010.8     918.6       172    ₹-15,859      -9.1%     120d  
  DBL         42     INFRA      02-Apr-18   1,099.5     847.7       152    ₹-38,270      -22.9%    60d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    1      CONSUMP    3.392    0.45   +166.5%   +23.0%    245.3       725    ₹177,839      -1.3%     
  COFORGE     4      IT         2.740    0.72   +120.6%   +32.7%    201.6       883    ₹177,971      +3.3%     
  PIDILITIND  5      MFG        2.662    0.47   +49.2%    +24.2%    538.3       330    ₹177,635      +0.5%     
  DMART       6      FMCG       2.562    0.47   +114.2%   +13.6%    1,531.1     116    ₹177,608      +3.3%     
  TECHM       7      IT         2.562    0.03   +89.2%    +14.4%    516.6       344    ₹177,715      +2.4%     
  KOTAKBANK   10     PVT BNK    2.516    0.66   +36.2%    +20.9%    262.2       679    ₹178,053      +3.5%     
  M&M         11     AUTO       2.434    0.70   +33.8%    +23.8%    829.0       214    ₹177,413      +4.7%     
  BAJFINANCE  15     FIN SVC    2.377    0.65   +59.3%    +26.7%    201.1       885    ₹177,964      +2.1%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  3      FMCG       01-Jun-17   940.4       1,385.4     154    ₹68,517       +47.3%      +2.3%     
  BRITANNIA   2      FMCG       03-Oct-17   1,920.1     2,571.5     85     ₹55,369       +33.9%      +2.9%     
  HDFC        13     PVT BNK    01-Jun-17   371.3       487.5       392    ₹45,547       +31.3%      +4.9%     
  HDFCBANK    14     PVT BNK    01-Jun-17   371.3       487.5       392    ₹45,547       +31.3%      +4.9%     
  VBL         17     FMCG       02-Apr-18   37.0        44.6        4528   ₹34,057       +20.3%      +7.1%     
  MPHASIS     8      IT         01-Feb-18   722.3       868.5       242    ₹35,370       +20.2%      +0.6%     
  TCS         19     IT         02-Apr-18   1,178.8     1,415.4     142    ₹33,601       +20.1%      +0.3%     
  TITAN       18     CON DUR    01-Feb-18   807.9       875.0       216    ₹14,490       +8.3%       -2.9%     
  FRETAIL     25     CONSUMP    01-Dec-17   544.8       586.9       322    ₹13,572       +7.7%       +1.4%     
  INDUSINDBK  22     PVT BNK    02-Apr-18   1,717.3     1,822.8     97     ₹10,228       +6.1%       +0.9%     

  AFTER: Invested ₹3,404,069 | Cash ₹155,302 | Total ₹3,559,371 | Positions 18/20 | Slot ₹178,053

========================================================================
  REBALANCE #22  —  01 Aug 2018
  NAV: ₹3,792,213  |  Slot: ₹189,611  |  Cash: ₹155,302
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────Model is not converging.  Current: 4113.699395208769 is not greater than 4113.71590403516. Delta is -0.016508826391145703
Model is not converging.  Current: 4113.920025702234 is not greater than 4113.9641711338145. Delta is -0.044145431580545846
Model is not converging.  Current: 4112.962950447336 is not greater than 4112.9992170210335. Delta is -0.03626657369750319
Model is not converging.  Current: 4235.0675131693015 is not greater than 4235.082397545891. Delta is -0.014884376589179737
Model is not converging.  Current: 4233.648822072 is not greater than 4233.691229474309. Delta is -0.0424074023085268

  VBL         38     FMCG       02-Apr-18   37.0        43.0        4528   ₹27,096       +16.2%    121d  
  INDUSINDBK  34     PVT BNK    02-Apr-18   1,717.3     1,919.4     97     ₹19,606       +11.8%    121d  
  TITAN       36     CON DUR    01-Feb-18   807.9       895.8       216    ₹18,977       +10.9%    181d  
  KOTAKBANK   28     PVT BNK    01-Jun-18   262.2       261.3       679    ₹-625         -0.4%     61d   
  FRETAIL     61     CONSUMP    01-Dec-17   544.8       540.8       322    ₹-1,272       -0.7%     243d  

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ABBOTINDIA  4      HEALTH     2.803    0.29   +80.6%    +23.2%    6,983.6     27     ₹188,558      +4.1%     
  RELIANCE    6      OIL&GAS    2.762    0.79   +50.6%    +25.8%    527.8       359    ₹189,473      +8.4%     
  PAGEIND     9      MFG        2.515    0.68   +82.6%    +26.9%    27,338.4    6      ₹164,030      +4.9%     
  BAJAJFINSV  10     FIN SVC    2.484    0.82   +41.9%    +30.7%    696.2       272    ₹189,375      +5.8%     
  BATAINDIA   11     CON DUR    2.434    0.68   +64.4%    +21.6%    878.9       215    ₹188,973      +7.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  3      FMCG       01-Jun-17   940.4       1,523.6     154    ₹89,805       +62.0%      +3.2%     
  BRITANNIA   1      FMCG       03-Oct-17   1,920.1     2,882.3     85     ₹81,792       +50.1%      +2.0%     
  MPHASIS     5      IT         01-Feb-18   722.3       1,006.5     242    ₹68,776       +39.3%      +6.6%     
  TCS         8      IT         02-Apr-18   1,178.8     1,618.1     142    ₹62,376       +37.3%      +1.6%     
  HDFC        20     PVT BNK    01-Jun-17   371.3       498.6       392    ₹49,902       +34.3%      +0.0%     
  HDFCBANK    21     PVT BNK    01-Jun-17   371.3       498.6       392    ₹49,902       +34.3%      +0.0%     
  BAJFINANCE  2      FIN SVC    01-Jun-18   201.1       264.0       885    ₹55,690       +31.3%      +5.8%     
  COFORGE     7      IT         01-Jun-18   201.6       227.3       883    ₹22,777       +12.8%      +6.2%     
  JUBLFOOD    11     CONSUMP    01-Jun-18   245.3       273.8       725    ₹20,642       +11.6%      -1.1%     
  DMART       15     FMCG       01-Jun-18   1,531.1     1,669.7     116    ₹16,078       +9.1%       +6.2%     
  M&M         22     AUTO       01-Jun-18   829.0       861.9       214    ₹7,025        +4.0%       +1.6%     
  PIDILITIND  23     MFG        01-Jun-18   538.3       543.6       330    ₹1,762        +1.0%       +3.8%     
  TECHM       17     IT         01-Jun-18   516.6       513.2       344    ₹-1,167       -0.7%       +5.1%     

  AFTER: Invested ₹3,631,253 | Cash ₹159,868 | Total ₹3,791,120 | Positions 18/20 | Slot ₹189,611

========================================================================
  REBALANCE #23  —  01 Oct 2018
  NAV: ₹3,631,694  |  Slot: ₹181,585  |  Cash: ₹159,868
========================================================================
  [SECTOR CAP≤4] dropped: INFY, WIPRO, HCLTECH

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HDFC        63     PVT BNK    01-Jun-17   371.3       470.2       392    ₹38,742       +26.6%    487d  
  HDFCBANK    64     PVT BNK    01-Jun-17   371.3       470.2       392    ₹38,742       +26.6%    487d  
  BAJFINANCE  50     FIN SVC    01-Jun-18   201.1       214.1       885    ₹11,556       +6.5%     122d  
  DMART       51     FMCG       01-Jun-18   1,531.1     1,348.8     116    ₹-21,153      -11.9%    122d  
  BAJAJFINSV  45     FIN SVC    01-Aug-18   696.2       585.8       272    ₹-30,026      -15.9%    61d   

  ENTRIES (5)
  [52w filter blocked 1: GRAPHITE(-24.2%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JSWSTEEL    9      METAL      2.334    1.14   +62.5%    +19.4%    351.3       516    ₹181,250      -3.0%     
  DABUR       11     FMCG       2.288    0.72   +43.5%    +15.2%    409.8       443    ₹181,536      -2.0%     
  ABFRL       15     CONSUMP    2.008    0.47   +16.8%    +34.0%    182.2       996    ₹181,430      -1.4%     
  GAIL        16     OIL&GAS    1.952    0.55   +30.2%    +19.7%    91.3        1988   ₹181,523      +1.3%     
  UBL         17     FMCG       1.935    0.95   +55.0%    +18.4%    1,293.7     140    ₹181,118      -1.0%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TCS         1      IT         02-Apr-18   1,178.8     1,846.5     142    ₹94,815       +56.6%      +6.3%     
  HINDUNILVR  23     FMCG       01-Jun-17   940.4       1,442.8     154    ₹77,359       +53.4%      -0.1%     
  BRITANNIA   29     FMCG       03-Oct-17   1,920.1     2,595.3     85     ₹57,393       +35.2%      -2.7%     
  MPHASIS     8      IT         01-Feb-18   722.3       968.9       242    ₹59,679       +34.1%      -3.5%     
  PAGEIND     5      MFG        01-Aug-18   27,338.4    30,462.0    6      ₹18,742       +11.4%      +0.7%     
  TECHM       6      IT         01-Jun-18   516.6       572.3       344    ₹19,168       +10.8%      +1.7%     
  COFORGE     7      IT         01-Jun-18   201.6       214.9       883    ₹11,793       +6.6%       -6.0%     
  RELIANCE    3      OIL&GAS    01-Aug-18   527.8       545.2       359    ₹6,253        +3.3%       -0.5%     
  BATAINDIA   14     CON DUR    01-Aug-18   878.9       903.9       215    ₹5,371        +2.8%       -2.9%     
  ABBOTINDIA  10     HEALTH     01-Aug-18   6,983.6     6,960.2     27     ₹-632         -0.3%       -5.4%     
  JUBLFOOD    22     CONSUMP    01-Jun-18   245.3       243.8       725    ₹-1,068       -0.6%       -7.2%     
  M&M         36     AUTO       01-Jun-18   829.0       785.7       214    ₹-9,271       -5.2%       -7.5%     
  PIDILITIND  27     MFG        01-Jun-18   538.3       503.3       330    ₹-11,531      -6.5%       -5.4%     

  AFTER: Invested ₹3,504,749 | Cash ₹125,868 | Total ₹3,630,617 | Positions 18/20 | Slot ₹181,585

========================================================================
  REBALANCE #24  —  03 Dec 2018
  NAV: ₹3,505,069  |  Slot: ₹175,253  |  Cash: ₹125,868
========================================================================
  [SECTOR CAP≤4] dropped: VBL, INFY

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  Model is not converging.  Current: 4380.064076962502 is not greater than 4380.071521717554. Delta is -0.007444755051437824
Model is not converging.  Current: 4380.659484315784 is not greater than 4380.685807984391. Delta is -0.026323668607801665

  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MPHASIS     42     IT         01-Feb-18   722.3       834.0       242    ₹27,042       +15.5%    305d  
  ABFRL       44     CONSUMP    01-Oct-18   182.2       182.8       996    ₹589          +0.3%     63d   
  RELIANCE    45     OIL&GAS    01-Aug-18   527.8       511.9       359    ₹-5,705       -3.0%     124d  
  DABUR       51     FMCG       01-Oct-18   409.8       386.6       443    ₹-10,290      -5.7%     63d   
  PAGEIND     73     MFG        01-Aug-18   27,338.4    24,355.1    6      ₹-17,900      -10.9%    124d  
  M&M         96     AUTO       01-Jun-18   829.0       705.5       214    ₹-26,441      -14.9%    185d  
  JSWSTEEL    52     METAL      01-Oct-18   351.3       290.4       516    ₹-31,386      -17.3%    63d   

  ENTRIES (7)
  [52w filter blocked 1: JUBILANT(-22.6%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COLPAL      1      FMCG       2.894    0.33   +24.6%    +11.1%    1,072.8     163    ₹174,874      +8.5%     
  AUROPHARMA  4      HEALTH     2.435    1.00   +16.3%    +17.5%    771.7       227    ₹175,169      +2.1%     
  WIPRO       5      IT         2.378    0.33   +12.2%    +11.0%    112.7       1555   ₹175,243      +2.4%     
  ICICIGI     6      FIN SVC    2.362    0.51   +23.3%    +13.0%    818.5       214    ₹175,166      +4.7%     
  HONAUT      7      MFG        2.234    0.79   +33.7%    +4.4%     22,459.3    7      ₹157,215      +7.7%     
  LT          9      INFRA      2.187    1.03   +18.9%    +5.3%     1,270.2     137    ₹174,016      +2.9%     
  HDFC        16     PVT BNK    1.957    0.67   +15.4%    +0.9%     488.1       359    ₹175,245      +4.0%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  2      FMCG       01-Jun-17   940.4       1,613.0     154    ₹103,567      +71.5%      +7.1%     
  BRITANNIA   33     FMCG       03-Oct-17   1,920.1     2,748.7     85     ₹70,434       +43.2%      +4.2%     
  TCS         8      IT         02-Apr-18   1,178.8     1,626.3     142    ₹63,548       +38.0%      +3.4%     
  BATAINDIA   14     CON DUR    01-Aug-18   878.9       993.5       215    ₹24,627       +13.0%      +6.7%     
  JUBLFOOD    28     CONSUMP    01-Jun-18   245.3       261.4       725    ₹11,640       +6.5%       +10.2%    
  TECHM       12     IT         01-Jun-18   516.6       537.0       344    ₹7,024        +4.0%       +1.4%     
  PIDILITIND  11     MFG        01-Jun-18   538.3       556.8       330    ₹6,113        +3.4%       +3.7%     
  ABBOTINDIA  3      HEALTH     01-Aug-18   6,983.6     7,082.9     27     ₹2,681        +1.4%       +4.3%     
  COFORGE     24     IT         01-Jun-18   201.6       203.2       883    ₹1,420        +0.8%       -1.8%     
  GAIL        38     OIL&GAS    01-Oct-18   91.3        85.2        1988   ₹-12,078      -6.7%       +1.8%     
  UBL         43     FMCG       01-Oct-18   1,293.7     1,205.4     140    ₹-12,363      -6.8%       -1.9%     

  AFTER: Invested ₹3,400,295 | Cash ₹103,341 | Total ₹3,503,636 | Positions 18/20 | Slot ₹175,253

========================================================================
  REBALANCE #25  —  01 Feb 2019
  NAV: ₹3,610,566  |  Slot: ₹180,528  |  Cash: ₹103,341
========================================================================
  [SECTOR CAP≤4] dropped: DABUR, MARICO

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TECHM       30     IT         01-Jun-18   516.6       562.6       344    ₹15,811       +8.9%     245d  
  ICICIGI     41     FIN SVC    03-Dec-18   818.5       839.3       214    ₹4,449        +2.5%     60d   
  AUROPHARMA  38     HEALTH     03-Dec-18   771.7       764.9       227    ₹-1,534       -0.9%     60d   
  LT          63     INFRA      03-Dec-18   1,270.2     1,178.6     137    ₹-12,544      -7.2%     60d   
  GAIL        85     OIL&GAS    01-Oct-18   91.3        79.9        1988   ₹-22,631      -12.5%    123d  

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ASIANPAINT  2      CONSUMP    2.849    0.03   +27.5%    +19.7%    1,364.0     132    ₹180,054      +4.3%     
  INFY        6      IT         2.630    0.03   +33.0%    +14.2%    620.1       291    ₹180,436      +5.2%     
  RELIANCE    7      OIL&GAS    2.578    0.20   +30.2%    +18.3%    553.3       326    ₹180,367      +5.0%     
  BAJFINANCE  13     FIN SVC    2.248    0.07   +51.4%    +8.7%     254.9       708    ₹180,442      +2.2%     
  LALPATHLAB  15     HEALTH     2.199    -0.07  +15.2%    +20.0%    492.9       366    ₹180,387      +2.2%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  5      FMCG       01-Jun-17   940.4       1,589.7     154    ₹99,988       +69.0%      +1.8%     
  BRITANNIA   1      FMCG       03-Oct-17   1,920.1     2,887.7     85     ₹82,251       +50.4%      +2.4%     
  TCS         20     IT         02-Apr-18   1,178.8     1,668.9     142    ₹69,594       +41.6%      +4.9%     
  BATAINDIA   3      CON DUR    01-Aug-18   878.9       1,068.3     215    ₹40,720       +21.5%      +1.3%     
  COFORGE     17     IT         01-Jun-18   201.6       243.0       883    ₹36,555       +20.5%      +5.6%     
  WIPRO       12     IT         03-Dec-18   112.7       127.9       1555   ₹23,677       +13.5%      +6.9%     
  UBL         18     FMCG       01-Oct-18   1,293.7     1,409.8     140    ₹16,250       +9.0%       +2.2%     
  JUBLFOOD    14     CONSUMP    01-Jun-18   245.3       266.9       725    ₹15,629       +8.8%       +10.6%    
  ABBOTINDIA  8      HEALTH     01-Aug-18   6,983.6     7,377.5     27     ₹10,634       +5.6%       +1.0%     
  COLPAL      9      FMCG       03-Dec-18   1,072.8     1,090.0     163    ₹2,788        +1.6%       -0.1%     
  PIDILITIND  11     MFG        01-Jun-18   538.3       542.8       330    ₹1,486        +0.8%       -0.2%     
  HDFC        22     PVT BNK    03-Dec-18   488.1       482.9       359    ₹-1,895       -1.1%       -0.3%     
  HONAUT      26     MFG        03-Dec-18   22,459.3    21,361.3    7      ₹-7,686       -4.9%       -0.3%     Model is not converging.  Current: 4510.761127914651 is not greater than 4510.781491267348. Delta is -0.02036335269713163
Model is not converging.  Current: 4470.4071445542695 is not greater than 4470.418935874816. Delta is -0.011791320546763018
Model is not converging.  Current: 4509.238650805785 is not greater than 4509.276003897795. Delta is -0.037353092009652755
Model is not converging.  Current: 4603.923758242833 is not greater than 4603.943515438905. Delta is -0.019757196071623184
Model is not converging.  Current: 4644.567552616699 is not greater than 4644.594770273075. Delta is -0.02721765637579665


  AFTER: Invested ₹3,541,771 | Cash ₹67,725 | Total ₹3,609,495 | Positions 18/20 | Slot ₹180,528

========================================================================
  REBALANCE #26  —  01 Apr 2019
  NAV: ₹3,698,105  |  Slot: ₹184,905  |  Cash: ₹67,725
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HINDUNILVR  57     FMCG       01-Jun-17   940.4       1,493.2     154    ₹85,117       +58.8%    669d  
  BRITANNIA   49     FMCG       03-Oct-17   1,920.1     2,709.0     85     ₹67,063       +41.1%    545d  
  WIPRO       35     IT         03-Dec-18   112.7       120.1       1555   ₹11,533       +6.6%     119d  
  COLPAL      55     FMCG       03-Dec-18   1,072.8     1,060.2     163    ₹-2,053       -1.2%     119d  
  ABBOTINDIA  42     HEALTH     01-Aug-18   6,983.6     6,646.9     27     ₹-9,093       -4.8%     243d  

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TITAN       3      CON DUR    2.559    -0.06  +29.0%    +26.2%    1,094.0     169    ₹184,885      +2.9%     
  VBL         5      FMCG       2.483    0.29   +46.0%    +21.6%    52.0        3557   ₹184,901      +9.5%     
  RBLBANK     6      PVT BNK    2.475    0.45   +47.7%    +17.8%    659.7       280    ₹184,711      +6.6%     
  MUTHOOTFIN  7      FIN SVC    2.471    0.08   +55.7%    +24.5%    537.4       344    ₹184,860      +5.2%     
  HDFCBANK    10     PVT BNK    2.354    0.14   +25.2%    +9.8%     534.0       346    ₹184,764      +3.5%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   1      CON DUR    01-Aug-18   878.9       1,295.6     215    ₹89,592       +47.4%      +3.3%     
  TCS         15     IT         02-Apr-18   1,178.8     1,670.3     142    ₹69,792       +41.7%      +1.5%     
  COFORGE     23     IT         01-Jun-18   201.6       245.6       883    ₹38,913       +21.9%      +1.1%     
  JUBLFOOD    21     CONSUMP    01-Jun-18   245.3       287.0       725    ₹30,211       +17.0%      +5.1%     
  BAJFINANCE  4      FIN SVC    01-Feb-19   254.9       291.0       708    ₹25,610       +14.2%      +5.5%     
  PIDILITIND  13     MFG        01-Jun-18   538.3       605.6       330    ₹22,201       +12.5%      +6.2%     
  RELIANCE    2      OIL&GAS    01-Feb-19   553.3       616.1       326    ₹20,476       +11.4%      +4.9%     
  HDFC        9      PVT BNK    03-Dec-18   488.1       534.0       359    ₹16,461       +9.4%       +3.5%     
  UBL         30     FMCG       01-Oct-18   1,293.7     1,376.4     140    ₹11,579       +6.4%       +2.8%     
  ASIANPAINT  14     CONSUMP    01-Feb-19   1,364.0     1,397.3     132    ₹4,393        +2.4%       +3.0%     
  LALPATHLAB  29     HEALTH     01-Feb-19   492.9       497.3       366    ₹1,629        +0.9%       -0.4%     
  INFY        8      IT         01-Feb-19   620.1       618.5       291    ₹-465         -0.3%       +3.3%     
  HONAUT      19     MFG        03-Dec-18   22,459.3    21,836.7    7      ₹-4,359       -2.8%       +0.4%     

  AFTER: Invested ₹3,555,224 | Cash ₹141,784 | Total ₹3,697,007 | Positions 18/20 | Slot ₹184,905

========================================================================
  REBALANCE #27  —  03 Jun 2019
  NAV: ₹3,827,288  |  Slot: ₹191,364  |  Cash: ₹141,784
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COFORGE     68     IT         01-Jun-18   201.6       241.5       883    ₹35,271       +19.8%    367d  
  PIDILITIND  —      MFG        01-Jun-18   538.3       626.4       330    ₹29,069       +16.4%    367d  
  JUBLFOOD    74     CONSUMP    01-Jun-18   245.3       263.8       725    ₹13,428       +7.6%     367d  
  LALPATHLAB  57     HEALTH     01-Feb-19   492.9       512.7       366    ₹7,271        +4.0%     122d  
  UBL         81     FMCG       01-Oct-18   1,293.7     1,299.1     140    ₹750          +0.4%     245d  
  ASIANPAINT  63     CONSUMP    01-Feb-19   1,364.0     1,366.0     132    ₹253          +0.1%     122d  
  INFY        —      IT         01-Feb-19   620.1       609.9       291    ₹-2,955       -1.6%     122d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ICICIGI     1      FIN SVC    2.975    0.25   +67.9%    +33.3%    1,164.0     164    ₹190,888      +8.1%     
  NAUKRI      2      IT         2.909    0.05   +97.5%    +39.0%    451.4       423    ₹190,937      +15.9%    
  GUJGASLTD   4      OIL&GAS    2.694    0.57   +10.2%    +55.1%    177.2       1080   ₹191,353      +9.8%     
  BAJAJFINSV  6      FIN SVC    2.509    0.25   +39.8%    +31.8%    831.5       230    ₹191,242      +4.8%     
  AXISBANK    9      PVT BNK    2.362    0.37   +55.3%    +15.5%    808.3       236    ₹190,751      +4.0%     
  PIIND       10     MFG        2.361    0.29   +35.4%    +23.7%    1,116.0     171    ₹190,836      +4.8%     
  SRF         11     MFG        2.351    0.42   +51.7%    +27.8%    558.0       342    ₹190,820      +2.6%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TCS         25     IT         02-Apr-18   1,178.8     1,843.5     142    ₹94,384       +56.4%      +5.4%     
  BATAINDIA   12     CON DUR    01-Aug-18   878.9       1,272.7     215    ₹84,663       +44.8%      +0.7%     
  BAJFINANCE  5      FIN SVC    01-Feb-19   254.9       340.4       708    ₹60,595       +33.6%      +6.1%     
  HONAUT      3      MFG        03-Dec-18   22,459.3    26,282.6    7      ₹26,763       +17.0%      +7.2%     
  HDFC        7      PVT BNK    03-Dec-18   488.1       567.6       359    ₹28,510       +16.3%      +3.3%     
  TITAN       13     CON DUR    01-Apr-19   1,094.0     1,236.5     169    ₹24,077       +13.0%      +5.1%     
  RELIANCE    23     OIL&GAS    01-Feb-19   553.3       602.1       326    ₹15,909       +8.8%       +2.7%     
  MUTHOOTFIN  14     FIN SVC    01-Apr-19   537.4       573.4       344    ₹12,399       +6.7%       +4.1%     Model is not converging.  Current: 4788.6147847837465 is not greater than 4788.629704573907. Delta is -0.014919790160092816
Model is not converging.  Current: 4789.303813402721 is not greater than 4789.345550270361. Delta is -0.04173686763988371
Model is not converging.  Current: 4787.259199243144 is not greater than 4787.3122326998355. Delta is -0.05303345669108239
Model is not converging.  Current: 4901.176694466442 is not greater than 4901.194532385332. Delta is -0.017837918890108995
Model is not converging.  Current: 4899.919278214822 is not greater than 4899.93726031219. Delta is -0.01798209736807621

  HDFCBANK    8      PVT BNK    01-Apr-19   534.0       567.6       346    ₹11,613       +6.3%       +3.3%     
  VBL         22     FMCG       01-Apr-19   52.0        54.8        3557   ₹10,022       +5.4%       +3.0%     
  RBLBANK     19     PVT BNK    01-Apr-19   659.7       675.0       280    ₹4,293        +2.3%       +2.7%     

  AFTER: Invested ₹3,683,803 | Cash ₹141,897 | Total ₹3,825,700 | Positions 18/20 | Slot ₹191,364

========================================================================
  REBALANCE #28  —  01 Aug 2019
  NAV: ₹3,461,810  |  Slot: ₹173,090  |  Cash: ₹141,897
========================================================================
  [SECTOR CAP≤4] dropped: HDFCLIFE

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HDFC        67     PVT BNK    03-Dec-18   488.1       517.5       359    ₹10,555       +6.0%     241d  
  HDFCBANK    68     PVT BNK    01-Apr-19   534.0       517.5       346    ₹-5,692       -3.1%     122d  
  TITAN       47     CON DUR    01-Apr-19   1,094.0     1,037.1     169    ₹-9,613       -5.2%     122d  
  RELIANCE    80     OIL&GAS    01-Feb-19   553.3       522.4       326    ₹-10,058      -5.6%     181d  
  BAJAJFINSV  55     FIN SVC    03-Jun-19   831.5       705.6       230    ₹-28,951      -15.1%    59d   
  AXISBANK    50     PVT BNK    03-Jun-19   808.3       666.5       236    ₹-33,453      -17.5%    59d   
  RBLBANK     130    PVT BNK    01-Apr-19   659.7       385.4       280    ₹-76,805      -41.6%    122d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  POWERGRID   2      ENERGY     2.746    0.37   +26.1%    +12.1%    82.9        2086   ₹173,027      +2.9%     
  SBILIFE     5      FIN SVC    2.605    0.28   +23.0%    +17.0%    767.6       225    ₹172,718      +0.9%     
  APOLLOHOSP  6      HEALTH     2.570    0.42   +43.8%    +13.9%    1,337.9     129    ₹172,591      +0.9%     
  ABBOTINDIA  7      HEALTH     2.518    0.28   +16.3%    +14.3%    7,695.4     22     ₹169,299      -1.5%     
  TRENT       10     CONSUMP    2.351    0.43   +28.7%    +9.8%     418.6       413    ₹172,889      -0.0%     
  DABUR       12     FMCG       2.158    0.19   +13.7%    +11.9%    398.2       434    ₹172,825      +1.5%     
  DIVISLAB    14     HEALTH     2.082    0.26   +51.0%    -5.1%     1,539.0     112    ₹172,370      -0.8%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TCS         30     IT         02-Apr-18   1,178.8     1,811.0     142    ₹89,776       +53.6%      +1.5%     
  BATAINDIA   13     CON DUR    01-Aug-18   878.9       1,221.7     215    ₹73,683       +39.0%      -2.1%     
  BAJFINANCE  23     FIN SVC    01-Feb-19   254.9       313.0       708    ₹41,146       +22.8%      -3.3%     
  VBL         16     FMCG       01-Apr-19   52.0        54.2        3557   ₹7,940        +4.3%       -0.8%     
  MUTHOOTFIN  9      FIN SVC    01-Apr-19   537.4       544.8       344    ₹2,542        +1.4%       +0.0%     
  HONAUT      34     MFG        03-Dec-18   22,459.3    22,664.6    7      ₹1,437        +0.9%       -0.7%     
  PIIND       4      MFG        03-Jun-19   1,116.0     1,093.1     171    ₹-3,921       -2.1%       -0.1%     
  NAUKRI      1      IT         03-Jun-19   451.4       433.1       423    ₹-7,734       -4.1%       +0.1%     
  ICICIGI     8      FIN SVC    03-Jun-19   1,164.0     1,101.5     164    ₹-10,241      -5.4%       +2.9%     
  SRF         3      MFG        03-Jun-19   558.0       515.0       342    ₹-14,682      -7.7%       -3.8%     
  GUJGASLTD   22     OIL&GAS    03-Jun-19   177.2       161.8       1080   ₹-16,590      -8.7%       +2.8%     

  AFTER: Invested ₹3,387,684 | Cash ₹72,693 | Total ₹3,460,378 | Positions 18/20 | Slot ₹173,090

========================================================================
  REBALANCE #29  —  01 Oct 2019
  NAV: ₹3,744,285  |  Slot: ₹187,214  |  Cash: ₹72,693
========================================================================
  [SECTOR CAP≤4] dropped: HDFCLIFE, PGHH

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TCS         76     IT         02-Apr-18   1,178.8     1,711.2     142    ₹75,608       +45.2%    547d  
  DABUR       43     FMCG       01-Aug-19   398.2       413.0       434    ₹6,409        +3.7%     61d   
  DIVISLAB    38     HEALTH     01-Aug-19   1,539.0     1,562.3     112    ₹2,605        +1.5%     61d   
  VBL         57     FMCG       01-Apr-19   52.0        52.7        3557   ₹2,407        +1.3%     183d  
  APOLLOHOSP  39     HEALTH     01-Aug-19   1,337.9     1,346.9     129    ₹1,164        +0.7%     61d   
  ICICIGI     26     FIN SVC    03-Jun-19   1,164.0     1,151.0     164    ₹-2,123       -1.1%     120d  
  SRF         55     MFG        03-Jun-19   558.0       524.9       342    ₹-11,291      -5.9%     120d  
  POWERGRID   70     ENERGY     01-Aug-19   82.9        77.1        2086   ₹-12,246      -7.1%     61d   
  GUJGASLTD   42     OIL&GAS    03-Jun-19   177.2       163.7       1080   ₹-14,567      -7.6%     120d  
  NAUKRI      52     IT         03-Jun-19   451.4       403.6       423    ₹-20,233      -10.6%    120d  

  ENTRIES (10)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COLPAL      1      FMCG       3.341    0.51   +36.9%    +32.6%    1,276.3     146    ₹186,346      +7.4%     
  HDFCAMC     2      FIN SVC    3.315    0.63   +74.0%    +33.9%    1,208.6     154    ₹186,124      +1.9%     
  BERGEPAINT  4      CON DUR    3.162    0.69   +43.0%    +37.3%    350.7       533    ₹186,949      +8.3%     
  ASIANPAINT  5      CONSUMP    2.991    0.56   +36.0%    +30.0%    1,662.1     112    ₹186,157      +6.0%     
  NESTLEIND   8      FMCG       2.616    0.37   +38.1%    +16.9%    636.8       293    ₹186,590      +3.8%     
  LALPATHLAB  10     HEALTH     2.413    0.56   +38.6%    +25.3%    634.6       295    ₹187,210      +1.8%     
  DMART       11     FMCG       2.354    0.70   +24.3%    +35.6%    1,896.0     98     ₹185,808      +9.4%     
  HINDUNILVR  12     FMCG       2.224    0.42   +25.6%    +11.3%    1,771.2     105    ₹185,975      +3.2%     
  WHIRLPOOL   13     CON DUR    2.186    0.54   +22.7%    +23.4%    1,921.6     97     ₹186,400      +10.5%    
  KOTAKBANK   15     PVT BNK    2.148    0.53   +34.0%    +11.6%    328.3       570    ₹187,113      +6.1%     Model is not converging.  Current: 5019.110150190949 is not greater than 5019.111478408604. Delta is -0.0013282176551001612
Model is not converging.  Current: 5019.512855448488 is not greater than 5019.522232246583. Delta is -0.009376798095217964


  HOLDS (8)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   3      CON DUR    01-Aug-18   878.9       1,630.0     215    ₹161,480      +85.5%      +6.5%     
  BAJFINANCE  19     FIN SVC    01-Feb-19   254.9       387.8       708    ₹94,120       +52.2%      +8.0%     
  ABBOTINDIA  7      HEALTH     01-Aug-19   7,695.4     9,965.1     22     ₹49,933       +29.5%      +6.8%     
  HONAUT      16     MFG        03-Dec-18   22,459.3    27,751.2    7      ₹37,043       +23.6%      +4.4%     
  TRENT       23     CONSUMP    01-Aug-19   418.6       475.7       413    ₹23,581       +13.6%      -0.2%     
  MUTHOOTFIN  22     FIN SVC    01-Apr-19   537.4       600.6       344    ₹21,734       +11.8%      +5.1%     
  PIIND       9      MFG        03-Jun-19   1,116.0     1,235.2     171    ₹20,376       +10.7%      +0.5%     
  SBILIFE     17     FIN SVC    01-Aug-19   767.6       817.7       225    ₹11,259       +6.5%       +1.6%     

  AFTER: Invested ₹3,701,430 | Cash ₹40,640 | Total ₹3,742,070 | Positions 18/20 | Slot ₹187,214

========================================================================
  REBALANCE #30  —  02 Dec 2019
  NAV: ₹3,967,131  |  Slot: ₹198,357  |  Cash: ₹40,640
========================================================================
  [SECTOR CAP≤4] dropped: NAM-INDIA, BAJAJFINSV

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HONAUT      37     MFG        03-Dec-18   22,459.3    26,924.4    7      ₹31,256       +19.9%    364d  
  MUTHOOTFIN  48     FIN SVC    01-Apr-19   537.4       604.4       344    ₹23,054       +12.5%    245d  
  HINDUNILVR  34     FMCG       01-Oct-19   1,771.2     1,827.7     105    ₹5,935        +3.2%     62d   
  KOTAKBANK   32     PVT BNK    01-Oct-19   328.3       325.2       570    ₹-1,726       -0.9%     62d   
  ASIANPAINT  36     CONSUMP    01-Oct-19   1,662.1     1,638.9     112    ₹-2,597       -1.4%     62d   
  DMART       30     FMCG       01-Oct-19   1,896.0     1,846.7     98     ₹-4,831       -2.6%     62d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IGL         7      OIL&GAS    2.491    0.12   +56.4%    +26.9%    185.5       1069   ₹198,277      +2.4%     
  MANAPPURAM  8      FIN SVC    2.365    0.17   +87.0%    +28.7%    139.6       1420   ₹198,253      -1.9%     
  GUJGASLTD   11     OIL&GAS    2.246    0.08   +74.3%    +21.1%    206.7       959    ₹198,179      +8.8%     
  RELIANCE    12     OIL&GAS    2.215    0.16   +41.6%    +24.4%    706.5       280    ₹197,809      +4.8%     
  SIEMENS     13     ENERGY     2.192    0.19   +52.8%    +23.9%    841.1       235    ₹197,650      -3.9%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   21     CON DUR    01-Aug-18   878.9       1,526.7     215    ₹139,266      +73.7%      -2.5%     
  BAJFINANCE  20     FIN SVC    01-Feb-19   254.9       383.7       708    ₹91,235       +50.6%      -3.4%     
  ABBOTINDIA  2      HEALTH     01-Aug-19   7,695.4     11,580.2    22     ₹85,464       +50.5%      +2.8%     
  PIIND       3      MFG        03-Jun-19   1,116.0     1,478.3     171    ₹61,955       +32.5%      +5.3%     
  TRENT       29     CONSUMP    01-Aug-19   418.6       523.2       413    ₹43,181       +25.0%      +2.6%     
  HDFCAMC     1      FIN SVC    01-Oct-19   1,208.6     1,498.3     154    ₹44,614       +24.0%      -0.9%     
  SBILIFE     17     FIN SVC    01-Aug-19   767.6       932.5       225    ₹37,089       +21.5%      -1.4%     
  LALPATHLAB  4      HEALTH     01-Oct-19   634.6       760.0       295    ₹36,980       +19.8%      +2.5%     
  BERGEPAINT  6      CON DUR    01-Oct-19   350.7       400.4       533    ₹26,461       +14.2%      +1.3%     
  WHIRLPOOL   5      CON DUR    01-Oct-19   1,921.6     2,121.8     97     ₹19,418       +10.4%      -0.8%     
  NESTLEIND   14     FMCG       01-Oct-19   636.8       677.6       293    ₹11,948       +6.4%       +0.9%     
  COLPAL      16     FMCG       01-Oct-19   1,276.3     1,248.1     146    ₹-4,116       -2.2%       -3.8%     

  AFTER: Invested ₹3,778,439 | Cash ₹187,516 | Total ₹3,965,955 | Positions 17/20 | Slot ₹198,357

========================================================================
  REBALANCE #31  —  03 Feb 2020
  NAV: ₹4,245,826  |  Slot: ₹212,291  |  Cash: ₹187,516
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SBILIFE     51     FIN SVC    01-Aug-19   767.6       901.7       225    ₹30,153       +17.5%    186d  
  SIEMENS     68     ENERGY     02-Dec-19   841.1       821.8       235    ₹-4,524       -2.3%     63d   
  COLPAL      112    FMCG       01-Oct-19   1,276.3     1,161.5     146    ₹-16,771      -9.0%     125d  
  RELIANCE    75     OIL&GAS    02-Dec-19   706.5       617.0       280    ₹-25,061      -12.7%    63d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COROMANDEL  2      MFG        3.274    0.04   +45.3%    +35.1%    585.9       362    ₹212,110      +6.6%     
  SRF         4      MFG        3.093    -0.04  +87.1%    +29.3%    741.4       286    ₹212,043      +4.2%     
  JUBLFOOD    5      CONSUMP    2.586    0.34   +66.3%    +25.1%    385.8       550    ₹212,190      +9.9%     
  COFORGE     8      IT         2.407    0.21   +49.5%    +18.8%    349.4       607    ₹212,087      +2.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   13     CON DUR    01-Aug-18   878.9       1,736.3     215    ₹184,329      +97.5%      +2.9%     
  BAJFINANCE  25     FIN SVC    01-Feb-19   254.9       423.4       708    ₹119,356      +66.1%      +3.0%     Model is not converging.  Current: 4966.95563238883 is not greater than 4966.9761064714985. Delta is -0.02047408266844286
Model is not converging.  Current: 4966.94488103774 is not greater than 4967.035111153697. Delta is -0.0902301159567287
Model is not converging.  Current: 4966.937242753511 is not greater than 4966.972972938287. Delta is -0.035730184776184615
Model is not converging.  Current: 4915.66749361614 is not greater than 4915.6707110020525. Delta is -0.0032173859126487514
Model is not converging.  Current: 4911.258458564445 is not greater than 4911.289799707518. Delta is -0.03134114307249547
Model is not converging.  Current: 4911.218038245066 is not greater than 4911.233254955781. Delta is -0.015216710715321824

  ABBOTINDIA  17     HEALTH     01-Aug-19   7,695.4     11,694.8    22     ₹87,987       +52.0%      +0.5%     
  TRENT       18     CONSUMP    01-Aug-19   418.6       590.4       413    ₹70,949       +41.0%      +3.4%     
  PIIND       7      MFG        03-Jun-19   1,116.0     1,520.1     171    ₹69,095       +36.2%      +3.5%     
  BERGEPAINT  6      CON DUR    01-Oct-19   350.7       461.5       533    ₹59,024       +31.6%      +4.1%     
  LALPATHLAB  10     HEALTH     01-Oct-19   634.6       830.1       295    ₹57,674       +30.8%      +4.0%     
  GUJGASLTD   1      OIL&GAS    02-Dec-19   206.7       269.8       959    ₹60,542       +30.5%      +2.3%     
  WHIRLPOOL   20     CON DUR    01-Oct-19   1,921.6     2,361.3     97     ₹42,649       +22.9%      -0.1%     
  IGL         3      OIL&GAS    02-Dec-19   185.5       227.3       1069   ₹44,665       +22.5%      +6.7%     
  NESTLEIND   14     FMCG       01-Oct-19   636.8       761.7       293    ₹36,587       +19.6%      +6.2%     
  MANAPPURAM  19     FIN SVC    02-Dec-19   139.6       163.8       1420   ₹34,295       +17.3%      +1.0%     
  HDFCAMC     11     FIN SVC    01-Oct-19   1,208.6     1,354.2     154    ₹22,416       +12.0%      -3.1%     

  AFTER: Invested ₹4,168,418 | Cash ₹76,400 | Total ₹4,244,818 | Positions 17/20 | Slot ₹212,291

========================================================================
  REBALANCE #32  —  01 Apr 2020
  NAV: ₹3,237,075  |  Slot: ₹161,854  |  Cash: ₹76,400
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, DIVISLAB

  [REGIME OFF] Nifty 200 HMM bear-state prob 1.00 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  1      HEALTH     01-Aug-19   7,695.4     14,325.7    22     ₹145,866      +86.2%      +6.4%     
  BATAINDIA   64     CON DUR    01-Aug-18   878.9       1,116.5     215    ₹51,065       +27.0% ⚠    -9.2%     
  NESTLEIND   3      FMCG       01-Oct-19   636.8       731.5       293    ₹27,736       +14.9%      +3.6%     
  BERGEPAINT  7      CON DUR    01-Oct-19   350.7       392.5       533    ₹22,267       +11.9%      +0.5%     
  TRENT       15     CONSUMP    01-Aug-19   418.6       462.8       413    ₹18,266       +10.6%      -11.9%    
  PIIND       21     MFG        03-Jun-19   1,116.0     1,175.0     171    ₹10,092       +5.3%       -3.7%     
  LALPATHLAB  12     HEALTH     01-Oct-19   634.6       664.5       295    ₹8,827        +4.7%       -5.6%     
  GUJGASLTD   9      OIL&GAS    02-Dec-19   206.7       216.2       959    ₹9,145        +4.6%       -6.3%     
  IGL         13     OIL&GAS    02-Dec-19   185.5       174.2       1069   ₹-12,108      -6.1%       +2.5%     
  WHIRLPOOL   30     CON DUR    01-Oct-19   1,921.6     1,742.9     97     ₹-17,335      -9.3%       -8.2%     
  COROMANDEL  11     MFG        03-Feb-20   585.9       497.7       362    ₹-31,952      -15.1%      -3.2%     
  BAJFINANCE  67     FIN SVC    01-Feb-19   254.9       216.1       708    ₹-27,464      -15.2% ⚠    -27.3%    
  HDFCAMC     20     FIN SVC    01-Oct-19   1,208.6     962.5       154    ₹-37,901      -20.4%      -9.0%     
  JUBLFOOD    27     CONSUMP    03-Feb-20   385.8       273.9       550    ₹-61,569      -29.0%      -5.2%     
  SRF         28     MFG        03-Feb-20   741.4       520.3       286    ₹-63,227      -29.8%      -14.5%    
  COFORGE     34     IT         03-Feb-20   349.4       216.9       607    ₹-80,445      -37.9%      -10.1%    
  MANAPPURAM  61     FIN SVC    02-Dec-19   139.6       83.7        1420   ₹-79,437      -40.1%      -16.9%    
  ⚠  WAZ < 0 (momentum below universe mean): BATAINDIA, BAJFINANCE

  AFTER: Invested ₹3,160,675 | Cash ₹76,400 | Total ₹3,237,075 | Positions 17/20 | Slot ₹161,854

========================================================================
  REBALANCE #33  —  01 Jun 2020
  NAV: ₹3,745,762  |  Slot: ₹187,288  |  Cash: ₹76,400
========================================================================
  [SECTOR CAP≤4] dropped: DIVISLAB, IPCALAB, SYNGENE, TORNTPHARM, AUROPHARMA, SUNPHARMA

  [REGIME OFF] Nifty 200 HMM bear-state prob 0.80 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  1      HEALTH     01-Aug-19   7,695.4     15,437.7    22     ₹170,330      +100.6%     -0.9%     
  BATAINDIA   —      CON DUR    01-Aug-18   878.9       1,302.1     215    ₹90,984       +48.1%      +5.7%     
  PIIND       16     MFG        03-Jun-19   1,116.0     1,552.3     171    ₹74,607       +39.1%      +4.1%     
  NESTLEIND   5      FMCG       01-Oct-19   636.8       802.9       293    ₹48,668       +26.1%      +2.0%     
  TRENT       55     CONSUMP    01-Aug-19   418.6       504.8       413    ₹35,608       +20.6% ⚠    +10.0%    
  LALPATHLAB  23     HEALTH     01-Oct-19   634.6       728.6       295    ₹27,730       +14.8%      -1.4%     
  BERGEPAINT  21     CON DUR    01-Oct-19   350.7       397.9       533    ₹25,123       +13.4%      +4.6%     
  GUJGASLTD   26     OIL&GAS    02-Dec-19   206.7       233.9       959    ₹26,156       +13.2%      +0.9%     
  IGL         12     OIL&GAS    02-Dec-19   185.5       209.8       1069   ₹26,027       +13.1%      +2.0%     
  WHIRLPOOL   24     CON DUR    01-Oct-19   1,921.6     1,998.5     97     ₹7,454        +4.0%       +6.4%     
  COROMANDEL  6      MFG        03-Feb-20   585.9       605.6       362    ₹7,125        +3.4%       +4.7%     
  SRF         35     MFG        03-Feb-20   741.4       726.6       286    ₹-4,249       -2.0%       +6.3%     
  HDFCAMC     25     FIN SVC    01-Oct-19   1,208.6     1,168.9     154    ₹-6,120       -3.3%       +5.5%     
  JUBLFOOD    —      CONSUMP    03-Feb-20   385.8       331.8       550    ₹-29,677      -14.0%      +4.1%     
  MANAPPURAM  54     FIN SVC    02-Dec-19   139.6       117.8       1420   ₹-31,033      -15.7%      +9.3%     
  BAJFINANCE  92     FIN SVC    01-Feb-19   254.9       210.4       708    ₹-31,496      -17.5%      +5.3%     
  COFORGE     46     IT         03-Feb-20   349.4       272.4       607    ₹-46,725      -22.0% ⚠    +4.4%     
  ⚠  WAZ < 0 (momentum below universe mean): COFORGE, TRENT

  AFTER: Invested ₹3,669,361 | Cash ₹76,400 | Total ₹3,745,762 | Positions 17/20 | Slot ₹187,288

========================================================================
  REBALANCE #34  —  03 Aug 2020
  NAV: ₹4,055,701  |  Slot: ₹202,785  |  Cash: ₹76,400
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, TORNTPHARM, DIVISLAB, INFY, AJANTPHARM

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BATAINDIA   —      CON DUR    01-Aug-18   878.9       1,180.7     215    ₹64,867       +34.3%    733d  
  TRENT       47     CONSUMP    01-Aug-19   418.6       550.7       413    ₹54,562       +31.6%    368d  Model is not converging.  Current: 4901.783323808414 is not greater than 4901.793128328795. Delta is -0.009804520381294424
Model is not converging.  Current: 4901.818190653518 is not greater than 4901.826126511875. Delta is -0.007935858357086545
Model is not converging.  Current: 4901.758379060151 is not greater than 4901.768896338843. Delta is -0.010517278691622778
Model is not converging.  Current: 4899.43927700224 is not greater than 4899.473149340901. Delta is -0.033872338661240065
Model is not converging.  Current: 4899.504293595515 is not greater than 4899.529963961169. Delta is -0.025670365654150373
Model is not converging.  Current: 4899.4362342472095 is not greater than 4899.446838419639. Delta is -0.010604172429339087

  NESTLEIND   51     FMCG       01-Oct-19   636.8       775.0       293    ₹40,489       +21.7%    307d  
  BAJFINANCE  28     FIN SVC    01-Feb-19   254.9       309.1       708    ₹38,374       +21.3%    549d  
  WHIRLPOOL   46     CON DUR    01-Oct-19   1,921.6     2,080.9     97     ₹15,447       +8.3%     307d  
  MANAPPURAM  37     FIN SVC    02-Dec-19   139.6       142.6       1420   ₹4,268        +2.2%     245d  
  SRF         48     MFG        03-Feb-20   741.4       751.3       286    ₹2,828        +1.3%     182d  
  IGL         88     OIL&GAS    02-Dec-19   185.5       175.4       1069   ₹-10,749      -5.4%     245d  
  HDFCAMC     77     FIN SVC    01-Oct-19   1,208.6     1,080.2     154    ₹-19,780      -10.6%    307d  
  JUBLFOOD    —      CONSUMP    03-Feb-20   385.8       344.3       550    ₹-22,849      -10.8%    182d  

  ENTRIES (10)
  [52w filter blocked 1: ENDURANCE(-22.9%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JUBILANT    1      CONSUMP    3.755    0.79   +100.3%   +127.5%   618.2       328    ₹202,764      +16.9%    
  SYNGENE     3      HEALTH     2.811    0.44   +48.5%    +51.9%    473.6       428    ₹202,686      +7.9%     
  MUTHOOTFIN  4      FIN SVC    2.672    1.12   +115.6%   +64.0%    1,172.9     172    ₹201,740      +4.0%     
  MPHASIS     5      IT         2.547    0.54   +30.2%    +66.8%    1,024.9     197    ₹201,914      +9.8%     
  BALKRISIND  6      MFG        2.349    0.94   +81.9%    +47.5%    1,249.0     162    ₹202,330      +3.8%     
  IPCALAB     7      HEALTH     2.213    0.23   +103.0%   +19.8%    927.0       218    ₹202,079      +7.8%     
  WIPRO       10     IT         1.986    0.61   +6.9%     +52.9%    129.8       1562   ₹202,701      +7.9%     
  HCLTECH     11     IT         1.960    0.71   +41.1%    +36.1%    559.9       362    ₹202,674      +8.0%     
  BRITANNIA   19     FMCG       1.591    0.74   +40.7%    +26.8%    3,411.0     59     ₹201,251      +0.6%     
  ATGL        21     OIL&GAS    1.506    0.73   -3.8%     +60.8%    156.2       1297   ₹202,645      +3.5%     

  HOLDS (7)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  35     HEALTH     01-Aug-19   7,695.4     14,688.6    22     ₹153,849      +90.9%      +4.1%     
  PIIND       15     MFG        03-Jun-19   1,116.0     1,809.4     171    ₹118,570      +62.1%      +6.6%     
  GUJGASLTD   9      OIL&GAS    02-Dec-19   206.7       291.7       959    ₹81,576       +41.2%      +3.6%     
  LALPATHLAB  14     HEALTH     01-Oct-19   634.6       892.7       295    ₹76,125       +40.7%      +0.5%     
  COROMANDEL  2      MFG        03-Feb-20   585.9       737.5       362    ₹54,865       +25.9%      +1.3%     
  BERGEPAINT  20     CON DUR    01-Oct-19   350.7       426.3       533    ₹40,291       +21.6%      +1.5%     
  COFORGE     16     IT         03-Feb-20   349.4       362.1       607    ₹7,717        +3.6%       +12.5%    

  AFTER: Invested ₹3,912,446 | Cash ₹140,853 | Total ₹4,053,299 | Positions 17/20 | Slot ₹202,785

========================================================================
  REBALANCE #35  —  01 Oct 2020
  NAV: ₹4,361,176  |  Slot: ₹218,059  |  Cash: ₹140,853
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, APOLLOHOSP, CIPLA, SANOFI, TORNTPHARM

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LALPATHLAB  28     HEALTH     01-Oct-19   634.6       891.4       295    ₹75,758       +40.5%    366d  
  COFORGE     11     IT         03-Feb-20   349.4       438.6       607    ₹54,119       +25.5%    241d  
  BRITANNIA   37     FMCG       03-Aug-20   3,411.0     3,515.2     59     ₹6,144        +3.1%     59d   
  JUBILANT    —      OTHER      03-Aug-20   618.2       527.1       328    ₹-29,878      -14.7%    59d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIENT    2      METAL      3.375    1.08   +103.2%   +87.7%    307.3       709    ₹217,852      +8.0%     
  DIVISLAB    3      HEALTH     3.215    0.55   +84.3%    +41.8%    2,973.1     73     ₹217,034      -1.9%     
  NAVINFLUOR  5      MFG        2.919    0.71   +180.4%   +24.5%    2,095.0     104    ₹217,880      +3.6%     
  INFY        15     IT         1.843    0.86   +33.5%    +33.2%    867.6       251    ₹217,758      +3.0%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  23     HEALTH     01-Aug-19   7,695.4     15,256.9    22     ₹166,352      +98.3%      +0.2%     
  PIIND       19     MFG        03-Jun-19   1,116.0     1,913.5     171    ₹136,365      +71.5%      +0.2%     
  GUJGASLTD   26     OIL&GAS    02-Dec-19   206.7       297.7       959    ₹87,338       +44.1%      +1.9%     
  BERGEPAINT  21     CON DUR    01-Oct-19   350.7       479.7       533    ₹68,746       +36.8%      +2.9%     
  COROMANDEL  14     MFG        03-Feb-20   585.9       749.7       362    ₹59,264       +27.9%      +0.9%     
  ATGL        27     OIL&GAS    03-Aug-20   156.2       193.0       1297   ₹47,715       +23.5%      +2.2%     
  IPCALAB     2      HEALTH     03-Aug-20   927.0       1,106.3     218    ₹39,097       +19.3%      +7.1%     
  MPHASIS     8      IT         03-Aug-20   1,024.9     1,215.0     197    ₹37,451       +18.5%      +4.9%     
  SYNGENE     7      HEALTH     03-Aug-20   473.6       554.2       428    ₹34,516       +17.0%      +3.6%     
  HCLTECH     6      IT         03-Aug-20   559.9       646.5       362    ₹31,368       +15.5%      +3.5%     
  BALKRISIND  17     MFG        03-Aug-20   1,249.0     1,397.6     162    ₹24,076       +11.9%      +6.1%     
  WIPRO       10     IT         03-Aug-20   129.8       144.3       1562   ₹22,678       +11.2%      +3.1%     
  MUTHOOTFIN  32     FIN SVC    03-Aug-20   1,172.9     1,055.2     172    ₹-20,238      -10.0%      +3.8%     

  AFTER: Invested ₹4,181,393 | Cash ₹178,750 | Total ₹4,360,142 | Positions 17/20 | Slot ₹218,059

========================================================================
  REBALANCE #36  —  01 Dec 2020
  NAV: ₹4,957,658  |  Slot: ₹247,883  |  Cash: ₹178,750
========================================================================
  [SECTOR CAP≤4] dropped: MRF, TATACHEM, LALPATHLAB, SRF

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  Model is not converging.  Current: 4886.17812961604 is not greater than 4886.185997770517. Delta is -0.007868154477364442
Model is not converging.  Current: 4886.2012868298425 is not greater than 4886.21371355208. Delta is -0.012426722237250942

  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ABBOTINDIA  87     HEALTH     01-Aug-19   7,695.4     14,338.5    22     ₹146,147      +86.3%    488d  
  MUTHOOTFIN  51     FIN SVC    03-Aug-20   1,172.9     1,055.1     172    ₹-20,254      -10.0%    120d  

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  APOLLOHOSP  3      HEALTH     2.631    -0.09  +71.1%    +47.7%    2,431.2     101    ₹245,552      +8.8%     
  ADANIENSOL  16     ENERGY     1.772    -0.17  +35.4%    +43.8%    379.0       654    ₹247,833      +8.4%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       360.4       1297   ₹264,733      +130.6%     +22.2%    
  PIIND       13     MFG        03-Jun-19   1,116.0     2,294.6     171    ₹201,543      +105.6%     +1.8%     
  GUJGASLTD   24     OIL&GAS    02-Dec-19   206.7       326.4       959    ₹114,847      +58.0%      +5.8%     
  BERGEPAINT  23     CON DUR    01-Oct-19   350.7       532.5       533    ₹96,868       +51.8%      +2.4%     
  ADANIENT    4      METAL      01-Oct-20   307.3       420.7       709    ₹80,417       +36.9%      +10.8%    
  COROMANDEL  17     MFG        03-Feb-20   585.9       778.3       362    ₹69,618       +32.8%      +6.0%     
  BALKRISIND  12     MFG        03-Aug-20   1,249.0     1,584.9     162    ₹54,420       +26.9%      +5.6%     
  NAVINFLUOR  2      MFG        01-Oct-20   2,095.0     2,638.5     104    ₹56,526       +25.9%      +6.2%     
  WIPRO       9      IT         03-Aug-20   129.8       162.6       1562   ₹51,332       +25.3%      +1.5%     
  IPCALAB     11     HEALTH     03-Aug-20   927.0       1,120.5     218    ₹42,182       +20.9%      +5.0%     
  HCLTECH     18     IT         03-Aug-20   559.9       666.4       362    ₹38,564       +19.0%      +0.6%     
  SYNGENE     6      HEALTH     03-Aug-20   473.6       563.1       428    ₹38,308       +18.9%      +1.1%     
  DIVISLAB    5      HEALTH     01-Oct-20   2,973.1     3,512.3     73     ₹39,361       +18.1%      +5.7%     
  MPHASIS     22     IT         03-Aug-20   1,024.9     1,173.6     197    ₹29,292       +14.5%      -1.1%     
  INFY        8      IT         01-Oct-20   867.6       980.5       251    ₹28,336       +13.0%      +2.3%     

  AFTER: Invested ₹4,775,362 | Cash ₹181,710 | Total ₹4,957,072 | Positions 17/20 | Slot ₹247,883

========================================================================
  REBALANCE #37  —  01 Feb 2021
  NAV: ₹5,204,435  |  Slot: ₹260,222  |  Cash: ₹181,710
========================================================================
  [SECTOR CAP≤4] dropped: MRF

  [REGIME OFF] Nifty 200 HMM bear-state prob 1.00 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       388.5       1297   ₹301,184      +148.6%     +5.3%     
  PIIND       90     MFG        03-Jun-19   1,116.0     1,966.5     171    ₹145,440      +76.2% ⚠    -7.7%     
  ADANIENT    2      METAL      01-Oct-20   307.3       535.7       709    ₹161,931      +74.3%      +4.7%     
  BERGEPAINT  44     CON DUR    01-Oct-19   350.7       585.6       533    ₹125,164      +67.0%      -4.5%     
  GUJGASLTD   50     OIL&GAS    02-Dec-19   206.7       344.2       959    ₹131,885      +66.5%      -2.5%     
  WIPRO       8      IT         03-Aug-20   129.8       194.7       1562   ₹101,439      +50.0%      -0.9%     
  COROMANDEL  42     MFG        03-Feb-20   585.9       786.2       362    ₹72,511       +34.2%      +0.5%     
  MPHASIS     27     IT         03-Aug-20   1,024.9     1,370.0     197    ₹67,978       +33.7%      -3.1%     
  HCLTECH     32     IT         03-Aug-20   559.9       744.7       362    ₹66,902       +33.0%      -3.3%     
  ADANIENSOL  6      ENERGY     01-Dec-20   379.0       481.5       654    ₹67,068       +27.1%      +6.6%     
  BALKRISIND  29     MFG        03-Aug-20   1,249.0     1,570.8     162    ₹52,139       +25.8%      -0.9%     
  INFY        12     IT         01-Oct-20   867.6       1,086.5     251    ₹54,949       +25.2%      -2.6%     
  SYNGENE     16     HEALTH     03-Aug-20   473.6       556.1       428    ₹35,321       +17.4%      -5.5%     
  DIVISLAB    11     HEALTH     01-Oct-20   2,973.1     3,359.8     73     ₹28,233       +13.0%      -3.7%     
  NAVINFLUOR  26     MFG        01-Oct-20   2,095.0     2,307.2     104    ₹22,068       +10.1%      -6.4%     
  APOLLOHOSP  18     HEALTH     01-Dec-20   2,431.2     2,629.1     101    ₹19,989       +8.1%       +3.7%     
  IPCALAB     85     HEALTH     03-Aug-20   927.0       924.7       218    ₹-486         -0.2% ⚠     -8.0%     
  ⚠  WAZ < 0 (momentum below universe mean): IPCALAB, PIIND

  AFTER: Invested ₹5,022,725 | Cash ₹181,710 | Total ₹5,204,435 | Positions 17/20 | Slot ₹260,222

========================================================================
  REBALANCE #38  —  01 Apr 2021
  NAV: ₹7,191,238  |  Slot: ₹359,562  |  Cash: ₹181,710
========================================================================
  [SECTOR CAP≤4] dropped: TATAELXSI

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PIIND       50     MFG        03-Jun-19   1,116.0     2,216.0     171    ₹188,092      +98.6%    668d  
  BERGEPAINT  69     CON DUR    01-Oct-19   350.7       620.9       533    ₹143,998      +77.0%    548d  
  NAVINFLUOR  47     MFG        01-Oct-20   2,095.0     2,731.4     104    ₹66,187       +30.4%    182d  
  BALKRISIND  48     MFG        03-Aug-20   1,249.0     1,615.7     162    ₹59,408       +29.4%    241d  
  COROMANDEL  89     MFG        03-Feb-20   585.9       713.7       362    ₹46,250       +21.8%    423d  
  DIVISLAB    70     HEALTH     01-Oct-20   2,973.1     3,507.8     73     ₹39,032       +18.0%    182d  
  SYNGENE     54     HEALTH     03-Aug-20   473.6       552.4       428    ₹33,753       +16.7%    241d  
  IPCALAB     96     HEALTH     03-Aug-20   927.0       927.4       218    ₹94           +0.0%     241d  

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TATACHEM    4      MFG        2.903    0.28   +259.6%   +62.4%    715.4       502    ₹359,121      +5.5%     
  DEEPAKNTR   5      MFG        2.783    0.00   +338.0%   +77.5%    1,618.8     222    ₹359,372      +7.1%     
  GRASIM      6      INFRA      2.606    0.06   +217.5%   +55.7%    1,412.7     254    ₹358,830      +5.6%     
  DIXON       8      CON DUR    2.507    0.03   +429.0%   +34.2%    3,580.2     100    ₹358,025      -5.8%     
  LAURUSLABS  9      HEALTH     1.956    0.01   +462.4%   +4.5%     359.2       1000   ₹359,215      +2.0%     
  IDFCFIRSTB  11     PVT BNK    1.716    0.20   +154.7%   +55.7%    56.9        6323   ₹359,550      -4.6%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       1,058.9     1297   ₹1,170,744    +577.7%     +31.6%    
  ADANIENT    2      METAL      01-Oct-20   307.3       1,104.0     709    ₹564,901      +259.3%     +16.5%    
  ADANIENSOL  3      ENERGY     01-Dec-20   379.0       999.2       654    ₹405,644      +163.7%     +21.3%    
  GUJGASLTD   10     OIL&GAS    02-Dec-19   206.7       524.7       959    ₹304,977      +153.9%     +5.5%     
  MPHASIS     27     IT         03-Aug-20   1,024.9     1,565.1     197    ₹106,408      +52.7%      +3.6%     
  WIPRO       33     IT         03-Aug-20   129.8       192.4       1562   ₹97,759       +48.2%      -0.1%     
  HCLTECH     32     IT         03-Aug-20   559.9       804.2       362    ₹88,463       +43.6%      +3.1%     
  INFY        25     IT         01-Oct-20   867.6       1,193.6     251    ₹81,833       +37.6%      +2.5%     
  APOLLOHOSP  19     HEALTH     01-Dec-20   2,431.2     2,856.7     101    ₹42,974       +17.5%      -1.0%     

  AFTER: Invested ₹6,954,923 | Cash ₹233,758 | Total ₹7,188,680 | Positions 15/20 | Slot ₹359,562

========================================================================
  REBALANCE #39  —  01 Jun 2021
  NAV: ₹8,583,395  |  Slot: ₹429,170  |  Cash: ₹233,758
========================================================================
  [SECTOR CAP≤4] dropped: POLYCAB

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIENT    —      METAL      01-Oct-20   307.3       1,412.2     709    ₹783,389      +359.6%   243d  
  HCLTECH     53     IT         03-Aug-20   559.9       776.1       362    ₹78,258       +38.6%    302d  
  APOLLOHOSP  —      HEALTH     01-Dec-20   2,431.2     3,198.0     101    ₹77,444       +31.5%    182d  
  IDFCFIRSTB  —      PVT BNK    01-Apr-21   56.9        57.4        6323   ₹3,151        +0.9%     61d   
  GRASIM      —      INFRA      01-Apr-21   1,412.7     1,403.1     254    ₹-2,455       -0.7%     61d   
  TATACHEM    —      MFG        01-Apr-21   715.4       648.6       502    ₹-33,511      -9.3%     61d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JSWSTEEL    3      METAL      3.491    0.45   +281.0%   +70.2%    657.4       652    ₹428,643      +0.4%     
  TATAELXSI   5      IT         2.697    0.37   +380.4%   +33.9%    3,384.3     126    ₹426,416      +1.8%     
  TATASTEEL   6      METAL      2.684    0.72   +282.6%   +51.3%    92.9        4621   ₹429,101      +0.6%     
  BSE         7      FIN SVC    2.506    0.40   +155.2%   +60.8%    97.3        4410   ₹429,135      +16.9%    
  SAIL        8      METAL      2.487    0.68   +308.4%   +69.7%    104.5       4105   ₹429,097      -0.8%     
  DALBHARAT   9      MFG        1.948    0.60   +227.2%   +25.2%    1,770.4     242    ₹428,439      +3.9%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       1,438.2     1297   ₹1,662,725    +820.5%     +10.6%    
  ADANIENSOL  2      ENERGY     01-Dec-20   379.0       1,499.4     654    ₹732,807      +295.7%     +13.2%    
  GUJGASLTD   37     OIL&GAS    02-Dec-19   206.7       517.6       959    ₹298,180      +150.5%     +3.1%     
  WIPRO       13     IT         03-Aug-20   129.8       250.7       1562   ₹188,965      +93.2%      +5.4%     
  MPHASIS     29     IT         03-Aug-20   1,024.9     1,746.1     197    ₹142,077      +70.4%      +5.5%     
  LAURUSLABS  4      HEALTH     01-Apr-21   359.2       524.5       1000   ₹165,270      +46.0%      +7.1%     
  INFY        27     IT         01-Oct-20   867.6       1,208.2     251    ₹85,503       +39.3%      +2.4%     
  DIXON       18     CON DUR    01-Apr-21   3,580.2     4,098.5     100    ₹51,822       +14.5%      +3.4%     
  DEEPAKNTR   10     MFG        01-Apr-21   1,618.8     1,730.5     222    ₹24,791       +6.9%       -0.6%     

  AFTER: Invested ₹8,270,613 | Cash ₹309,729 | Total ₹8,580,342 | Positions 15/20 | Slot ₹429,170

========================================================================
  REBALANCE #40  —  02 Aug 2021
  NAV: ₹8,649,096  |  Slot: ₹432,455  |  Cash: ₹309,729
========================================================================
  [SECTOR CAP≤4] dropped: COFORGE, POLYCAB

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIENSOL  53     ENERGY     01-Dec-20   379.0       908.8       654    ₹346,522      +139.8%   244d  
  DIXON       41     CON DUR    01-Apr-21   3,580.2     4,339.1     100    ₹75,890       +21.2%    123d  

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SRF         8      MFG        2.230    0.60   +130.0%   +37.9%    1,773.4     243    ₹430,946      +14.7%    
  ABBOTINDIA  10     HEALTH     2.103    -0.01  +37.1%    +32.3%    18,684.8    23     ₹429,751      +11.5%    
  AMBUJACEM   11     INFRA      2.047    0.59   +108.0%   +34.2%    402.3       1074   ₹432,094      +7.3%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────Model is not converging.  Current: 4921.214347689006 is not greater than 4921.222453430458. Delta is -0.008105741451799986
Model is not converging.  Current: 4921.743770037834 is not greater than 4921.747870762202. Delta is -0.004100724368072406

  ATGL        28     OIL&GAS    03-Aug-20   156.2       887.1       1297   ₹947,859      +467.7%     -5.7%     
  GUJGASLTD   7      OIL&GAS    02-Dec-19   206.7       722.7       959    ₹494,923      +249.7%     +8.3%     
  MPHASIS     9      IT         03-Aug-20   1,024.9     2,352.2     197    ₹261,468      +129.5%     +8.5%     
  WIPRO       20     IT         03-Aug-20   129.8       273.6       1562   ₹224,646      +110.8%     +3.4%     
  LAURUSLABS  5      HEALTH     01-Apr-21   359.2       644.6       1000   ₹285,433      +79.5%      +1.3%     
  INFY        16     IT         01-Oct-20   867.6       1,421.0     251    ₹138,922      +63.8%      +3.3%     
  BSE         1      FIN SVC    01-Jun-21   97.3        133.3       4410   ₹158,596      +37.0%      +10.0%    
  TATASTEEL   2      METAL      01-Jun-21   92.9        121.6       4621   ₹132,798      +30.9%      +8.9%     
  DEEPAKNTR   24     MFG        01-Apr-21   1,618.8     2,032.0     222    ₹91,735       +25.5%      +6.9%     
  DALBHARAT   6      MFG        01-Jun-21   1,770.4     2,108.0     242    ₹81,697       +19.1%      +1.3%     
  TATAELXSI   4      IT         01-Jun-21   3,384.3     4,010.4     126    ₹78,892       +18.5%      +0.6%     
  SAIL        19     METAL      01-Jun-21   104.5       120.3       4105   ₹64,789       +15.1%      +6.6%     
  JSWSTEEL    13     METAL      01-Jun-21   657.4       713.8       652    ₹36,724       +8.6%       +5.1%     

  AFTER: Invested ₹8,603,888 | Cash ₹43,672 | Total ₹8,647,560 | Positions 16/20 | Slot ₹432,455

========================================================================
  REBALANCE #41  —  01 Oct 2021
  NAV: ₹9,491,636  |  Slot: ₹474,582  |  Cash: ₹43,672
========================================================================
  [SECTOR CAP≤4] dropped: TECHM

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  GUJGASLTD   83     OIL&GAS    02-Dec-19   206.7       589.5       959    ₹367,131      +185.3%   669d  
  LAURUSLABS  74     HEALTH     01-Apr-21   359.2       609.9       1000   ₹250,665      +69.8%    183d  
  ABBOTINDIA  —      OTHER      02-Aug-21   18,684.8    20,842.3    23     ₹49,622       +11.5%    60d   
  JSWSTEEL    44     METAL      01-Jun-21   657.4       644.5       652    ₹-8,429       -2.0%     122d  
  SAIL        48     METAL      01-Jun-21   104.5       100.6       4105   ₹-16,204      -3.8%     122d  

  ENTRIES (5)
  [52w filter blocked 1: ADANIENSOL(-20.7%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IRCTC       3      PSE        3.381    0.73   +181.5%   +86.7%    725.8       653    ₹473,932      +6.9%     
  BAJAJFINSV  6      FIN SVC    2.758    1.15   +196.1%   +45.4%    1,712.9     277    ₹474,467      -0.1%     
  BAJAJHLDNG  7      FIN SVC    2.404    0.47   +99.7%    +34.8%    4,433.3     107    ₹474,362      +4.4%     
  PRESTIGE    8      REALTY     2.353    0.77   +97.5%    +69.3%    477.2       994    ₹474,343      +8.8%     
  POLYCAB     11     MFG        2.085    0.48   +184.7%   +17.1%    2,283.5     207    ₹472,685      +0.5%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        2      OIL&GAS    03-Aug-20   156.2       1,421.5     1297   ₹1,641,062    +809.8%     +3.2%     
  MPHASIS     9      IT         03-Aug-20   1,024.9     2,770.0     197    ₹343,782      +170.3%     -1.8%     
  WIPRO       24     IT         03-Aug-20   129.8       293.9       1562   ₹256,395      +126.5%     -2.4%     
  INFY        40     IT         01-Oct-20   867.6       1,450.3     251    ₹146,267      +67.2%      -2.0%     
  TATAELXSI   4      IT         01-Jun-21   3,384.3     5,491.9     126    ₹265,564      +62.3%      +6.9%     
  DEEPAKNTR   11     MFG        01-Apr-21   1,618.8     2,348.6     222    ₹162,017      +45.1%      +0.2%     
  BSE         15     FIN SVC    01-Jun-21   97.3        129.4       4410   ₹141,354      +32.9%      +1.3%     
  SRF         5      MFG        02-Aug-21   1,773.4     2,186.9     243    ₹100,475      +23.3%      +3.6%     
  TATASTEEL   12     METAL      01-Jun-21   92.9        111.9       4621   ₹88,086       +20.5%      -3.1%     
  DALBHARAT   21     MFG        01-Jun-21   1,770.4     2,064.4     242    ₹71,151       +16.6%      -1.7%     
  AMBUJACEM   18     INFRA      02-Aug-21   402.3       387.1       1074   ₹-16,380      -3.8%       -3.1%     

  AFTER: Invested ₹9,330,083 | Cash ₹158,738 | Total ₹9,488,822 | Positions 16/20 | Slot ₹474,582

========================================================================
  REBALANCE #42  —  01 Dec 2021
  NAV: ₹9,738,226  |  Slot: ₹486,911  |  Cash: ₹158,738
========================================================================
  [SECTOR CAP≤4] dropped: TECHM

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DALBHARAT   66     MFG        01-Jun-21   1,770.4     1,797.2     242    ₹6,472        +1.5%     183d  
  BAJAJFINSV  25     FIN SVC    01-Oct-21   1,712.9     1,733.3     277    ₹5,666        +1.2%     61d   
  TATASTEEL   74     METAL      01-Jun-21   92.9        93.4        4621   ₹2,290        +0.5%     183d  
  AMBUJACEM   81     INFRA      02-Aug-21   402.3       357.4       1074   ₹-48,263      -11.2%    121d  

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIENSOL  2      ENERGY     3.199    0.85   +373.4%   +19.4%    1,797.2     270    ₹485,231      -4.2%     
  ZEEL        5      MEDIA      2.823    1.08   +82.7%    +97.6%    324.0       1503   ₹486,899      +4.7%     
  DIXON       9      CON DUR    2.316    0.80   +137.1%   +23.4%    5,067.9     96     ₹486,519      -2.5%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        4      OIL&GAS    03-Aug-20   156.2       1,623.7     1297   ₹1,903,321    +939.2%     +1.9%     
  MPHASIS     15     IT         03-Aug-20   1,024.9     2,756.6     197    ₹341,141      +169.0%     -6.0%     
  WIPRO       34     IT         03-Aug-20   129.8       293.2       1562   ₹255,349      +126.0%     -1.7%     Model is not converging.  Current: 4903.860366591513 is not greater than 4903.863525954872. Delta is -0.0031593633584634517
Model is not converging.  Current: 4903.851805275479 is not greater than 4903.906876922449. Delta is -0.05507164696973632
Model is not converging.  Current: 4903.816062967805 is not greater than 4903.844336169054. Delta is -0.028273201249248814

  BSE         5      FIN SVC    01-Jun-21   97.3        175.3       4410   ₹343,950      +80.1%      +9.5%     
  INFY        35     IT         01-Oct-20   867.6       1,506.9     251    ₹160,466      +73.7%      -0.7%     
  TATAELXSI   1      IT         01-Jun-21   3,384.3     5,587.5     126    ₹277,608      +65.1%      -3.0%     
  DEEPAKNTR   24     MFG        01-Apr-21   1,618.8     2,120.9     222    ₹111,464      +31.0%      -4.0%     
  SRF         30     MFG        02-Aug-21   1,773.4     1,988.5     243    ₹52,258       +12.1%      -5.0%     
  BAJAJHLDNG  8      FIN SVC    01-Oct-21   4,433.3     4,853.5     107    ₹44,964       +9.5%       +4.8%     
  IRCTC       6      PSE        01-Oct-21   725.8       776.7       653    ₹33,250       +7.0%       -4.6%     
  POLYCAB     9      MFG        01-Oct-21   2,283.5     2,268.6     207    ₹-3,082       -0.7%       -1.5%     
  PRESTIGE    23     REALTY     01-Oct-21   477.2       439.3       994    ₹-37,675      -7.9%       -2.2%     

  AFTER: Invested ₹9,307,869 | Cash ₹428,625 | Total ₹9,736,494 | Positions 15/20 | Slot ₹486,911

========================================================================
  REBALANCE #43  —  01 Feb 2022
  NAV: ₹10,553,509  |  Slot: ₹527,675  |  Cash: ₹428,625
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  WIPRO       82     IT         03-Aug-20   129.8       266.8       1562   ₹214,067      +105.6%   547d  
  DIXON       74     CON DUR    01-Dec-21   5,067.9     4,429.2     96     ₹-61,316      -12.6%    62d   
  ZEEL        69     MEDIA      01-Dec-21   324.0       278.3       1503   ₹-68,546      -14.1%    62d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ONGC        6      OIL&GAS    2.390    0.96   +100.5%   +15.7%    131.8       4004   ₹527,640      +5.2%     
  ESCORTS     8      MFG        2.308    0.58   +51.7%    +19.6%    1,800.2     293    ₹527,469      -0.5%     
  POWERGRID   9      ENERGY     2.306    0.66   +60.2%    +17.4%    130.0       4058   ₹527,665      +1.4%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        2      OIL&GAS    03-Aug-20   156.2       1,861.9     1297   ₹2,212,267    +1091.7%    +3.3%     
  MPHASIS     29     IT         03-Aug-20   1,024.9     2,880.5     197    ₹365,547      +181.0%     +0.4%     
  BSE         1      FIN SVC    01-Jun-21   97.3        211.4       4410   ₹503,093      +117.2%     +0.2%     
  TATAELXSI   3      IT         01-Jun-21   3,384.3     7,089.3     126    ₹466,841      +109.5%     +10.5%    
  INFY        25     IT         01-Oct-20   867.6       1,557.1     251    ₹173,070      +79.5%      -1.4%     
  DEEPAKNTR   22     MFG        01-Apr-21   1,618.8     2,232.1     222    ₹136,161      +37.9%      -5.7%     
  SRF         5      MFG        02-Aug-21   1,773.4     2,413.0     243    ₹155,406      +36.1%      -0.3%     
  IRCTC       12     PSE        01-Oct-21   725.8       819.7       653    ₹61,328       +12.9%      +0.5%     
  ADANIENSOL  4      ENERGY     01-Dec-21   1,797.2     1,986.9     270    ₹51,232       +10.6%      +1.7%     
  BAJAJHLDNG  10     FIN SVC    01-Oct-21   4,433.3     4,889.1     107    ₹48,769       +10.3%      -0.4%     
  POLYCAB     7      MFG        01-Oct-21   2,283.5     2,471.0     207    ₹38,804       +8.2%       +0.5%     
  PRESTIGE    23     REALTY     01-Oct-21   477.2       480.5       994    ₹3,304        +0.7%       -0.7%     

  AFTER: Invested ₹10,447,334 | Cash ₹104,295 | Total ₹10,551,630 | Positions 15/20 | Slot ₹527,675

========================================================================
  REBALANCE #44  —  01 Apr 2022
  NAV: ₹11,817,401  |  Slot: ₹590,870  |  Cash: ₹104,295
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BSE         1      FIN SVC    01-Jun-21   97.3        297.8       4410   ₹883,949      +206.0%   304d  
  DEEPAKNTR   52     MFG        01-Apr-21   1,618.8     2,267.7     222    ₹144,061      +40.1%    365d  
  IRCTC       27     PSE        01-Oct-21   725.8       765.4       653    ₹25,873       +5.5%     182d  
  PRESTIGE    31     REALTY     01-Oct-21   477.2       501.1       994    ₹23,720       +5.0%     182d  
  ESCORTS     73     MFG        01-Feb-22   1,800.2     1,654.2     293    ₹-42,789      -8.1%     59d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HAL         5      DEFENCE    2.725    0.84   +59.8%    +28.3%    723.8       816    ₹590,624      +7.9%     
  PERSISTENT  6      IT         2.636    0.94   +163.5%   -1.4%     2,292.8     257    ₹589,242      +4.7%     
  NTPC        8      ENERGY     2.240    0.80   +46.8%    +15.9%    126.0       4687   ₹590,784      +6.4%     
  RELIANCE    12     OIL&GAS    1.959    1.09   +33.8%    +12.6%    1,202.9     491    ₹590,615      +5.3%     
  BEL         13     DEFENCE    1.945    0.86   +84.2%    +4.0%     68.5        8620   ₹590,860      +3.3%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        6      OIL&GAS    03-Aug-20   156.2       2,246.2     1297   ₹2,710,687    +1337.7%    +16.1%    
  MPHASIS     18     IT         03-Aug-20   1,024.9     3,060.9     197    ₹401,088      +198.6%     +2.7%     
  TATAELXSI   2      IT         01-Jun-21   3,384.3     8,464.4     126    ₹640,105      +150.1%     +12.8%    
  INFY        28     IT         01-Oct-20   867.6       1,672.6     251    ₹202,073      +92.8%      +2.7%     
  SRF         4      MFG        02-Aug-21   1,773.4     2,590.2     243    ₹198,471      +46.1%      +3.3%     
  ADANIENSOL  3      ENERGY     01-Dec-21   1,797.2     2,421.4     270    ₹168,561      +34.7%      +4.1%     
  BAJAJHLDNG  14     FIN SVC    01-Oct-21   4,433.3     5,052.3     107    ₹66,232       +14.0%      +6.5%     
  POWERGRID   15     ENERGY     01-Feb-22   130.0       141.2       4058   ₹45,477       +8.6%       +6.2%     Model is not converging.  Current: 4870.181150963414 is not greater than 4870.185186765504. Delta is -0.004035802089674689
Model is not converging.  Current: 4855.169068927694 is not greater than 4855.193918223496. Delta is -0.024849295802596316
Model is not converging.  Current: 4855.178222775301 is not greater than 4855.202295872994. Delta is -0.02407309769296262
Model is not converging.  Current: 4855.1340863092455 is not greater than 4855.15620620204. Delta is -0.022119892794762563

  POLYCAB     25     MFG        01-Oct-21   2,283.5     2,369.9     207    ₹17,886       +3.8%       +2.3%     
  ONGC        10     OIL&GAS    01-Feb-22   131.8       130.8       4004   ₹-3,799       -0.7%       -1.3%     

  AFTER: Invested ₹11,366,167 | Cash ₹447,729 | Total ₹11,813,896 | Positions 15/20 | Slot ₹590,870

========================================================================
  REBALANCE #45  —  01 Jun 2022
  NAV: ₹11,548,329  |  Slot: ₹577,416  |  Cash: ₹447,729
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MPHASIS     81     IT         03-Aug-20   1,024.9     2,315.3     197    ₹254,199      +125.9%   667d  
  INFY        111    IT         01-Oct-20   867.6       1,312.9     251    ₹111,792      +51.3%    608d  
  SRF         18     MFG        02-Aug-21   1,773.4     2,368.5     243    ₹144,600      +33.6%    303d  
  ADANIENSOL  —      ENERGY     01-Dec-21   1,797.2     1,957.9     270    ₹43,402       +8.9%     182d  

  ENTRIES (4)
  [52w filter blocked 1: TRIDENT(-32.9%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ITC         3      FMCG       2.982    0.75   +39.0%    +30.2%    224.5       2571   ₹577,286      +3.2%     
  M&M         8      AUTO       2.468    0.93   +27.8%    +29.7%    999.8       577    ₹576,863      +11.3%    
  VBL         10     FMCG       2.258    0.64   +64.1%    +19.0%    146.2       3949   ₹577,356      +2.9%     
  OIL         11     OIL&GAS    2.146    0.29   +93.4%    +10.3%    135.1       4273   ₹577,333      +5.8%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        5      OIL&GAS    03-Aug-20   156.2       2,322.5     1297   ₹2,809,633    +1386.5%    -2.8%     
  TATAELXSI   2      IT         01-Jun-21   3,384.3     8,173.1     126    ₹603,393      +141.5%     +5.6%     
  HAL         1      DEFENCE    01-Apr-22   723.8       899.0       816    ₹142,997      +24.2%      +9.7%     
  BEL         4      DEFENCE    01-Apr-22   68.5        78.4        8620   ₹85,154       +14.4%      +6.0%     
  POWERGRID   10     ENERGY     01-Feb-22   130.0       143.8       4058   ₹56,045       +10.6%      -0.3%     
  NTPC        7      ENERGY     01-Apr-22   126.0       138.3       4687   ₹57,239       +9.7%       +3.0%     
  BAJAJHLDNG  35     FIN SVC    01-Oct-21   4,433.3     4,726.9     107    ₹31,416       +6.6%       +0.4%     
  POLYCAB     22     MFG        01-Oct-21   2,283.5     2,426.5     207    ₹29,599       +6.3%       +0.4%     
  RELIANCE    17     OIL&GAS    01-Apr-22   1,202.9     1,192.8     491    ₹-4,970       -0.8%       +1.5%     
  ONGC        49     OIL&GAS    01-Feb-22   131.8       116.7       4004   ₹-60,566      -11.5%      -3.5%     
  PERSISTENT  37     IT         01-Apr-22   2,292.8     1,815.3     257    ₹-122,721     -20.8%      -0.8%     

  AFTER: Invested ₹11,519,596 | Cash ₹25,991 | Total ₹11,545,587 | Positions 15/20 | Slot ₹577,416

========================================================================
  REBALANCE #46  —  01 Aug 2022
  NAV: ₹13,001,654  |  Slot: ₹650,083  |  Cash: ₹25,991
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJ-AUTO

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  POWERGRID   62     ENERGY     01-Feb-22   130.0       137.4       4058   ₹29,941       +5.7%     181d  
  POLYCAB     63     MFG        01-Oct-21   2,283.5     2,294.6     207    ₹2,300        +0.5%     304d  
  RELIANCE    71     OIL&GAS    01-Apr-22   1,202.9     1,166.2     491    ₹-18,013      -3.0%     122d  
  ONGC        90     OIL&GAS    01-Feb-22   131.8       107.8       4004   ₹-95,967      -18.2%    181d  
  OIL         87     OIL&GAS    01-Jun-22   135.1       107.0       4273   ₹-120,253     -20.8%    61d   
  PERSISTENT  93     IT         01-Apr-22   2,292.8     1,780.3     257    ₹-131,714     -22.4%    122d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TVSMOTOR    2      AUTO       3.431    0.78   +62.0%    +47.7%    911.5       713    ₹649,892      +7.4%     
  SIEMENS     7      ENERGY     2.413    0.88   +42.9%    +25.4%    1,598.0     406    ₹648,783      +3.8%     
  CUMMINSIND  8      INFRA      2.302    0.76   +48.7%    +20.6%    1,157.0     561    ₹649,051      +6.1%     
  EICHERMOT   11     AUTO       1.955    0.96   +21.8%    +24.4%    2,962.8     219    ₹648,857      +2.7%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    03-Aug-20   156.2       3,213.2     1297   ₹3,964,931    +1956.6%    +13.8%    
  TATAELXSI   10     IT         01-Jun-21   3,384.3     8,324.1     126    ₹622,421      +146.0%     +5.1%     
  HAL         4      DEFENCE    01-Apr-22   723.8       963.6       816    ₹195,635      +33.1%      +8.3%     
  BEL         9      DEFENCE    01-Apr-22   68.5        90.6        8620   ₹189,811      +32.1%      +10.0%    
  VBL         6      FMCG       01-Jun-22   146.2       183.1       3949   ₹145,784      +25.3%      +7.8%     
  M&M         3      AUTO       01-Jun-22   999.8       1,193.6     577    ₹111,845      +19.4%      +8.2%     
  ITC         5      FMCG       01-Jun-22   224.5       254.0       2571   ₹75,704       +13.1%      +3.9%     
  BAJAJHLDNG  43     FIN SVC    01-Oct-21   4,433.3     4,946.0     107    ₹54,861       +11.6%      +7.1%     
  NTPC        38     ENERGY     01-Apr-22   126.0       138.0       4687   ₹55,999       +9.5%       +4.9%     

  AFTER: Invested ₹12,620,771 | Cash ₹377,800 | Total ₹12,998,571 | Positions 13/20 | Slot ₹650,083

========================================================================
  REBALANCE #47  —  03 Oct 2022
  NAV: ₹13,366,009  |  Slot: ₹668,300  |  Cash: ₹377,800
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 Model is not converging.  Current: 4863.03699362553 is not greater than 4863.066331445197. Delta is -0.02933781966657989
Model is not converging.  Current: 4862.927861341892 is not greater than 4862.939043118469. Delta is -0.011181776577359415
Model is not converging.  Current: 4873.587345819356 is not greater than 4873.609940686354. Delta is -0.022594866997678764
Model is not converging.  Current: 4873.602198663646 is not greater than 4873.6555816386335. Delta is -0.05338297498747124

  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        5      OIL&GAS    03-Aug-20   156.2       3,096.6     1297   ₹3,813,584    +1881.9%    -10.2%    
  TATAELXSI   51     IT         01-Jun-21   3,384.3     7,934.6     126    ₹573,338      +134.5%     -4.7%     
  HAL         2      DEFENCE    01-Apr-22   723.8       1,092.9     816    ₹301,159      +51.0%      -3.3%     
  VBL         3      FMCG       01-Jun-22   146.2       212.1       3949   ₹260,392      +45.1%      +0.2%     
  BAJAJHLDNG  4      FIN SVC    01-Oct-21   4,433.3     6,221.9     107    ₹191,381      +40.3%      +1.1%     
  BEL         8      DEFENCE    01-Apr-22   68.5        94.6        8620   ₹224,870      +38.1%      -5.2%     
  M&M         13     AUTO       01-Jun-22   999.8       1,206.7     577    ₹119,412      +20.7%      -1.5%     
  ITC         11     FMCG       01-Jun-22   224.5       267.9       2571   ₹111,592      +19.3%      -1.9%     
  NTPC        24     ENERGY     01-Apr-22   126.0       144.1       4687   ₹84,622       +14.3%      -1.8%     
  EICHERMOT   18     AUTO       01-Aug-22   2,962.8     3,344.6     219    ₹83,618       +12.9%      -2.5%     
  TVSMOTOR    6      AUTO       01-Aug-22   911.5       979.2       713    ₹48,287       +7.4%       -2.8%     
  SIEMENS     31     ENERGY     01-Aug-22   1,598.0     1,568.0     406    ₹-12,193      -1.9%       -4.7%     
  CUMMINSIND  29     INFRA      01-Aug-22   1,157.0     1,129.1     561    ₹-15,633      -2.4%       -1.9%     

  AFTER: Invested ₹12,988,209 | Cash ₹377,800 | Total ₹13,366,009 | Positions 13/20 | Slot ₹668,300

========================================================================
  REBALANCE #48  —  01 Dec 2022
  NAV: ₹14,537,960  |  Slot: ₹726,898  |  Cash: ₹377,800
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ATGL        21     OIL&GAS    03-Aug-20   156.2       3,607.5     1297   ₹4,476,220    +2208.9%  850d  
  TATAELXSI   116    IT         01-Jun-21   3,384.3     6,750.1     126    ₹424,100      +99.5%    548d  
  SIEMENS     50     ENERGY     01-Aug-22   1,598.0     1,607.3     406    ₹3,786        +0.6%     122d  

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  UNIONBANK   1      PSU BNK    4.613    1.16   +96.9%    +92.4%    72.2        10071  ₹726,862      +14.1%    
  BANKINDIA   3      PSU BNK    3.185    1.19   +53.1%    +60.1%    73.9        9835   ₹726,850      +9.9%     
  INDIANB     4      PSU BNK    3.127    1.19   +94.8%    +42.1%    250.0       2908   ₹726,870      +4.6%     
  AMBUJACEM   5      INFRA      2.983    0.84   +59.3%    +41.3%    570.9       1273   ₹726,780      +3.4%     
  SUNPHARMA   9      HEALTH     2.352    0.63   +37.9%    +17.2%    1,009.8     719    ₹726,061      +2.3%     
  PFC         10     FIN SVC    2.296    0.88   +29.9%    +20.9%    94.9        7658   ₹726,829      +11.0%    
  TIINDIA     11     AUTO       2.254    0.88   +77.8%    +24.5%    2,806.1     259    ₹726,789      +5.2%     
  AXISBANK    12     PVT BNK    2.096    1.04   +36.8%    +20.3%    901.5       806    ₹726,578      +3.3%     
  FEDERALBNK  13     PVT BNK    1.940    1.17   +52.3%    +13.4%    130.1       5585   ₹726,865      -0.2%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         2      DEFENCE    01-Apr-22   723.8       1,323.9     816    ₹489,703      +82.9%      +4.1%     
  VBL         7      FMCG       01-Jun-22   146.2       250.6       3949   ₹412,254      +71.4%      +9.5%     
  BEL         24     DEFENCE    01-Apr-22   68.5        100.1       8620   ₹272,099      +46.1%      -2.3%     
  BAJAJHLDNG  26     FIN SVC    01-Oct-21   4,433.3     6,038.0     107    ₹171,707      +36.2%      -2.6%     
  ITC         11     FMCG       01-Jun-22   224.5       280.5       2571   ₹143,764      +24.9%      -0.9%     
  M&M         29     AUTO       01-Jun-22   999.8       1,247.3     577    ₹142,837      +24.8%      +1.6%     
  NTPC        25     ENERGY     01-Apr-22   126.0       154.8       4687   ₹134,535      +22.8%      +1.2%     
  CUMMINSIND  8      INFRA      01-Aug-22   1,157.0     1,372.2     561    ₹120,725      +18.6%      +5.9%     
  TVSMOTOR    20     AUTO       01-Aug-22   911.5       1,032.8     713    ₹86,510       +13.3%      -2.1%     
  EICHERMOT   31     AUTO       01-Aug-22   2,962.8     3,319.6     219    ₹78,140       +12.0%      -1.4%     

  AFTER: Invested ₹14,518,697 | Cash ₹11,497 | Total ₹14,530,194 | Positions 19/20 | Slot ₹726,898

========================================================================
  REBALANCE #49  —  01 Feb 2023
  NAV: ₹13,693,399  |  Slot: ₹684,670  |  Cash: ₹11,497
========================================================================

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJAJHLDNG  67     FIN SVC    01-Oct-21   4,433.3     5,815.5     107    ₹147,895      +31.2%    488d  
  BEL         64     DEFENCE    01-Apr-22   68.5        87.4        8620   ₹162,727      +27.5%    306d  
  EICHERMOT   66     AUTO       01-Aug-22   2,962.8     3,189.9     219    ₹49,726       +7.7%     184d  
  INDIANB     2      PSU BNK    01-Dec-22   250.0       265.1       2908   ₹43,973       +6.0%     62d   
  BANKINDIA   9      PSU BNK    01-Dec-22   73.9        70.9        9835   ₹-29,739      -4.1%     62d   
  AXISBANK    59     PVT BNK    01-Dec-22   901.5       855.0       806    ₹-37,460      -5.2%     62d   
  UNIONBANK   3      PSU BNK    01-Dec-22   72.2        65.5        10071  ₹-67,335      -9.3%     62d   
  AMBUJACEM   131    INFRA      01-Dec-22   570.9       328.3       1273   ₹-308,814     -42.5%    62d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RECLTD      2      FIN SVC    2.895    0.94   +33.0%    +19.3%    97.8        6999   ₹684,649      -2.1%     
  HINDZINC    6      METAL      2.686    0.68   +24.8%    +25.9%    268.2       2552   ₹684,495      +0.1%     
  IOC         9      OIL&GAS    2.106    0.67   +4.3%     +16.3%    66.3        10331  ₹684,655      -1.4%     Model is not converging.  Current: 4869.96287967489 is not greater than 4869.993643348826. Delta is -0.03076367393532564
Model is not converging.  Current: 4870.038183051818 is not greater than 4870.081545482147. Delta is -0.0433624303286706
Model is not converging.  Current: 4869.903201821128 is not greater than 4869.934365394198. Delta is -0.031163573069534323
Model is not converging.  Current: 4872.528036903997 is not greater than 4872.543998665489. Delta is -0.015961761491780635
Model is not converging.  Current: 4872.543155418656 is not greater than 4872.591278430337. Delta is -0.04812301168112754
Model is not converging.  Current: 4872.403303090647 is not greater than 4872.436066227709. Delta is -0.03276313706192013

  ABBOTINDIA  10     HEALTH     2.090    0.35   +29.4%    +5.6%     19,849.0    34     ₹674,867      -3.6%     
  LINDEINDIA  11     MFG        1.934    0.75   +27.1%    +11.9%    3,352.6     204    ₹683,926      +0.3%     
  INDIGO      12     CONSUMP    1.875    1.15   +11.9%    +15.5%    2,081.4     328    ₹682,693      +0.2%     
  HCLTECH     14     IT         1.859    0.91   +10.1%    +9.3%     990.4       691    ₹684,353      +3.1%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VBL         7      FMCG       01-Jun-22   146.2       232.5       3949   ₹340,746      +59.0%      -4.7%     
  HAL         14     DEFENCE    01-Apr-22   723.8       1,136.0     816    ₹336,382      +57.0%      -5.1%     
  ITC         1      FMCG       01-Jun-22   224.5       298.5       2571   ₹190,164      +32.9%      +6.2%     
  M&M         12     AUTO       01-Jun-22   999.8       1,303.8     577    ₹175,443      +30.4%      +2.7%     
  NTPC        41     ENERGY     01-Apr-22   126.0       152.8       4687   ₹125,269      +21.2%      +1.1%     
  CUMMINSIND  6      INFRA      01-Aug-22   1,157.0     1,361.3     561    ₹114,649      +17.7%      -0.0%     
  TVSMOTOR    22     AUTO       01-Aug-22   911.5       1,001.7     713    ₹64,312       +9.9%       -0.2%     
  FEDERALBNK  35     PVT BNK    01-Dec-22   130.1       128.8       5585   ₹-7,395       -1.0%       -2.4%     
  PFC         8      FIN SVC    01-Dec-22   94.9        93.4        7658   ₹-11,283      -1.6%       -6.0%     
  SUNPHARMA   38     HEALTH     01-Dec-22   1,009.8     979.4       719    ₹-21,890      -3.0%       -1.5%     
  TIINDIA     36     AUTO       01-Dec-22   2,806.1     2,605.8     259    ₹-51,890      -7.1%       -1.5%     

  AFTER: Invested ₹13,152,547 | Cash ₹535,176 | Total ₹13,687,723 | Positions 18/20 | Slot ₹684,670

========================================================================
  REBALANCE #50  —  03 Apr 2023
  NAV: ₹14,355,608  |  Slot: ₹717,780  |  Cash: ₹535,176
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LINDEINDIA  —      OTHER      01-Feb-23   3,352.6     4,026.1     204    ₹137,404      +20.1%    61d   
  IOC         56     OIL&GAS    01-Feb-23   66.3        64.4        10331  ₹-19,733      -2.9%     61d   
  HCLTECH     48     IT         01-Feb-23   990.4       960.7       691    ₹-20,523      -3.0%     61d   
  SUNPHARMA   57     HEALTH     01-Dec-22   1,009.8     951.9       719    ₹-41,664      -5.7%     123d  
  INDIGO      84     CONSUMP    01-Feb-23   2,081.4     1,896.8     328    ₹-60,545      -8.9%     61d   
  TIINDIA     43     AUTO       01-Dec-22   2,806.1     2,553.0     259    ₹-65,554      -9.0%     123d  

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SIEMENS     2      ENERGY     2.863    0.83   +46.6%    +19.3%    1,953.4     367    ₹716,896      +2.0%     
  ZYDUSLIFE   6      HEALTH     2.609    0.63   +39.3%    +17.3%    481.1       1491   ₹717,359      +2.7%     
  BOSCHLTD    7      AUTO       2.504    1.01   +41.1%    +13.8%    18,666.5    38     ₹709,327      +5.1%     
  INDIANB     9      PSU BNK    2.334    1.20   +104.2%   +1.7%     262.7       2732   ₹717,758      +4.0%     
  GODREJCP    12     FMCG       2.158    0.60   +38.7%    +10.3%    921.3       779    ₹717,657      +2.0%     
  PETRONET    13     OIL&GAS    2.049    0.48   +27.7%    +8.5%     206.3       3479   ₹717,640      +2.6%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VBL         3      FMCG       01-Jun-22   146.2       280.4       3949   ₹529,869      +91.8%      +5.0%     
  HAL         5      DEFENCE    01-Apr-22   723.8       1,312.6     816    ₹480,498      +81.4%      +1.9%     
  ITC         1      FMCG       01-Jun-22   224.5       318.1       2571   ₹240,512      +41.7%      -0.3%     
  CUMMINSIND  4      INFRA      01-Aug-22   1,157.0     1,543.4     561    ₹216,783      +33.4%      -1.1%     
  NTPC        11     ENERGY     01-Apr-22   126.0       164.0       4687   ₹177,692      +30.1%      +1.8%     
  TVSMOTOR    10     AUTO       01-Aug-22   911.5       1,072.3     713    ₹114,635      +17.6%      +2.1%     
  PFC         8      FIN SVC    01-Dec-22   94.9        108.8       7658   ₹106,622      +14.7%      +1.5%     
  M&M         28     AUTO       01-Jun-22   999.8       1,128.2     577    ₹74,092       +12.8%      -1.4%     
  ABBOTINDIA  19     HEALTH     01-Feb-23   19,849.0    21,340.6    34     ₹50,715       +7.5%       +4.7%     
  RECLTD      17     FIN SVC    01-Feb-23   97.8        100.2       6999   ₹16,824       +2.5%       +0.3%     
  FEDERALBNK  41     PVT BNK    01-Dec-22   130.1       130.6       5585   ₹2,465        +0.3%       +2.8%     
  HINDZINC    31     METAL      01-Feb-23   268.2       261.3       2552   ₹-17,699      -2.6%       +1.9%     

  AFTER: Invested ₹13,999,206 | Cash ₹351,300 | Total ₹14,350,506 | Positions 18/20 | Slot ₹717,780

========================================================================
  REBALANCE #51  —  01 Jun 2023
  NAV: ₹15,424,272  |  Slot: ₹771,214  |  Cash: ₹351,300
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  NTPC        95     ENERGY     01-Apr-22   126.0       160.8       4687   ₹162,997      +27.6%    426d  
  M&M         63     AUTO       01-Jun-22   999.8       1,271.9     577    ₹156,998      +27.2%    365d  
  GODREJCP    —      FMCG       03-Apr-23   921.3       1,006.9     779    ₹66,743       +9.3%     59d   
  ABBOTINDIA  64     HEALTH     01-Feb-23   19,849.0    20,714.1    34     ₹29,413       +4.4%     120d  
  ZYDUSLIFE   —      HEALTH     03-Apr-23   481.1       501.6       1491   ₹30,594       +4.3%     59d   
  HINDZINC    81     METAL      01-Feb-23   268.2       267.8       2552   ₹-1,041       -0.2%     120d  
  BOSCHLTD    60     AUTO       03-Apr-23   18,666.5    17,867.1    38     ₹-30,376      -4.3%     59d   
  PETRONET    —      OIL&GAS    03-Apr-23   206.3       197.0       3479   ₹-32,383      -4.5%     59d   
  INDIANB     65     PSU BNK    03-Apr-23   262.7       248.9       2732   ₹-37,725      -5.3%     59d   
  FEDERALBNK  80     PVT BNK    01-Dec-22   130.1       123.1       5585   ₹-39,438      -5.4%     182d  Model is not converging.  Current: 4874.39260648874 is not greater than 4874.414555525925. Delta is -0.021949037184640474
Model is not converging.  Current: 4874.411277171396 is not greater than 4874.462520651905. Delta is -0.05124348050867411
Model is not converging.  Current: 4874.33746804901 is not greater than 4874.356897366289. Delta is -0.01942931727899122
Model is not converging.  Current: 4879.631053822377 is not greater than 4879.6609558072305. Delta is -0.029901984853495378
Model is not converging.  Current: 4879.704486500925 is not greater than 4879.70459790822. Delta is -0.00011140729566250229
Model is not converging.  Current: 4879.536099030269 is not greater than 4879.536916752428. Delta is -0.0008177221588994144


  ENTRIES (9)
  [52w filter blocked 1: ADANIGREEN(-61.4%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  CHOLAFIN    4      FIN SVC    2.758    1.11   +63.8%    +36.8%    1,039.4     742    ₹771,206      +3.0%     
  SYNGENE     6      HEALTH     2.421    0.43   +37.7%    +28.3%    720.6       1070   ₹771,087      +3.7%     
  APOLLOTYRE  7      AUTO       2.374    1.12   +86.4%    +22.7%    375.8       2052   ₹771,134      +4.2%     
  TORNTPHARM  8      HEALTH     2.192    0.39   +24.5%    +19.9%    1,716.8     449    ₹770,857      +4.8%     
  NESTLEIND   11     FMCG       2.105    0.49   +25.3%    +17.7%    1,062.6     725    ₹770,352      +1.6%     
  BAJAJ-AUTO  13     AUTO       1.947    0.60   +24.9%    +20.6%    4,298.5     179    ₹769,426      +2.5%     
  BEL         15     DEFENCE    1.812    0.93   +52.6%    +19.2%    109.9       7016   ₹771,213      +3.8%     
  MAXHEALTH   16     HEALTH     1.793    0.25   +45.8%    +23.5%    530.5       1453   ₹770,867      +2.0%     
  TRENT       17     CONSUMP    1.776    1.09   +49.1%    +19.8%    1,558.5     494    ₹769,890      +4.3%     

  HOLDS (8)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VBL         4      FMCG       01-Jun-22   146.2       335.0       3949   ₹745,705      +129.2%     +5.7%     
  HAL         13     DEFENCE    01-Apr-22   723.8       1,490.9     816    ₹625,958      +106.0%     +1.9%     
  ITC         1      FMCG       01-Jun-22   224.5       377.4       2571   ₹393,056      +68.1%      +3.5%     
  CUMMINSIND  16     INFRA      01-Aug-22   1,157.0     1,683.2     561    ₹295,235      +45.5%      +4.5%     
  TVSMOTOR    10     AUTO       01-Aug-22   911.5       1,257.5     713    ₹246,678      +38.0%      +1.9%     
  PFC         5      FIN SVC    01-Dec-22   94.9        128.6       7658   ₹257,939      +35.5%      +7.0%     
  RECLTD      3      FIN SVC    01-Feb-23   97.8        120.3       6999   ₹157,598      +23.0%      +5.1%     
  SIEMENS     37     ENERGY     03-Apr-23   1,953.4     2,059.2     367    ₹38,820       +5.4%       -0.9%     

  AFTER: Invested ₹14,869,606 | Cash ₹546,430 | Total ₹15,416,037 | Positions 17/20 | Slot ₹771,214

========================================================================
  REBALANCE #52  —  01 Aug 2023
  NAV: ₹17,457,268  |  Slot: ₹872,863  |  Cash: ₹546,430
========================================================================
  [SECTOR CAP≤4] dropped: SUNPHARMA

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VBL         49     FMCG       01-Jun-22   146.2       317.7       3949   ₹677,085      +117.3%   426d  
  SIEMENS     53     ENERGY     03-Apr-23   1,953.4     2,267.0     367    ₹115,080      +16.1%    120d  
  BAJAJ-AUTO  46     AUTO       01-Jun-23   4,298.5     4,697.4     179    ₹71,413       +9.3%     61d   
  NESTLEIND   85     FMCG       01-Jun-23   1,062.6     1,097.5     725    ₹25,335       +3.3%     61d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  POLYCAB     3      MFG        3.148    0.82   +109.5%   +41.5%    4,561.7     191    ₹871,283      +7.3%     
  TATACOMM    4      CONSUMP    2.528    0.90   +69.3%    +44.3%    1,705.7     511    ₹871,598      +6.9%     
  NTPC        5      ENERGY     2.500    0.70   +56.9%    +27.7%    207.6       4204   ₹872,847      +12.7%    
  COLPAL      7      FMCG       2.167    0.36   +31.3%    +28.3%    1,880.1     464    ₹872,351      +7.0%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         8      DEFENCE    01-Apr-22   723.8       1,870.9     816    ₹936,013      +158.5%     +0.8%     
  PFC         1      FIN SVC    01-Dec-22   94.9        185.9       7658   ₹696,466      +95.8%      +10.0%    
  RECLTD      2      FIN SVC    01-Feb-23   97.8        175.8       6999   ₹545,807      +79.7%      +14.9%    
  ITC         15     FMCG       01-Jun-22   224.5       399.0       2571   ₹448,447      +77.7%      -0.7%     
  CUMMINSIND  12     INFRA      01-Aug-22   1,157.0     1,865.6     561    ₹397,538      +61.2%      +1.2%     
  TVSMOTOR    24     AUTO       01-Aug-22   911.5       1,353.9     713    ₹315,466      +48.5%      +1.7%     
  BEL         23     DEFENCE    01-Jun-23   109.9       127.0       7016   ₹119,883      +15.5%      +3.1%     
  TORNTPHARM  14     HEALTH     01-Jun-23   1,716.8     1,927.0     449    ₹94,362       +12.2%      +2.0%     
  SYNGENE     25     HEALTH     01-Jun-23   720.6       800.9       1070   ₹85,890       +11.1%      +3.4%     
  APOLLOTYRE  10     AUTO       01-Jun-23   375.8       415.0       2052   ₹80,520       +10.4%      +2.1%     
  TRENT       33     CONSUMP    01-Jun-23   1,558.5     1,703.3     494    ₹71,549       +9.3%       -0.1%     
  CHOLAFIN    13     FIN SVC    01-Jun-23   1,039.4     1,126.2     742    ₹64,452       +8.4%       -0.7%     
  MAXHEALTH   35     HEALTH     01-Jun-23   530.5       569.7       1453   ₹56,916       +7.4%       -4.7%     

  AFTER: Invested ₹16,675,975 | Cash ₹777,151 | Total ₹17,453,126 | Positions 17/20 | Slot ₹872,863

========================================================================
  REBALANCE #53  —  03 Oct 2023
  NAV: ₹18,771,121  |  Slot: ₹938,556  |  Cash: ₹777,151
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PFC         4      FIN SVC    01-Dec-22   94.9        225.2       7658   ₹997,992      +137.3%   306d  
  ITC         75     FMCG       01-Jun-22   224.5       377.5       2571   ₹393,166      +68.1%    489d  
  CUMMINSIND  91     INFRA      01-Aug-22   1,157.0     1,624.9     561    ₹262,536      +40.4%    428d  
  MAXHEALTH   79     HEALTH     01-Jun-23   530.5       590.5       1453   ₹87,123       +11.3%    124d  
  TORNTPHARM  76     HEALTH     01-Jun-23   1,716.8     1,826.7     449    ₹49,323       +6.4%     124d  
  APOLLOTYRE  85     AUTO       01-Jun-23   375.8       361.3       2052   ₹-29,812      -3.9%     124d  Model is not converging.  Current: 4878.565026841758 is not greater than 4878.599473656693. Delta is -0.034446814935108705
Model is not converging.  Current: 4878.645666805575 is not greater than 4878.697006136385. Delta is -0.051339330810151296
Model is not converging.  Current: 4878.572614406181 is not greater than 4878.591213026899. Delta is -0.018598620717966696
Model is not converging.  Current: 4866.4495683381065 is not greater than 4866.450272298548. Delta is -0.0007039604415695067
Model is not converging.  Current: 4866.391696818763 is not greater than 4866.432300854378. Delta is -0.040604035614705936
Model is not converging.  Current: 4866.391258547923 is not greater than 4866.399508465913. Delta is -0.008249917989815003


  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RVNL        3      PSE        3.481    1.14   +426.1%   +41.9%    170.4       5507   ₹938,408      +6.7%     
  LT          4      INFRA      2.898    0.83   +67.9%    +26.7%    2,991.9     313    ₹936,459      +5.9%     
  INDIANB     5      PSU BNK    2.753    1.10   +144.4%   +48.2%    412.0       2278   ₹938,545      +7.3%     
  COALINDIA   7      ENERGY     2.353    0.77   +50.3%    +28.2%    242.6       3868   ₹938,352      +5.4%     
  SUNTV       8      MEDIA      2.338    0.76   +31.4%    +43.0%    586.4       1600   ₹938,275      +4.2%     
  OIL         9      OIL&GAS    2.070    0.42   +78.2%    +22.4%    180.7       5193   ₹938,495      +3.8%     
  PGHH        10     FMCG       1.897    0.31   +23.2%    +24.1%    16,905.8    55     ₹929,818      +2.8%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RECLTD      1      FIN SVC    01-Feb-23   97.8        260.0       6999   ₹1,134,923    +165.8%     +11.0%    
  HAL         48     DEFENCE    01-Apr-22   723.8       1,901.3     816    ₹960,820      +162.7%     +0.3%     
  TVSMOTOR    25     AUTO       01-Aug-22   911.5       1,512.4     713    ₹428,445      +65.9%      +2.7%     
  TRENT       24     CONSUMP    01-Jun-23   1,558.5     2,053.9     494    ₹244,755      +31.8%      -0.3%     
  BEL         46     DEFENCE    01-Jun-23   109.9       136.2       7016   ₹184,019      +23.9%      +1.8%     
  CHOLAFIN    30     FIN SVC    01-Jun-23   1,039.4     1,249.1     742    ₹155,635      +20.2%      +6.0%     
  POLYCAB     5      MFG        01-Aug-23   4,561.7     5,293.7     191    ₹139,823      +16.0%      +3.5%     
  SYNGENE     50     HEALTH     01-Jun-23   720.6       805.4       1070   ₹90,673       +11.8%      +1.7%     
  NTPC        10     ENERGY     01-Aug-23   207.6       225.5       4204   ₹75,306       +8.6%       +2.0%     
  TATACOMM    21     CONSUMP    01-Aug-23   1,705.7     1,840.2     511    ₹68,729       +7.9%       +1.5%     
  COLPAL      22     FMCG       01-Aug-23   1,880.1     1,853.9     464    ₹-12,152      -1.4%       -0.8%     

  AFTER: Invested ₹18,525,967 | Cash ₹237,366 | Total ₹18,763,334 | Positions 18/20 | Slot ₹938,556

========================================================================
  REBALANCE #54  —  01 Dec 2023
  NAV: ₹20,617,678  |  Slot: ₹1,030,884  |  Cash: ₹237,366
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  RECLTD      2      FIN SVC    01-Feb-23   97.8        336.8       6999   ₹1,672,830    +244.3%   303d  
  SYNGENE     88     HEALTH     01-Jun-23   720.6       741.8       1070   ₹22,691       +2.9%     183d  
  RVNL        25     PSE        03-Oct-23   170.4       163.0       5507   ₹-40,800      -4.3%     59d   
  TATACOMM    96     CONSUMP    01-Aug-23   1,705.7     1,606.1     511    ₹-50,879      -5.8%     122d  
  INDIANB     73     PSU BNK    03-Oct-23   412.0       374.5       2278   ₹-85,390      -9.1%     59d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BAJAJ-AUTO  4      AUTO       3.054    0.63   +72.1%    +29.6%    5,767.9     178    ₹1,026,686    +5.8%     
  TORNTPOWER  6      ENERGY     2.487    0.93   +86.0%    +43.1%    914.4       1127   ₹1,030,517    +14.4%    
  HEROMOTOCO  10     AUTO       2.157    0.89   +46.0%    +25.8%    3,443.0     299    ₹1,029,448    +9.6%     
  PERSISTENT  11     IT         2.104    1.12   +67.2%    +26.0%    3,167.3     325    ₹1,029,358    +1.8%     
  BOSCHLTD    12     AUTO       2.095    0.67   +36.2%    +18.5%    21,526.2    47     ₹1,011,731    +6.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         13     DEFENCE    01-Apr-22   723.8       2,392.1     816    ₹1,361,315    +230.5%     +13.8%    
  TVSMOTOR    4      AUTO       01-Aug-22   911.5       1,887.8     713    ₹696,127      +107.1%     +9.7%     
  TRENT       5      CONSUMP    01-Jun-23   1,558.5     2,800.7     494    ₹613,635      +79.7%      +10.1%    
  BEL         52     DEFENCE    01-Jun-23   109.9       144.2       7016   ₹240,633      +31.2%      +4.4%     
  COALINDIA   3      ENERGY     03-Oct-23   242.6       301.3       3868   ₹227,111      +24.2%      +6.1%     
  NTPC        8      ENERGY     01-Aug-23   207.6       253.9       4204   ₹194,600      +22.3%      +7.3%     
  COLPAL      16     FMCG       01-Aug-23   1,880.1     2,158.3     464    ₹129,123      +14.8%      +5.5%     
  POLYCAB     23     MFG        01-Aug-23   4,561.7     5,157.1     191    ₹113,723      +13.1%      +0.5%     
  SUNTV       32     MEDIA      03-Oct-23   586.4       639.8       1600   ₹85,344       +9.1%       +2.0%     
  CHOLAFIN    61     FIN SVC    01-Jun-23   1,039.4     1,124.0     742    ₹62,788       +8.1%       -0.4%     
  OIL         28     OIL&GAS    03-Oct-23   180.7       192.7       5193   ₹61,975       +6.6%       +1.5%     
  LT          12     INFRA      03-Oct-23   2,991.9     3,106.2     313    ₹35,773       +3.8%       +4.3%     
  PGHH        49     FMCG       03-Oct-23   16,905.8    16,628.0    55     ₹-15,279      -1.6%       -1.2%     

  AFTER: Invested ₹19,785,313 | Cash ₹826,277 | Total ₹20,611,590 | Positions 18/20 | Slot ₹1,030,884

========================================================================
  REBALANCE #55  —  01 Feb 2024
  NAV: ₹23,235,317  |  Slot: ₹1,161,766  |  Cash: ₹826,277
========================================================================
  [SECTOR CAP≤4] dropped: TATAPOWER, APOLLOTYRE

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LT          51     INFRA      03-Oct-23   2,991.9     3,308.0     313    ₹98,956       +10.6%    121d  
  CHOLAFIN    80     FIN SVC    01-Jun-23   1,039.4     1,140.8     742    ₹75,244       +9.8%     245d  
  SUNTV       85     MEDIA      03-Oct-23   586.4       621.7       1600   ₹56,477       +6.0%     121d  Model is not converging.  Current: 4867.360817723894 is not greater than 4867.369806655304. Delta is -0.008988931410385703
Model is not converging.  Current: 4867.406849944024 is not greater than 4867.416536158147. Delta is -0.009686214123576065
Model is not converging.  Current: 4867.309030598606 is not greater than 4867.323294820143. Delta is -0.014264221536905097
Model is not converging.  Current: 4866.987035609947 is not greater than 4866.995315385047. Delta is -0.008279775100163533
Model is not converging.  Current: 4867.060686163778 is not greater than 4867.080689667261. Delta is -0.02000350348316715

  PGHH        90     FMCG       03-Oct-23   16,905.8    16,275.4    55     ₹-34,671      -3.7%     121d  
  POLYCAB     106    MFG        01-Aug-23   4,561.7     4,200.9     191    ₹-68,919      -7.9%     184d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IRFC        1      FIN SVC    4.124    1.11   +423.4%   +136.1%   164.1       7078   ₹1,161,698    +18.3%    
  NHPC        5      ENERGY     2.833    0.83   +124.2%   +79.2%    85.8        13541  ₹1,161,685    +17.7%    
  RVNL        6      PSE        2.644    1.17   +297.9%   +90.3%    293.9       3952   ₹1,161,620    +16.8%    
  MRF         11     MFG        2.093    0.44   +59.4%    +30.7%    142,044.1   8      ₹1,136,353    +4.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         6      DEFENCE    01-Apr-22   723.8       2,912.1     816    ₹1,785,672    +302.3%     +2.0%     
  TVSMOTOR    16     AUTO       01-Aug-22   911.5       1,972.3     713    ₹756,352      +116.4%     +0.1%     
  TRENT       4      CONSUMP    01-Jun-23   1,558.5     3,095.0     494    ₹759,049      +98.6%      -0.6%     
  BEL         22     DEFENCE    01-Jun-23   109.9       179.4       7016   ₹487,332      +63.2%      -0.8%     
  OIL         21     OIL&GAS    03-Oct-23   180.7       271.3       5193   ₹470,201      +50.1%      +9.7%     
  NTPC        11     ENERGY     01-Aug-23   207.6       304.0       4204   ₹405,152      +46.4%      +3.3%     
  COALINDIA   15     ENERGY     03-Oct-23   242.6       353.5       3868   ₹429,003      +45.7%      +4.9%     
  PERSISTENT  20     IT         01-Dec-23   3,167.3     4,099.4     325    ₹302,936      +29.4%      +5.1%     
  BAJAJ-AUTO  5      AUTO       01-Dec-23   5,767.9     7,303.5     178    ₹273,345      +26.6%      +5.9%     
  COLPAL      24     FMCG       01-Aug-23   1,880.1     2,370.6     464    ₹227,611      +26.1%      +0.8%     
  HEROMOTOCO  13     AUTO       01-Dec-23   3,443.0     4,200.0     299    ₹226,342      +22.0%      +5.2%     
  TORNTPOWER  17     ENERGY     01-Dec-23   914.4       1,009.7     1127   ₹107,385      +10.4%      +5.3%     
  BOSCHLTD    38     AUTO       01-Dec-23   21,526.2    23,082.1    47     ₹73,126       +7.2%       +3.1%     

  AFTER: Invested ₹22,456,267 | Cash ₹773,562 | Total ₹23,229,829 | Positions 17/20 | Slot ₹1,161,766

========================================================================
  REBALANCE #56  —  01 Apr 2024
  NAV: ₹25,729,000  |  Slot: ₹1,286,450  |  Cash: ₹773,562
========================================================================
  [SECTOR CAP≤4] dropped: MARUTI

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BEL         24     DEFENCE    01-Jun-23   109.9       208.0       7016   ₹688,116      +89.2%    305d  
  COALINDIA   25     ENERGY     03-Oct-23   242.6       388.7       3868   ₹564,973      +60.2%    181d  
  NTPC        34     ENERGY     01-Aug-23   207.6       326.4       4204   ₹499,187      +57.2%    244d  
  NHPC        28     ENERGY     01-Feb-24   85.8        86.2        13541  ₹6,209        +0.5%     60d   
  RVNL        13     PSE        01-Feb-24   293.9       258.6       3952   ₹-139,761     -12.0%    60d   
  IRFC        2      FIN SVC    01-Feb-24   164.1       139.9       7078   ₹-171,622     -14.8%    60d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SUNPHARMA   2      HEALTH     2.815    0.47   +71.1%    +30.8%    1,598.7     804    ₹1,285,346    +3.3%     
  CUMMINSIND  5      INFRA      2.732    0.77   +84.2%    +52.5%    2,927.3     439    ₹1,285,099    +6.4%     
  KALYANKJIL  9      CON DUR    2.226    0.53   +260.5%   +20.9%    423.7       3036   ₹1,286,376    +8.3%     
  DIXON       10     CON DUR    2.118    0.79   +160.8%   +17.5%    7,586.0     169    ₹1,282,027    +7.3%     
  UNIONBANK   11     PSU BNK    2.116    1.16   +155.2%   +31.7%    143.5       8965   ₹1,286,358    +4.9%     
  SIEMENS     12     ENERGY     2.115    0.74   +66.6%    +37.9%    3,195.5     402    ₹1,284,577    +11.4%    

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         14     DEFENCE    01-Apr-22   723.8       3,330.6     816    ₹2,127,156    +360.2%     +6.8%     
  TRENT       7      CONSUMP    01-Jun-23   1,558.5     3,877.1     494    ₹1,145,374    +148.8%     -0.9%     
  TVSMOTOR    37     AUTO       01-Aug-22   911.5       2,122.8     713    ₹863,697      +132.9%     +1.1%     
  OIL         10     OIL&GAS    03-Oct-23   180.7       373.1       5193   ₹999,108      +106.5%     +2.3%     
  TORNTPOWER  6      ENERGY     01-Dec-23   914.4       1,379.4     1127   ₹524,014      +50.8%      +13.6%    
  BAJAJ-AUTO  1      AUTO       01-Dec-23   5,767.9     8,626.2     178    ₹508,771      +49.6%      +4.3%     
  BOSCHLTD    5      AUTO       01-Dec-23   21,526.2    29,732.3    47     ₹385,687      +38.1%      +2.7%     
  COLPAL      36     FMCG       01-Aug-23   1,880.1     2,572.1     464    ₹321,113      +36.8%      +2.5%     
  HEROMOTOCO  27     AUTO       01-Dec-23   3,443.0     4,380.0     299    ₹280,175      +27.2%      +1.7%     
  PERSISTENT  55     IT         01-Dec-23   3,167.3     3,949.9     325    ₹254,363      +24.7%      -2.3%     
  MRF         42     MFG        01-Feb-24   142,044.1   135,309.0   8      ₹-53,881      -4.7%       -1.2%     

  AFTER: Invested ₹25,150,704 | Cash ₹569,142 | Total ₹25,719,845 | Positions 17/20 | Slot ₹1,286,450

========================================================================
  REBALANCE #57  —  03 Jun 2024
  NAV: ₹28,934,485  |  Slot: ₹1,446,724  |  Cash: ₹569,142
========================================================================
  [SECTOR CAP≤4] dropped: ASHOKLEY

  [REGIME OFF] Nifty 200 HMM bear-state prob 0.84 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         1      DEFENCE    01-Apr-22   723.8       5,160.9     816    ₹3,620,703    +613.0%     +13.2%    Model is not converging.  Current: 4863.132941475247 is not greater than 4863.170030410879. Delta is -0.03708893563180027
Model is not converging.  Current: 4861.6735534484105 is not greater than 4861.6908538024345. Delta is -0.01730035402397334
Model is not converging.  Current: 4863.092648618837 is not greater than 4863.112232378716. Delta is -0.019583759878514684
Model is not converging.  Current: 4852.957216595127 is not greater than 4852.957477832716. Delta is -0.00026123758925677976
Model is not converging.  Current: 4853.075313426996 is not greater than 4853.097069104629. Delta is -0.02175567763242725
Model is not converging.  Current: 4852.850669637871 is not greater than 4852.894099887017. Delta is -0.04343024914578564

  TRENT       11     CONSUMP    01-Jun-23   1,558.5     4,652.6     494    ₹1,528,512    +198.5%     +2.2%     
  TVSMOTOR    58     AUTO       01-Aug-22   911.5       2,232.9     713    ₹942,168      +145.0%     +3.9%     
  OIL         —      OIL&GAS    03-Oct-23   180.7       422.6       5193   ₹1,255,980    +133.8%     +4.4%     
  TORNTPOWER  —      ENERGY     01-Dec-23   914.4       1,469.3     1127   ₹625,372      +60.7%      +6.0%     
  BAJAJ-AUTO  31     AUTO       01-Dec-23   5,767.9     8,906.0     178    ₹558,576      +54.4%      +3.9%     
  HEROMOTOCO  38     AUTO       01-Dec-23   3,443.0     4,829.0     299    ₹414,432      +40.3%      +3.5%     
  COLPAL      55     FMCG       01-Aug-23   1,880.1     2,578.9     464    ₹324,235      +37.2%      -0.3%     
  BOSCHLTD    60     AUTO       01-Dec-23   21,526.2    29,444.7    47     ₹372,168      +36.8%      -2.0%     
  SIEMENS     3      ENERGY     01-Apr-24   3,195.5     4,254.8     402    ₹425,872      +33.2%      +6.5%     
  DIXON       6      CON DUR    01-Apr-24   7,586.0     9,878.1     169    ₹387,367      +30.2%      +10.6%    
  CUMMINSIND  13     INFRA      01-Apr-24   2,927.3     3,618.2     439    ₹303,304      +23.6%      +2.9%     
  UNIONBANK   43     PSU BNK    01-Apr-24   143.5       155.6       8965   ₹108,735      +8.5%       +11.6%    
  PERSISTENT  138    IT         01-Dec-23   3,167.3     3,385.8     325    ₹71,023       +6.9% ⚠     -3.1%     
  KALYANKJIL  32     CON DUR    01-Apr-24   423.7       388.9       3036   ₹-105,562     -8.2%       -2.3%     
  SUNPHARMA   98     HEALTH     01-Apr-24   1,598.7     1,425.8     804    ₹-139,007     -10.8% ⚠    -2.9%     
  MRF         137    MFG        01-Feb-24   142,044.1   126,586.5   8      ₹-123,661     -10.9% ⚠    -1.2%     
  ⚠  WAZ < 0 (momentum below universe mean): SUNPHARMA, MRF, PERSISTENT

  AFTER: Invested ₹28,365,344 | Cash ₹569,142 | Total ₹28,934,485 | Positions 17/20 | Slot ₹1,446,724

========================================================================
  REBALANCE #58  —  01 Aug 2024
  NAV: ₹32,441,569  |  Slot: ₹1,622,078  |  Cash: ₹569,142
========================================================================
  [SECTOR CAP≤4] dropped: MOTHERSON, M&M

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HAL         27     DEFENCE    01-Apr-22   723.8       4,710.4     816    ₹3,253,085    +550.8%   853d  
  OIL         —      OIL&GAS    03-Oct-23   180.7       567.3       5193   ₹2,007,498    +213.9%   303d  
  TORNTPOWER  —      ENERGY     01-Dec-23   914.4       1,776.3     1127   ₹971,397      +94.3%    244d  
  DIXON       6      CON DUR    01-Apr-24   7,586.0     11,661.2    169    ₹688,714      +53.7%    122d  
  SIEMENS     40     ENERGY     01-Apr-24   3,195.5     4,111.4     402    ₹368,215      +28.7%    122d  
  MRF         96     MFG        01-Feb-24   142,044.1   140,039.9   8      ₹-16,034      -1.4%     182d  
  UNIONBANK   138    PSU BNK    01-Apr-24   143.5       127.1       8965   ₹-147,017     -11.4%    122d  

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  OFSS        4      IT         2.603    0.95   +186.5%   +48.1%    10,120.6    160    ₹1,619,303    +1.7%     
  ZYDUSLIFE   5      HEALTH     2.445    0.63   +103.4%   +30.5%    1,227.3     1321   ₹1,621,317    +5.2%     
  LUPIN       8      HEALTH     2.075    0.39   +107.4%   +19.2%    1,941.3     835    ₹1,620,966    +7.9%     
  SUNTV       10     MEDIA      1.987    0.48   +78.1%    +35.8%    852.7       1902   ₹1,621,836    +8.2%     
  BDL         12     DEFENCE    1.952    1.10   +148.8%   +46.7%    1,438.5     1127   ₹1,621,194    -3.8%     
  INFY        14     IT         1.872    0.77   +32.1%    +33.0%    1,741.4     931    ₹1,621,252    +4.6%     
  TORNTPHARM  15     HEALTH     1.781    0.34   +67.8%    +21.5%    3,146.3     515    ₹1,620,338    +5.1%     
  ESCORTS     17     MFG        1.733    0.62   +77.8%    +24.1%    4,093.5     396    ₹1,621,027    +1.4%     
  BALKRISIND  21     MFG        1.688    0.88   +39.7%    +37.9%    3,310.3     490    ₹1,622,024    +4.1%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       2      CONSUMP    01-Jun-23   1,558.5     5,759.0     494    ₹2,075,036    +269.5%     +5.3%     
  TVSMOTOR    11     AUTO       01-Aug-22   911.5       2,564.6     713    ₹1,178,642    +181.4%     +5.0%     
  COLPAL      15     FMCG       01-Aug-23   1,880.1     3,238.2     464    ₹630,178      +72.2%      +7.2%     
  BAJAJ-AUTO  25     AUTO       01-Dec-23   5,767.9     9,358.3     178    ₹639,089      +62.2%      +2.2%     
  BOSCHLTD    19     AUTO       01-Dec-23   21,526.2    33,910.4    47     ₹582,060      +57.5%      -0.3%     
  PERSISTENT  10     IT         01-Dec-23   3,167.3     4,750.7     325    ₹514,625      +50.0%      +2.8%     
  HEROMOTOCO  30     AUTO       01-Dec-23   3,443.0     5,063.6     299    ₹484,572      +47.1%      -1.2%     
  KALYANKJIL  7      CON DUR    01-Apr-24   423.7       561.8       3036   ₹419,228      +32.6%      +5.2%     
  CUMMINSIND  43     INFRA      01-Apr-24   2,927.3     3,738.1     439    ₹355,913      +27.7%      +0.9%     
  SUNPHARMA   28     HEALTH     01-Apr-24   1,598.7     1,688.4     804    ₹72,098       +5.6%       +5.2%     

  AFTER: Invested ₹31,786,874 | Cash ₹637,371 | Total ₹32,424,245 | Positions 19/20 | Slot ₹1,622,078

========================================================================
  REBALANCE #59  —  01 Oct 2024
  NAV: ₹35,030,488  |  Slot: ₹1,751,524  |  Cash: ₹637,371
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  CUMMINSIND  38     INFRA      01-Apr-24   2,927.3     3,797.7     439    ₹382,087      +29.7%    183d  
  ESCORTS     99     MFG        01-Aug-24   4,093.5     4,151.9     396    ₹23,124       +1.4%     61d   
  SUNTV       —      OTHER      01-Aug-24   852.7       819.0       1902   ₹-64,137      -4.0%     61d   
  BALKRISIND  131    MFG        01-Aug-24   3,310.3     3,033.9     490    ₹-135,436     -8.3%     61d   
  ZYDUSLIFE   62     HEALTH     01-Aug-24   1,227.3     1,068.1     1321   ₹-210,338     -13.0%    61d   
  BDL         127    DEFENCE    01-Aug-24   1,438.5     1,118.9     1127   ₹-360,196     -22.2%    61d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────Model is not converging.  Current: 4870.34974679676 is not greater than 4870.372650110542. Delta is -0.02290331378208066
Model is not converging.  Current: 4870.439064711756 is not greater than 4870.472645966678. Delta is -0.033581254921955406
Model is not converging.  Current: 4870.301876177373 is not greater than 4870.317586853787. Delta is -0.01571067641452828
Model is not converging.  Current: 4864.757606616786 is not greater than 4864.8262552022325. Delta is -0.06864858544668095
Model is not converging.  Current: 4864.778003919393 is not greater than 4864.787308617201. Delta is -0.009304697808147466
Model is not converging.  Current: 4864.727256812291 is not greater than 4864.730148844089. Delta is -0.0028920317981828703

  BSE         7      FIN SVC    2.386    0.76   +222.9%   +55.2%    1,282.3     1365   ₹1,750,279    +11.1%    
  VOLTAS      8      CON DUR    2.162    0.85   +113.5%   +27.9%    1,838.5     952    ₹1,750,233    +0.4%     
  UNITDSPR    11     FMCG       1.936    0.54   +56.3%    +26.8%    1,589.7     1101   ₹1,750,239    +3.2%     
  ALKEM       12     HEALTH     1.839    0.57   +68.2%    +25.1%    6,044.3     289    ₹1,746,796    +0.5%     
  HCLTECH     15     IT         1.750    0.70   +45.9%    +23.6%    1,693.6     1034   ₹1,751,222    +2.4%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       1      CONSUMP    01-Jun-23   1,558.5     7,597.1     494    ₹2,983,068    +387.5%     +2.9%     
  TVSMOTOR    11     AUTO       01-Aug-22   911.5       2,817.1     713    ₹1,358,686    +209.1%     +0.7%     
  BAJAJ-AUTO  3      AUTO       01-Dec-23   5,767.9     11,692.4    178    ₹1,054,561    +102.7%     +2.8%     
  COLPAL      4      FMCG       01-Aug-23   1,880.1     3,666.2     464    ₹828,760      +95.0%      +3.9%     
  KALYANKJIL  2      CON DUR    01-Apr-24   423.7       748.0       3036   ₹984,434      +76.5%      +6.8%     
  BOSCHLTD    15     AUTO       01-Dec-23   21,526.2    37,334.5    47     ₹742,989      +73.4%      +6.5%     
  PERSISTENT  16     IT         01-Dec-23   3,167.3     5,435.4     325    ₹737,155      +71.6%      +3.4%     
  HEROMOTOCO  29     AUTO       01-Dec-23   3,443.0     5,420.2     299    ₹591,178      +57.4%      -1.5%     
  SUNPHARMA   5      HEALTH     01-Apr-24   1,598.7     1,889.9     804    ₹234,142      +18.2%      +2.9%     
  LUPIN       6      HEALTH     01-Aug-24   1,941.3     2,180.8     835    ₹200,032      +12.3%      -0.1%     
  TORNTPHARM  10     HEALTH     01-Aug-24   3,146.3     3,308.2     515    ₹83,368       +5.1%       -1.3%     
  OFSS        20     IT         01-Aug-24   10,120.6    10,613.9    160    ₹78,918       +4.9%       +0.6%     
  INFY        37     IT         01-Aug-24   1,741.4     1,790.1     931    ₹45,287       +2.8%       +0.1%     

  AFTER: Invested ₹34,114,286 | Cash ₹905,814 | Total ₹35,020,100 | Positions 18/20 | Slot ₹1,751,524

========================================================================
  REBALANCE #60  —  02 Dec 2024
  NAV: ₹33,050,287  |  Slot: ₹1,652,514  |  Cash: ₹905,814
========================================================================
  [SECTOR CAP≤4] dropped: COFORGE, WIPRO

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TVSMOTOR    79     AUTO       01-Aug-22   911.5       2,474.5     713    ₹1,114,393    +171.5%   854d  
  BAJAJ-AUTO  74     AUTO       01-Dec-23   5,767.9     8,781.1     178    ₹536,349      +52.2%    367d  
  COLPAL      134    FMCG       01-Aug-23   1,880.1     2,792.9     464    ₹423,573      +48.6%    489d  
  HEROMOTOCO  89     AUTO       01-Dec-23   3,443.0     4,476.0     299    ₹308,870      +30.0%    367d  
  ALKEM       95     HEALTH     01-Oct-24   6,044.3     5,593.8     289    ₹-130,202     -7.5%     62d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  INDHOTEL    3      CONSUMP    2.912    1.13   +90.9%    +23.7%    795.2       2078   ₹1,652,343    +6.1%     
  CGPOWER     10     ENERGY     2.015    0.82   +93.6%    +8.5%     752.0       2197   ₹1,652,233    +2.9%     
  ICICIBANK   12     PVT BNK    1.958    0.98   +42.1%    +6.1%     1,294.7     1276   ₹1,651,987    +1.8%     
  OBEROIRLTY  13     REALTY     1.931    1.19   +48.0%    +16.9%    2,054.7     804    ₹1,651,969    +4.6%     
  FEDERALBNK  17     PVT BNK    1.763    0.91   +43.4%    +7.4%     207.8       7951   ₹1,652,392    +1.3%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       8      CONSUMP    01-Jun-23   1,558.5     6,791.3     494    ₹2,585,031    +335.8%     +0.3%     
  PERSISTENT  9      IT         01-Dec-23   3,167.3     5,875.8     325    ₹880,288      +85.5%      +3.1%     
  KALYANKJIL  6      CON DUR    01-Apr-24   423.7       720.0       3036   ₹899,523      +69.9%      +3.3%     
  BOSCHLTD    10     AUTO       01-Dec-23   21,526.2    34,460.4    47     ₹607,909      +60.1%      +0.1%     
  BSE         2      FIN SVC    01-Oct-24   1,282.3     1,516.5     1365   ₹319,744      +18.3%      +0.2%     
  OFSS        3      IT         01-Aug-24   10,120.6    11,378.1    160    ₹201,187      +12.4%      +5.8%     
  SUNPHARMA   19     HEALTH     01-Apr-24   1,598.7     1,780.3     804    ₹145,978      +11.4%      +0.8%     
  LUPIN       35     HEALTH     01-Aug-24   1,941.3     2,056.8     835    ₹96,427       +5.9%       -0.3%     
  TORNTPHARM  31     HEALTH     01-Aug-24   3,146.3     3,278.0     515    ₹67,822       +4.2%       +3.5%     
  HCLTECH     15     IT         01-Oct-24   1,693.6     1,756.4     1034   ₹64,844       +3.7%       +1.0%     
  INFY        55     IT         01-Aug-24   1,741.4     1,787.1     931    ₹42,534       +2.6%       +1.0%     
  UNITDSPR    22     FMCG       01-Oct-24   1,589.7     1,512.0     1101   ₹-85,559      -4.9%       +2.7%     
  VOLTAS      12     CON DUR    01-Oct-24   1,838.5     1,706.2     952    ₹-125,944     -7.2%       +1.4%     

  AFTER: Invested ₹32,827,239 | Cash ₹213,240 | Total ₹33,040,478 | Positions 18/20 | Slot ₹1,652,514

========================================================================
  REBALANCE #61  —  01 Feb 2025
  NAV: ₹30,218,533  |  Slot: ₹1,510,927  |  Cash: ₹213,240
========================================================================
  [SECTOR CAP≤4] dropped: BAJFINANCE

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BOSCHLTD    105    AUTO       01-Dec-23   21,526.2    28,305.1    47     ₹318,608      +31.5%    428d  
  KALYANKJIL  87     CON DUR    01-Apr-24   423.7       504.0       3036   ₹243,693      +18.9%    306d  
  SUNPHARMA   39     HEALTH     01-Apr-24   1,598.7     1,714.9     804    ₹93,467       +7.3%     306d  
  HCLTECH     50     IT         01-Oct-24   1,693.6     1,605.9     1034   ₹-90,721      -5.2%     123d  
  FEDERALBNK  52     PVT BNK    02-Dec-24   207.8       183.0       7951   ₹-197,500     -12.0%    61d   
  OFSS        76     IT         01-Aug-24   10,120.6    8,249.1     160    ₹-299,442     -18.5%    184d  
  CGPOWER     74     ENERGY     02-Dec-24   752.0       609.4       2197   ₹-313,294     -19.0%    61d   Model is not converging.  Current: 4852.936760944686 is not greater than 4852.984521844236. Delta is -0.047760899549757596
Model is not converging.  Current: 4852.861129506308 is not greater than 4852.936494349821. Delta is -0.07536484351294348
Model is not converging.  Current: 4830.439440912872 is not greater than 4830.477301060011. Delta is -0.03786014713932673

  VOLTAS      88     CON DUR    01-Oct-24   1,838.5     1,312.5     952    ₹-500,754     -28.6%    123d  

  ENTRIES (7)
  [52w filter blocked 1: BDL(-30.1%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    2      CONSUMP    2.920    0.62   +45.9%    +28.1%    738.5       2045   ₹1,510,258    +6.3%     
  MUTHOOTFIN  4      FIN SVC    2.798    0.88   +59.5%    +13.0%    2,138.0     706    ₹1,509,418    +0.5%     
  EICHERMOT   5      AUTO       2.534    0.91   +50.7%    +8.8%     5,321.2     283    ₹1,505,886    +5.4%     
  MARUTI      6      AUTO       2.524    0.87   +32.0%    +16.3%    12,778.2    118    ₹1,507,827    +7.8%     
  BAJAJHLDNG  7      FIN SVC    2.328    0.50   +44.7%    +13.8%    11,529.3    131    ₹1,510,335    +4.8%     
  WIPRO       8      IT         2.105    1.04   +32.3%    +12.8%    291.4       5185   ₹1,510,894    +1.3%     
  SBICARD     9      FIN SVC    2.103    0.53   +9.0%     +18.7%    819.6       1843   ₹1,510,562    +9.2%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       20     CONSUMP    01-Jun-23   1,558.5     6,176.8     494    ₹2,281,450    +296.3%     +2.9%     
  PERSISTENT  15     IT         01-Dec-23   3,167.3     5,896.4     325    ₹886,986      +86.2%      -2.5%     
  BSE         3      FIN SVC    01-Oct-24   1,282.3     1,795.7     1365   ₹700,833      +40.0%      -1.5%     
  LUPIN       30     HEALTH     01-Aug-24   1,941.3     2,043.5     835    ₹85,390       +5.3%       -3.1%     
  INFY        25     IT         01-Aug-24   1,741.4     1,760.0     931    ₹17,353       +1.1%       -1.3%     
  TORNTPHARM  21     HEALTH     01-Aug-24   3,146.3     3,173.5     515    ₹14,018       +0.9%       -1.6%     
  INDHOTEL    5      CONSUMP    02-Dec-24   795.2       795.6       2078   ₹825          +0.0%       +1.3%     
  ICICIBANK   33     PVT BNK    02-Dec-24   1,294.7     1,245.9     1276   ₹-62,172      -3.8%       +0.8%     
  UNITDSPR    17     FMCG       01-Oct-24   1,589.7     1,478.3     1101   ₹-122,607     -7.0%       +1.9%     
  OBEROIRLTY  36     REALTY     02-Dec-24   2,054.7     1,832.9     804    ₹-178,294     -10.8%      -2.8%     

  AFTER: Invested ₹29,307,583 | Cash ₹898,405 | Total ₹30,205,988 | Positions 17/20 | Slot ₹1,510,927

========================================================================
  REBALANCE #62  —  01 Apr 2025
  NAV: ₹28,728,130  |  Slot: ₹1,436,407  |  Cash: ₹898,405
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJFINSV, CHOLAFIN, HDFCLIFE

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PERSISTENT  60     IT         01-Dec-23   3,167.3     5,179.0     325    ₹653,804      +63.5%    487d  
  SBICARD     —      FIN SVC    01-Feb-25   819.6       859.1       1843   ₹72,742       +4.8%     59d   
  LUPIN       72     HEALTH     01-Aug-24   1,941.3     1,943.3     835    ₹1,701        +0.1%     243d  
  UNITDSPR    56     FMCG       01-Oct-24   1,589.7     1,387.6     1101   ₹-222,508     -12.7%    182d  
  WIPRO       73     IT         01-Feb-25   291.4       251.1       5185   ₹-209,185     -13.8%    59d   
  INFY        118    IT         01-Aug-24   1,741.4     1,451.2     931    ₹-270,167     -16.7%    243d  
  OBEROIRLTY  121    REALTY     02-Dec-24   2,054.7     1,564.3     804    ₹-394,281     -23.9%    120d  

  ENTRIES (7)
  [52w filter blocked 1: HINDZINC(-41.7%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BAJFINANCE  2      FIN SVC    2.821    1.15   +29.3%    +27.5%    859.2       1671   ₹1,435,730    +0.0%     
  BHARTIARTL  5      CONSUMP    2.495    0.94   +40.3%    +8.6%     1,709.9     840    ₹1,436,279    +2.3%     
  KOTAKBANK   7      PVT BNK    2.365    0.87   +20.9%    +20.1%    428.7       3350   ₹1,436,097    +4.4%     
  DIVISLAB    9      HEALTH     2.095    0.58   +62.7%    -9.0%     5,524.5     260    ₹1,436,382    -3.3%     
  HINDALCO    13     METAL      1.806    1.10   +21.9%    +10.2%    658.9       2179   ₹1,435,782    -1.9%     
  MARICO      14     FMCG       1.777    0.33   +30.8%    +1.8%     641.2       2240   ₹1,436,207    +3.0%     
  HDFCBANK    16     PVT BNK    1.697    0.91   +24.2%    -0.3%     857.9       1674   ₹1,436,202    +0.6%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       54     CONSUMP    01-Jun-23   1,558.5     5,565.3     494    ₹1,979,372    +257.1%     +6.8%     
  BSE         6      FIN SVC    01-Oct-24   1,282.3     1,816.3     1365   ₹728,931      +41.6%      +15.8%    
  MUTHOOTFIN  3      FIN SVC    01-Feb-25   2,138.0     2,291.2     706    ₹108,134      +7.2%       +1.9%     
  BAJAJHLDNG  17     FIN SVC    01-Feb-25   11,529.3    11,683.7    131    ₹20,232       +1.3%       -0.8%     
  ICICIBANK   16     PVT BNK    02-Dec-24   1,294.7     1,308.4     1276   ₹17,474       +1.1%       +1.7%     
  INDHOTEL    29     CONSUMP    02-Dec-24   795.2       799.8       2078   ₹9,695        +0.6%       +2.7%     
  TORNTPHARM  31     HEALTH     01-Aug-24   3,146.3     3,150.0     515    ₹1,892        +0.1%       +0.8%     
  EICHERMOT   10     AUTO       01-Feb-25   5,321.2     5,239.3     283    ₹-23,174      -1.5%       +2.1%     
  JUBLFOOD    25     CONSUMP    01-Feb-25   738.5       659.7       2045   ₹-161,172     -10.7%      +2.1%     
  MARUTI      49     AUTO       01-Feb-25   12,778.2    11,358.2    118    ₹-167,564     -11.1%      -2.4%     

  AFTER: Invested ₹27,555,060 | Cash ₹1,161,133 | Total ₹28,716,194 | Positions 17/20 | Slot ₹1,436,407

========================================================================
  REBALANCE #63  —  02 Jun 2025
  NAV: ₹30,845,370  |  Slot: ₹1,542,268  |  Cash: ₹1,161,133
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TRENT       65     CONSUMP    01-Jun-23   1,558.5     5,610.0     494    ₹2,001,433    +260.0%   732d  Model is not converging.  Current: 4848.768587455565 is not greater than 4848.810582263707. Delta is -0.041994808142590045
Model is not converging.  Current: 4848.777014530556 is not greater than 4848.827128168411. Delta is -0.05011363785524736
Model is not converging.  Current: 4848.73138687983 is not greater than 4848.758214957819. Delta is -0.026828077989193844

  BSE         2      FIN SVC    01-Oct-24   1,282.3     2,693.3     1365   ₹1,926,075    +110.0%   244d  
  BAJAJHLDNG  —      FIN SVC    01-Feb-25   11,529.3    13,161.5    131    ₹213,816      +14.2%    121d  
  INDHOTEL    42     CONSUMP    02-Dec-24   795.2       777.8       2078   ₹-35,995      -2.2%     182d  
  HINDALCO    120    METAL      01-Apr-25   658.9       626.5       2179   ₹-70,627      -4.9%     62d   
  MARUTI      115    AUTO       01-Feb-25   12,778.2    12,158.4    118    ₹-73,135      -4.9%     121d  
  JUBLFOOD    —      CONSUMP    01-Feb-25   738.5       659.3       2045   ₹-162,090     -10.7%    121d  

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SOLARINDS   1      DEFENCE    4.247    1.12   +68.3%    +83.8%    16,284.3    94     ₹1,530,720    +10.5%    
  BHARTIHEXA  2      IT         2.926    1.13   +80.6%    +46.6%    1,840.5     837    ₹1,540,465    +7.6%     
  HDFCLIFE    3      FIN SVC    2.572    0.72   +36.3%    +24.3%    761.9       2024   ₹1,542,027    +1.4%     
  SBILIFE     7      FIN SVC    2.100    0.79   +28.1%    +21.5%    1,800.0     856    ₹1,540,798    +2.0%     
  UNITDSPR    9      FMCG       2.021    0.57   +34.7%    +15.7%    1,533.0     1006   ₹1,542,219    +0.6%     
  PAGEIND     10     MFG        1.635    0.56   +28.4%    +12.0%    45,270.3    34     ₹1,539,190    -1.0%     
  MAXHEALTH   12     HEALTH     1.623    0.78   +43.4%    +16.6%    1,150.3     1340   ₹1,541,365    +0.4%     
  TVSMOTOR    13     AUTO       1.575    1.07   +23.4%    +17.5%    2,753.8     560    ₹1,542,114    +0.1%     
  FEDERALBNK  14     PVT BNK    1.560    0.78   +26.8%    +13.8%    205.0       7522   ₹1,542,077    +3.3%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  DIVISLAB    8      HEALTH     01-Apr-25   5,524.5     6,509.4     260    ₹256,052      +17.8%      +2.0%     
  ICICIBANK   7      PVT BNK    02-Dec-24   1,294.7     1,439.4     1276   ₹184,680      +11.2%      +0.9%     
  HDFCBANK    11     PVT BNK    01-Apr-25   857.9       937.7       1674   ₹133,437      +9.3%       +0.5%     
  MARICO      29     FMCG       01-Apr-25   641.2       697.4       2240   ₹125,977      +8.8%       -1.1%     
  BHARTIARTL  9      CONSUMP    01-Apr-25   1,709.9     1,838.7     840    ₹108,253      +7.5%       +0.7%     
  BAJFINANCE  22     FIN SVC    01-Apr-25   859.2       906.3       1671   ₹78,662       +5.5%       +0.4%     
  MUTHOOTFIN  49     FIN SVC    01-Feb-25   2,138.0     2,203.7     706    ₹46,371       +3.1%       +3.8%     
  EICHERMOT   61     AUTO       01-Feb-25   5,321.2     5,292.4     283    ₹-8,144       -0.5% ⚠     -0.9%     
  TORNTPHARM  59     HEALTH     01-Aug-24   3,146.3     3,094.5     515    ₹-26,691      -1.6% ⚠     -2.5%     
  KOTAKBANK   46     PVT BNK    01-Apr-25   428.7       412.2       3350   ₹-55,077      -3.8%       -1.5%     
  ⚠  WAZ < 0 (momentum below universe mean): TORNTPHARM, EICHERMOT

  AFTER: Invested ₹29,609,020 | Cash ₹1,219,891 | Total ₹30,828,911 | Positions 19/20 | Slot ₹1,542,268

========================================================================
  REBALANCE #64  —  01 Aug 2025
  NAV: ₹31,113,473  |  Slot: ₹1,555,674  |  Cash: ₹1,219,891
========================================================================
  [SECTOR CAP≤4] dropped: APOLLOHOSP

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MARICO      56     FMCG       01-Apr-25   641.2       711.2       2240   ₹156,881      +10.9%    122d  
  TVSMOTOR    33     AUTO       02-Jun-25   2,753.8     2,848.2     560    ₹52,903       +3.4%     60d   
  KOTAKBANK   64     PVT BNK    01-Apr-25   428.7       398.5       3350   ₹-101,189     -7.0%     122d  
  UNITDSPR    128    FMCG       02-Jun-25   1,533.0     1,316.4     1006   ₹-217,873     -14.1%    60d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ETERNAL     2      CONSUMP    2.758    1.13   +34.2%    +31.0%    304.8       5104   ₹1,555,444    +5.8%     
  MANKIND     10     HEALTH     1.937    0.62   +24.8%    +8.3%     2,565.3     606    ₹1,554,565    +0.4%     
  INDIANB     11     PSU BNK    1.808    1.03   +6.0%     +13.5%    608.2       2557   ₹1,555,277    -1.4%     
  BRITANNIA   15     FMCG       1.613    0.46   +0.5%     +7.5%     5,723.0     271    ₹1,550,933    +1.3%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  3      FIN SVC    01-Feb-25   2,138.0     2,571.0     706    ₹305,680      +20.3%      -1.5%     
  TORNTPHARM  10     HEALTH     01-Aug-24   3,146.3     3,645.9     515    ₹257,304      +15.9%      +3.8%     
  HDFCBANK    9      PVT BNK    01-Apr-25   857.9       989.7       1674   ₹220,606      +15.4%      +0.7%     
  DIVISLAB    15     HEALTH     01-Apr-25   5,524.5     6,361.5     260    ₹217,608      +15.1%      -4.2%     
  ICICIBANK   13     PVT BNK    02-Dec-24   1,294.7     1,460.3     1276   ₹211,397      +12.8%      +0.7%     
  BHARTIARTL  16     CONSUMP    01-Apr-25   1,709.9     1,884.4     840    ₹146,617      +10.2%      -2.0%     
  MAXHEALTH   14     HEALTH     02-Jun-25   1,150.3     1,246.0     1340   ₹128,275      +8.3%       -0.4%     
  EICHERMOT   38     AUTO       01-Feb-25   5,321.2     5,528.0     283    ₹58,538       +3.9%       +1.3%     
  PAGEIND     36     MFG        02-Jun-25   45,270.3    46,152.7    34     ₹30,003       +1.9%       -1.3%     
  BAJFINANCE  23     FIN SVC    01-Apr-25   859.2       870.3       1671   ₹18,459       +1.3%       -4.2%     
  BHARTIHEXA  11     IT         02-Jun-25   1,840.5     1,844.6     837    ₹3,465        +0.2%       +2.0%     
  SBILIFE     42     FIN SVC    02-Jun-25   1,800.0     1,793.7     856    ₹-5,385       -0.3%       -1.3%     
  HDFCLIFE    43     FIN SVC    02-Jun-25   761.9       739.1       2024   ₹-46,158      -3.0%       -2.5%     
  FEDERALBNK  53     PVT BNK    02-Jun-25   205.0       194.9       7522   ₹-76,188      -4.9%       -5.7%     
  SOLARINDS   22     DEFENCE    02-Jun-25   16,284.3    13,807.0    94     ₹-232,862     -15.2%      -8.1%     

  AFTER: Invested ₹30,262,442 | Cash ₹843,651 | Total ₹31,106,092 | Positions 19/20 | Slot ₹1,555,674

========================================================================
  REBALANCE #65  —  01 Oct 2025Model is not converging.  Current: 4866.303506466851 is not greater than 4866.329164065007. Delta is -0.025657598155703454
Model is not converging.  Current: 4866.42163228684 is not greater than 4866.479181885541. Delta is -0.057549598701370996
Model is not converging.  Current: 4866.237069362104 is not greater than 4866.248650974458. Delta is -0.011581612353438686
Model is not converging.  Current: 4888.550574992595 is not greater than 4888.569430560734. Delta is -0.01885556813886069
Model is not converging.  Current: 4888.57905722794 is not greater than 4888.613166258664. Delta is -0.034109030723811884
Model is not converging.  Current: 4888.470545228462 is not greater than 4888.476044316895. Delta is -0.005499088433680299

  NAV: ₹31,600,183  |  Slot: ₹1,580,009  |  Cash: ₹843,651
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BHARTIARTL  66     CONSUMP    01-Apr-25   1,709.9     1,867.6     840    ₹132,505      +9.2%     183d  
  ETERNAL     —      CONSUMP    01-Aug-25   304.8       329.0       5104   ₹123,772      +8.0%     61d   
  DIVISLAB    97     HEALTH     01-Apr-25   5,524.5     5,710.0     260    ₹48,218       +3.4%     183d  
  SBILIFE     74     FIN SVC    02-Jun-25   1,800.0     1,798.6     856    ₹-1,197       -0.1%     121d  
  MAXHEALTH   69     HEALTH     02-Jun-25   1,150.3     1,113.2     1340   ₹-49,677      -3.2%     121d  
  MANKIND     —      HEALTH     01-Aug-25   2,565.3     2,439.6     606    ₹-76,167      -4.9%     61d   
  FEDERALBNK  —      PVT BNK    02-Jun-25   205.0       193.8       7522   ₹-84,615      -5.5%     121d  
  PAGEIND     86     MFG        02-Jun-25   45,270.3    41,660.5    34     ₹-122,733     -8.0%     121d  
  SOLARINDS   89     DEFENCE    02-Jun-25   16,284.3    13,374.0    94     ₹-273,564     -17.9%    121d  

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  FORTIS      2      HEALTH     3.545    0.73   +62.4%    +25.0%    989.7       1596   ₹1,579,481    +3.4%     
  HEROMOTOCO  5      AUTO       2.461    1.02   -6.6%     +29.9%    5,328.6     296    ₹1,577,277    +2.3%     
  BOSCHLTD    6      AUTO       2.335    0.87   +4.6%     +19.7%    38,320.0    41     ₹1,571,120    -2.3%     
  NYKAA       7      CONSUMP    2.312    0.65   +20.0%    +14.0%    241.3       6548   ₹1,579,770    +2.1%     
  GODFRYPHLP  9      FMCG       1.937    1.15   +43.6%    +15.1%    3,357.6     470    ₹1,578,054    -1.7%     
  COROMANDEL  10     MFG        1.875    0.92   +38.7%    -0.6%     2,242.2     704    ₹1,578,482    -0.5%     
  SBIN        11     PSU BNK    1.863    0.97   +9.9%     +6.3%     848.8       1861   ₹1,579,617    +1.9%     
  BANKINDIA   12     PSU BNK    1.676    1.12   +16.9%    +4.5%     120.7       13091  ₹1,579,922    +4.9%     
  POLYCAB     15     MFG        1.603    1.06   +9.6%     +8.6%     7,316.3     215    ₹1,573,001    +0.2%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  3      FIN SVC    01-Feb-25   2,138.0     3,118.1     706    ₹691,985      +45.8%      +5.7%     
  EICHERMOT   1      AUTO       01-Feb-25   5,321.2     7,021.5     283    ₹481,198      +32.0%      +3.0%     
  INDIANB     5      PSU BNK    01-Aug-25   608.2       721.7       2557   ₹290,050      +18.6%      +4.8%     
  BAJFINANCE  10     FIN SVC    01-Apr-25   859.2       981.7       1671   ₹204,633      +14.3%      +0.7%     
  TORNTPHARM  25     HEALTH     01-Aug-24   3,146.3     3,535.8     515    ₹200,589      +12.4%      -0.6%     
  HDFCBANK    37     PVT BNK    01-Apr-25   857.9       949.5       1674   ₹153,336      +10.7%      +0.4%     
  ICICIBANK   55     PVT BNK    02-Dec-24   1,294.7     1,372.0     1276   ₹98,685       +6.0%       -1.3%     
  BRITANNIA   39     FMCG       01-Aug-25   5,723.0     5,966.5     271    ₹65,988       +4.3%       -0.3%     
  HDFCLIFE    52     FIN SVC    02-Jun-25   761.9       761.3       2024   ₹-1,183       -0.1%       -0.7%     
  BHARTIHEXA  63     IT         02-Jun-25   1,840.5     1,660.8     837    ₹-150,376     -9.8% ⚠     -2.7%     
  ⚠  WAZ < 0 (momentum below universe mean): BHARTIHEXA

  AFTER: Invested ₹31,579,895 | Cash ₹3,430 | Total ₹31,583,326 | Positions 19/20 | Slot ₹1,580,009

========================================================================
  REBALANCE #66  —  01 Dec 2025
  NAV: ₹33,259,373  |  Slot: ₹1,662,969  |  Cash: ₹3,430
========================================================================
  [SECTOR CAP≤4] dropped: PNB

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ICICIBANK   78     PVT BNK    02-Dec-24   1,294.7     1,390.1     1276   ₹121,781      +7.4%     364d  
  POLYCAB     77     MFG        01-Oct-25   7,316.3     7,366.0     215    ₹10,699       +0.7%     61d   
  HDFCLIFE    76     FIN SVC    02-Jun-25   761.9       764.0       2024   ₹4,363        +0.3%     182d  
  BHARTIHEXA  64     IT         02-Jun-25   1,840.5     1,748.7     837    ₹-76,803      -5.0%     182d  
  BOSCHLTD    115    AUTO       01-Oct-25   38,320.0    36,335.0    41     ₹-81,385      -5.2%     61d   
  GODFRYPHLP  89     FMCG       01-Oct-25   3,357.6     2,837.9     470    ₹-244,241     -15.5%    61d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  CANBK       2      PSU BNK    3.171    1.00   +53.2%    +44.9%    145.7       11414  ₹1,662,950    +3.7%     
  M&MFIN      4      FIN SVC    2.857    1.20   +39.5%    +44.9%    367.9       4520   ₹1,662,908    +9.0%     
  MARUTI      7      AUTO       2.296    0.79   +48.7%    +8.8%     16,097.0    103    ₹1,657,991    +1.2%     
  TVSMOTOR    12     AUTO       2.060    1.19   +51.7%    +11.8%    3,649.0     455    ₹1,660,316    +4.5%     
  RELIANCE    13     OIL&GAS    1.890    1.16   +21.4%    +15.4%    1,558.9     1066   ₹1,661,779    +2.7%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  2      FIN SVC    01-Feb-25   2,138.0     3,779.1     706    ₹1,158,659    +76.8%      +6.6%     
  INDIANB     4      PSU BNK    01-Aug-25   608.2       868.8       2557   ₹666,350      +42.8%      +2.6%     
  EICHERMOT   7      AUTO       01-Feb-25   5,321.2     7,125.5     283    ₹510,630      +33.9%      +1.6%     
  BANKINDIA   8      PSU BNK    01-Oct-25   120.7       142.6       13091  ₹286,625      +18.1%      +1.8%     
  BAJFINANCE  10     FIN SVC    01-Apr-25   859.2       1,014.9     1671   ₹260,104      +18.1%      -0.3%     
  TORNTPHARM  54     HEALTH     01-Aug-24   3,146.3     3,704.1     515    ₹287,269      +17.7%      +0.5%     
  HEROMOTOCO  11     AUTO       01-Oct-25   5,328.6     6,175.1     296    ₹250,564      +15.9%      +7.1%     Model is not converging.  Current: 4885.796798817921 is not greater than 4885.797879002197. Delta is -0.0010801842763612512
Model is not converging.  Current: 4884.695353552464 is not greater than 4884.756070619852. Delta is -0.06071706738748617
Model is not converging.  Current: 4884.64362477252 is not greater than 4884.672504244801. Delta is -0.02887947228100529
Model is not converging.  Current: 4870.6540240189925 is not greater than 4870.798257033. Delta is -0.14423301400711352
Model is not converging.  Current: 4868.433535959512 is not greater than 4868.597109411529. Delta is -0.16357345201777207
Model is not converging.  Current: 4868.327853396782 is not greater than 4868.350977890013. Delta is -0.023124493231080123

  HDFCBANK    42     PVT BNK    01-Apr-25   857.9       985.8       1674   ₹214,019      +14.9%      +0.5%     
  SBIN        9      PSU BNK    01-Oct-25   848.8       955.9       1861   ₹199,257      +12.6%      +1.1%     
  NYKAA       16     CONSUMP    01-Oct-25   241.3       264.9       6548   ₹154,795      +9.8%       +0.6%     
  COROMANDEL  53     MFG        01-Oct-25   2,242.2     2,382.9     704    ₹99,085       +6.3%       +5.3%     
  BRITANNIA   62     FMCG       01-Aug-25   5,723.0     5,813.5     271    ₹24,526       +1.6% ⚠     -1.0%     
  FORTIS      58     HEALTH     01-Oct-25   989.7       904.8       1596   ₹-135,341     -8.6%       -4.9%     
  ⚠  WAZ < 0 (momentum below universe mean): BRITANNIA

  AFTER: Invested ₹32,370,818 | Cash ₹878,692 | Total ₹33,249,511 | Positions 18/20 | Slot ₹1,662,969

========================================================================
  REBALANCE #67  —  02 Feb 2026
  NAV: ₹31,790,302  |  Slot: ₹1,589,515  |  Cash: ₹878,692
========================================================================
  [SECTOR CAP≤4] dropped: UNIONBANK, HINDZINC

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  EICHERMOT   28     AUTO       01-Feb-25   5,321.2     6,985.5     283    ₹471,010      +31.3%    366d  
  HDFCBANK    66     PVT BNK    01-Apr-25   857.9       913.0       1674   ₹92,159       +6.4%     307d  
  BAJFINANCE  82     FIN SVC    01-Apr-25   859.2       898.2       1671   ₹65,127       +4.5%     307d  
  BRITANNIA   47     FMCG       01-Aug-25   5,723.0     5,888.5     271    ₹44,850       +2.9%     185d  
  NYKAA       36     CONSUMP    01-Oct-25   241.3       237.6       6548   ₹-23,835      -1.5%     124d  
  COROMANDEL  38     MFG        01-Oct-25   2,242.2     2,206.9     704    ₹-24,824      -1.6%     124d  
  MARUTI      81     AUTO       01-Dec-25   16,097.0    14,384.0    103    ₹-176,439     -10.6%    63d   
  RELIANCE    72     OIL&GAS    01-Dec-25   1,558.9     1,384.0     1066   ₹-186,434     -11.2%    63d   
  FORTIS      69     HEALTH     01-Oct-25   989.7       838.0       1596   ₹-242,113     -15.3%    124d  

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ASHOKLEY    1      AUTO       4.155    0.04   +101.9%   +38.3%    191.1       8316   ₹1,589,515    +3.5%     
  NATIONALUM  2      METAL      3.828    0.12   +96.4%    +60.3%    363.3       4374   ₹1,589,152    +1.4%     
  SHRIRAMFIN  3      FIN SVC    3.288    -0.04  +90.9%    +29.3%    962.1       1652   ₹1,589,389    -2.6%     
  VEDL        4      METAL      3.064    0.11   +64.1%    +33.9%    649.8       2446   ₹1,589,406    -1.1%     
  LTF         5      FIN SVC    2.651    0.05   +99.2%    +2.6%     274.7       5785   ₹1,589,264    -5.2%     
  HINDALCO    7      METAL      2.435    0.10   +59.7%    +9.7%     930.5       1708   ₹1,589,294    -1.2%     
  BPCL        12     OIL&GAS    2.300    0.01   +52.5%    +7.9%     366.7       4334   ₹1,589,278    +4.0%     
  APLAPOLLO   13     METAL      2.293    0.17   +37.6%    +16.1%    2,079.3     764    ₹1,588,585    +5.3%     
  BEL         15     DEFENCE    2.201    0.28   +68.3%    +3.1%     437.2       3635   ₹1,589,362    +4.1%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  11     FIN SVC    01-Feb-25   2,138.0     3,505.5     706    ₹965,436      +64.0%      -8.4%     
  INDIANB     21     PSU BNK    01-Aug-25   608.2       817.4       2557   ₹534,783      +34.4%      -3.0%     
  TORNTPHARM  18     HEALTH     01-Aug-24   3,146.3     3,952.2     515    ₹415,041      +25.6%      +0.8%     
  BANKINDIA   20     PSU BNK    01-Oct-25   120.7       146.9       13091  ₹342,783      +21.7%      -3.1%     
  SBIN        6      PSU BNK    01-Oct-25   848.8       1,010.5     1861   ₹300,897      +19.0%      -0.3%     
  HEROMOTOCO  26     AUTO       01-Oct-25   5,328.6     5,515.5     296    ₹55,310       +3.5%       -0.1%     
  TVSMOTOR    10     AUTO       01-Dec-25   3,649.0     3,633.1     455    ₹-7,255       -0.4%       -0.6%     
  CANBK       9      PSU BNK    01-Dec-25   145.7       141.8       11414  ₹-44,529      -2.7%       -3.6%     
  M&MFIN      23     FIN SVC    01-Dec-25   367.9       353.6       4520   ₹-64,636      -3.9%       -2.9%     

  AFTER: Invested ₹31,209,098 | Cash ₹564,220 | Total ₹31,773,318 | Positions 18/20 | Slot ₹1,589,515

========================================================================
  REBALANCE #68  —  01 Apr 2026
  NAV: ₹29,894,709  |  Slot: ₹1,494,735  |  Cash: ₹564,220
========================================================================
  [SECTOR CAP≤4] dropped: SAIL

  [REGIME OFF] Nifty 200 HMM bear-state prob 0.88 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  45     FIN SVC    01-Feb-25   2,138.0     3,228.7     706    ₹770,044      +51.0%      -1.7%     
  INDIANB     5      PSU BNK    01-Aug-25   608.2       869.5       2557   ₹667,978      +42.9%      -0.2%     
  TORNTPHARM  10     HEALTH     01-Aug-24   3,146.3     4,103.1     515    ₹492,782      +30.4%      -3.5%     
  SBIN        11     PSU BNK    01-Oct-25   848.8       999.8       1861   ₹280,971      +17.8%      -4.7%     
  BANKINDIA   27     PSU BNK    01-Oct-25   120.7       137.2       13091  ₹216,395      +13.7%      -6.2%     
  NATIONALUM  1      METAL      02-Feb-26   363.3       397.5       4374   ₹149,382      +9.4%       +6.3%     
  VEDL        6      METAL      02-Feb-26   649.8       677.2       2446   ₹67,026       +4.2%       +1.1%     
  HINDALCO    22     METAL      02-Feb-26   930.5       904.6       1708   ₹-44,237      -2.8%       +0.1%     
  HEROMOTOCO  26     AUTO       01-Oct-25   5,328.6     5,122.0     296    ₹-61,165      -3.9%       -3.5%     
  BEL         14     DEFENCE    02-Feb-26   437.2       418.7       3635   ₹-67,388      -4.2%       -2.0%     
  TVSMOTOR    23     AUTO       01-Dec-25   3,649.0     3,425.8     455    ₹-101,577     -6.1%       -2.8%     
  SHRIRAMFIN  —      FIN SVC    02-Feb-26   962.1       900.5       1652   ₹-101,681     -6.4%       -6.5%     
  APLAPOLLO   25     METAL      02-Feb-26   2,079.3     1,934.8     764    ₹-110,398     -6.9%       -3.7%     
  LTF         39     FIN SVC    02-Feb-26   274.7       242.1       5785   ₹-188,444     -11.9%      -6.6%     
  CANBK       44     PSU BNK    01-Dec-25   145.7       123.2       11414  ₹-256,348     -15.4%      -6.8%     
  M&MFIN      —      FIN SVC    01-Dec-25   367.9       289.7       4520   ₹-353,464     -21.3%      -10.3%    
  ASHOKLEY    43     AUTO       02-Feb-26   191.1       146.6       8316   ₹-370,314     -23.3%      -14.6%    
  BPCL        89     OIL&GAS    02-Feb-26   366.7       281.2       4334   ₹-370,340     -23.3% ⚠    -8.6%     
  ⚠  WAZ < 0 (momentum below universe mean): BPCL

  AFTER: Invested ₹29,330,488 | Cash ₹564,220 | Total ₹29,894,709 | Positions 18/20 | Slot ₹1,494,735

========================================================================
  REBALANCE #69  —  01 Jun 2026
  NAV: ₹29,504,071  |  Slot: ₹1,475,204  |  Cash: ₹564,220
========================================================================
  [SECTOR CAP≤4] dropped: ADANIGREEN, PREMIERENE, CGPOWER

  EXITS (15)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  INDIANB     72     PSU BNK    01-Aug-25   608.2       792.9       2557   ₹472,066      +30.4%    304d  
  HINDALCO    —      METAL      02-Feb-26   930.5       1,141.3     1708   ₹360,046      +22.7%    119d  
  BANKINDIA   89     PSU BNK    01-Oct-25   120.7       136.7       13091  ₹210,142      +13.3%    243d  
  SBIN        99     PSU BNK    01-Oct-25   848.8       954.1       1861   ₹195,963      +12.4%    243d  
  LTF         41     FIN SVC    02-Feb-26   274.7       271.2       5785   ₹-20,083      -1.3%     119d  
  SHRIRAMFIN  66     FIN SVC    02-Feb-26   962.1       919.0       1652   ₹-71,119      -4.5%     119d  
  BEL         85     DEFENCE    02-Feb-26   437.2       407.2       3635   ₹-109,190     -6.9%     119d  
  TVSMOTOR    86     AUTO       01-Dec-25   3,649.0     3,344.4     455    ₹-138,614     -8.3%     182d  
  HEROMOTOCO  98     AUTO       01-Oct-25   5,328.6     4,819.9     296    ₹-150,586     -9.5%     243d  
  APLAPOLLO   125    METAL      02-Feb-26   2,079.3     1,788.3     764    ₹-222,324     -14.0%    119d  
  CANBK       96     PSU BNK    01-Dec-25   145.7       123.9       11414  ₹-249,056     -15.0%    182d  
  BPCL        124    OIL&GAS    02-Feb-26   366.7       296.9       4334   ₹-302,730     -19.0%    119d  
  M&MFIN      103    FIN SVC    01-Dec-25   367.9       295.1       4520   ₹-329,056     -19.8%    182d  
  ASHOKLEY    107    AUTO       02-Feb-26   191.1       147.3       8316   ₹-364,836     -23.0%    119d  
  VEDL        115    METAL      02-Feb-26   649.8       337.1       2446   ₹-764,737     -48.1%    119d  

  ENTRIES (15)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  LAURUSLABS  1      HEALTH     3.887    0.18   +131.1%   +29.1%    1,388.4     1062   ₹1,474,481    +6.1%     
  BHEL        3      ENERGY     3.044    -0.03  +55.9%    +54.5%    404.8       3644   ₹1,475,091    +1.7%     
  POWERINDIA  4      ENERGY     3.010    0.11   +110.3%   +45.6%    36,380.0    40     ₹1,455,200    +4.3%     
  OFSS        5      IT         2.976    0.11   +26.7%    +58.5%    10,191.0    144    ₹1,467,504    +6.7%     
  CUMMINSIND  6      INFRA      2.963    0.15   +93.7%    +15.5%    5,680.5     259    ₹1,471,250    +3.4%     
  ADANIENSOL  7      ENERGY     2.901    0.23   +70.2%    +47.7%    1,496.5     985    ₹1,474,052    +6.7%     
  MCX         8      FIN SVC    2.837    -0.13  +125.9%   +18.0%    2,890.5     510    ₹1,474,155    -7.7%     
  GVT&D       9      ENERGY     2.731    -0.10  +115.5%   +24.6%    4,755.0     310    ₹1,474,050    +1.2%     
  BSE         10     FIN SVC    2.691    -0.28  +69.2%    +46.7%    4,066.6     362    ₹1,472,109    +0.5%     
  SAIL        12     METAL      2.320    -0.13  +60.4%    +23.5%    203.7       7242   ₹1,475,123    +4.0%     
  RADICO      13     FMCG       2.289    -0.09  +43.9%    +28.8%    3,506.1     420    ₹1,472,562    +1.4%     
  POLYCAB     14     MFG        2.275    0.28   +59.9%    +13.6%    9,435.7     156    ₹1,471,974    +3.7%     
  SOLARINDS   15     DEFENCE    2.121    -0.12  +14.7%    +35.2%    18,203.0    81     ₹1,474,443    +4.3%     
  ZYDUSLIFE   16     HEALTH     2.086    0.12   +20.2%    +18.7%    1,091.2     1351   ₹1,474,211    +6.7%     
  GLENMARK    19     HEALTH     1.937    0.20   +59.3%    +5.6%     2,201.2     670    ₹1,474,804    -5.3%     

  HOLDS (3)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  39     FIN SVC    01-Feb-25   2,138.0     3,246.4     706    ₹782,540      +51.8%      -3.3%     
  TORNTPHARM  29     HEALTH     01-Aug-24   3,146.3     4,350.4     515    ₹620,118      +38.3%      -1.1%     
  NATIONALUM  2      METAL      02-Feb-26   363.3       434.2       4374   ₹310,038      +19.5%      +4.6%     

  AFTER: Invested ₹28,512,614 | Cash ₹965,238 | Total ₹29,477,852 | Positions 18/20 | Slot ₹1,475,204

========================================================================
  FINAL SUMMARY
========================================================================
  Period        : 2015-01-01 → 2026-07-01  (11.5 years)
  Starting Cap  : ₹2,000,000
  Final Value   : ₹29,175,288
  Total Return  : +1358.8%  (on total invested)
  CAGR          : +26.3%

  Closed Trades : 378  |  Open: 18
  Win Rate      : 54.2%  (205W / 173L)
  Profit Factor : 3.19
  Avg hold      : 187 days
  Total charges : ₹760,735
  Closed net P&L: ₹25,556,567
  Open unreal   : ₹1,410,132

  YEAR-BY-YEAR:
  2015  +  4.2%  ████
  2016  + 11.3%  ███████████
  2017  + 51.6%  ████████████████████████████████████████
  2018  -  0.3%  
  2019  + 13.2%  █████████████
  2020  + 25.0%  ████████████████████████
  2021  + 96.4%  ████████████████████████████████████████
  2022  + 49.3%  ████████████████████████████████████████
  2023  + 41.8%  ████████████████████████████████████████
  2024  + 60.3%  ████████████████████████████████████████
  2025  +  0.6%  
  2026  - 11.3%  ░░░░░░░░░░░

  Rebalance NAV exported → mom20_rebal.csv (69 rows)
