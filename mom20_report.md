=== Mom20 — Monthly Rebalance, β≤1.2 | Regime ON [SMA200] | Sector≤4 ===
    top_n=20 buffer_in=15 buffer_out=40 beta_cap=1.2
Loading PIT universe...
  388 unique PIT tickers across all periods
Loading EPS data...
  377 stocks with EPS data
  Sector map loaded: 43 PIT dates
Loading cached data from /Users/jay/dev/relative_strength/data/cache/mom15_daily.pkl...
Fetching Nifty 50 (beta)...
  3126 bars
Fetching Nifty 200 (regime filter)...
  3132 bars
  Trading days in backtest: 2840 (2015-01-01 → 2026-07-01)
  Rebalance dates: 139

==============================================================================================
  MOM20 PIT BACKTEST  |  NAV/20 slot  |  Monthly Rebalance  |  Beta≤1.2  |  Regime ON [SMA200]
==============================================================================================

========================================================================
  REBALANCE #01  —  01 Jan 2015
  NAV: ₹2,000,000  |  Slot: ₹100,000  |  Cash: ₹2,000,000
========================================================================
  [SECTOR CAP≤4] dropped: ASHOKLEY

  EXITS (0)
    —

  ENTRIES (20)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  REPCOHOME   1      FIN SVC    3.577    -0.16  +115.7%   +66.0%    631.6       158    ₹99,788       +12.6%    
  PAGEIND     2      MFG        3.387    0.15   +140.1%   +56.0%    11,008.1    9      ₹99,073       +9.7%     
  WHIRLPOOL   3      CON DUR    3.067    0.28   +216.6%   +49.8%    637.1       156    ₹99,388       +4.1%     
  TVSMOTOR    4      AUTO       2.991    -0.58  +359.7%   +29.3%    258.8       386    ₹99,878       +8.4%     
  EICHERMOT   5      AUTO       2.982    0.03   +212.5%   +31.9%    1,410.3     70     ₹98,724       +3.6%     
  HONAUT      6      MFG        2.982    -0.24  +171.3%   +29.2%    6,711.0     14     ₹93,954       +0.1%     
  AUROPHARMA  7      HEALTH     2.792    -0.08  +199.8%   +41.0%    532.2       187    ₹99,525       +1.7%     
  TORNTPHARM  8      HEALTH     2.719    -0.15  +145.3%   +34.0%    489.9       204    ₹99,930       +6.4%     
  PIDILITIND  9      MFG        2.712    -0.05  +91.3%    +38.5%    259.5       385    ₹99,900       +10.1%    
  BOSCHLTD    10     AUTO       2.631    0.01   +118.4%   +32.6%    17,477.5    5      ₹87,387       +2.5%     
  BEL         11     DEFENCE    2.577    -0.03  +196.5%   +39.9%    24.5        4089   ₹99,988       +7.5%     
  SRF         12     MFG        2.560    -0.30  +308.8%   +21.9%    165.2       605    ₹99,937       +1.6%     
  DCBBANK     13     PVT BNK    2.535    -0.32  +141.0%   +46.0%    116.6       857    ₹99,918       +8.8%     
  AMARAJABAT  14     AUTO       2.484    -0.01  +146.5%   +35.0%    741.2       134    ₹99,322       +6.1%     
  LICHSGFIN   15     FIN SVC    2.465    -0.42  +120.7%   +41.8%    367.9       271    ₹99,704       +3.5%     
  BAJFINANCE  16     FIN SVC    2.418    -0.08  +138.0%   +34.0%    33.6        2980   ₹99,985       +4.9%     
  INDUSINDBK  18     PVT BNK    2.357    -0.01  +93.3%    +30.4%    763.4       130    ₹99,241       +4.3%     
  BRITANNIA   19     FMCG       2.335    0.01   +111.4%   +31.6%    795.6       125    ₹99,453       +4.1%     
  BHARATFORG  20     DEFENCE    2.322    -0.26  +190.5%   +18.0%    431.4       231    ₹99,664       +1.3%     
  AXISBANK    21     PVT BNK    2.247    -0.11  +102.2%   +26.7%    487.0       205    ₹99,827       +2.6%     

  HOLDS (0)
    —

  AFTER: Invested ₹1,974,586 | Cash ₹23,070 | Total ₹1,997,655 | Positions 20/20 | Slot ₹100,000

========================================================================
  REBALANCE #02  —  02 Feb 2015
  NAV: ₹2,161,329  |  Slot: ₹108,066  |  Cash: ₹23,070
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJFINSV

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  INDUSINDBK  33     PVT BNK    01-Jan-15   763.4       822.2       130    ₹7,645        +7.7%     32d   
  BRITANNIA   34     FMCG       01-Jan-15   795.6       825.4       125    ₹3,725        +3.7%     32d   
  TVSMOTOR    27     AUTO       01-Jan-15   258.8       265.9       386    ₹2,771        +2.8%     32d   
  PIDILITIND  28     MFG        01-Jan-15   259.5       257.4       385    ₹-782         -0.8%     32d   
  DCBBANK     40     PVT BNK    01-Jan-15   116.6       114.1       857    ₹-2,176       -2.2%     32d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SHREECEM    7      INFRA      2.660    0.26   +147.9%   +25.9%    10,437.7    10     ₹104,377      +6.1%     
  IBULHSGFIN  8      FIN SVC    2.652    0.53   +214.3%   +50.8%    382.5       282    ₹107,871      +8.0%     
  ASHOKLEY    9      AUTO       2.626    -0.15  +293.2%   +42.7%    26.9        4016   ₹108,058      +6.9%     
  BBTC        11     FMCG       2.556    0.14   +322.4%   +84.3%    438.1       246    ₹107,776      +1.1%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BOSCHLTD    1      AUTO       01-Jan-15   17,477.5    22,437.0    5      ₹24,798       +28.4%      +14.3%    
  AXISBANK    3      PVT BNK    01-Jan-15   487.0       597.6       205    ₹22,672       +22.7%      +11.9%    
  BAJFINANCE  10     FIN SVC    01-Jan-15   33.6        39.9        2980   ₹18,867       +18.9%      +7.9%     
  BEL         5      DEFENCE    01-Jan-15   24.5        28.5        4089   ₹16,660       +16.7%      +6.7%     
  BHARATFORG  12     DEFENCE    01-Jan-15   431.4       490.3       231    ₹13,601       +13.6%      +6.6%     
  AUROPHARMA  14     HEALTH     01-Jan-15   532.2       596.9       187    ₹12,102       +12.2%      +7.4%     
  SRF         16     MFG        01-Jan-15   165.2       181.9       605    ₹10,092       +10.1%      +4.3%     
  LICHSGFIN   22     FIN SVC    01-Jan-15   367.9       403.0       271    ₹9,513        +9.5%       +1.2%     
  WHIRLPOOL   4      CON DUR    01-Jan-15   637.1       693.1       156    ₹8,728        +8.8%       +3.3%     
  EICHERMOT   6      AUTO       01-Jan-15   1,410.3     1,530.4     70     ₹8,405        +8.5%       +6.3%     
  AMARAJABAT  15     AUTO       01-Jan-15   741.2       801.8       134    ₹8,113        +8.2%       +4.9%     
  HONAUT      2      MFG        01-Jan-15   6,711.0     7,224.6     14     ₹7,190        +7.7%       +3.3%     
  TORNTPHARM  24     HEALTH     01-Jan-15   489.9       487.4       204    ₹-499         -0.5%       +0.5%     
  REPCOHOME   19     FIN SVC    01-Jan-15   631.6       609.4       158    ₹-3,501       -3.5%       -0.6%     
  PAGEIND     25     MFG        01-Jan-15   11,008.1    10,535.8    9      ₹-4,251       -4.3%       +1.8%     

  AFTER: Invested ₹2,056,769 | Cash ₹104,052 | Total ₹2,160,821 | Positions 19/20 | Slot ₹108,066

========================================================================
  REBALANCE #03  —  02 Mar 2015
  NAV: ₹2,225,006  |  Slot: ₹111,250  |  Cash: ₹104,052
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  AMARAJABAT  49     AUTO       01-Jan-15   741.2       787.7       134    ₹6,224        +6.3%     60d   
  BBTC        48     FMCG       02-Feb-15   438.1       428.9       246    ₹-2,277       -2.1%     28d   
  AUROPHARMA  85     HEALTH     01-Jan-15   532.2       514.3       187    ₹-3,360       -3.4%     60d   
  TORNTPHARM  81     HEALTH     01-Jan-15   489.9       454.8       204    ₹-7,155       -7.2%     60d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  WOCKPHARMA  1      HEALTH     3.542    0.27   +336.9%   +96.7%    1,510.0     73     ₹110,227      +14.3%    
  SPARC       3      HEALTH     3.270    0.29   +171.7%   +130.5%   417.9       266    ₹111,163      +15.5%    
  SIEMENS     11     ENERGY     2.758    0.28   +167.3%   +56.2%    731.1       152    ₹111,129      +14.6%    
  INDUSINDBK  13     PVT BNK    2.540    0.31   +152.2%   +25.9%    888.0       125    ₹110,995      +9.2%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BHARATFORG  4      DEFENCE    01-Jan-15   431.4       593.4       231    ₹37,412       +37.5%      +8.0%     
  HONAUT      2      MFG        01-Jan-15   6,711.0     8,967.0     14     ₹31,585       +33.6%      +7.8%     
  BOSCHLTD    5      AUTO       01-Jan-15   17,477.5    23,270.7    5      ₹28,966       +33.1%      +3.2%     
  AXISBANK    10     PVT BNK    01-Jan-15   487.0       626.8       205    ₹28,675       +28.7%      +13.9%    
  BAJFINANCE  12     FIN SVC    01-Jan-15   33.6        41.9        2980   ₹24,903       +24.9%      +6.0%     
  BEL         7      DEFENCE    01-Jan-15   24.5        30.4        4089   ₹24,121       +24.1%      +1.0%     
  LICHSGFIN   36     FIN SVC    01-Jan-15   367.9       408.1       271    ₹10,878       +10.9%      +3.0%     
  ASHOKLEY    8      AUTO       02-Feb-15   26.9        29.7        4016   ₹11,288       +10.4%      +10.3%    
  SRF         19     MFG        01-Jan-15   165.2       180.3       605    ₹9,124        +9.1%       +1.7%     
  EICHERMOT   15     AUTO       01-Jan-15   1,410.3     1,517.9     70     ₹7,527        +7.6%       +3.1%     
  IBULHSGFIN  9      FIN SVC    02-Feb-15   382.5       403.3       282    ₹5,863        +5.4%       +3.1%     
  SHREECEM    6      INFRA      02-Feb-15   10,437.7    10,965.3    10     ₹5,276        +5.1%       +4.5%     
  WHIRLPOOL   21     CON DUR    01-Jan-15   637.1       660.5       156    ₹3,649        +3.7%       -0.8%     
  PAGEIND     32     MFG        01-Jan-15   11,008.1    10,691.8    9      ₹-2,846       -2.9%       +2.4%     
  REPCOHOME   33     FIN SVC    01-Jan-15   631.6       611.5       158    ₹-3,178       -3.2%       +1.2%     

  AFTER: Invested ₹2,164,485 | Cash ₹59,995 | Total ₹2,224,480 | Positions 19/20 | Slot ₹111,250

========================================================================
  REBALANCE #04  —  01 Apr 2015
  NAV: ₹2,228,778  |  Slot: ₹111,439  |  Cash: ₹59,995
========================================================================
  [SECTOR CAP≤4] dropped: NATCOPHARM, SUNPHARMA

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJFINANCE  25     FIN SVC    01-Jan-15   33.6        39.7        2980   ₹18,294       +18.3%    90d   
  AXISBANK    38     PVT BNK    01-Jan-15   487.0       551.4       205    ₹13,216       +13.2%    90d   
  EICHERMOT   31     AUTO       01-Jan-15   1,410.3     1,485.0     70     ₹5,223        +5.3%     90d   
  LICHSGFIN   78     FIN SVC    01-Jan-15   367.9       374.9       271    ₹1,900        +1.9%     90d   
  INDUSINDBK  27     PVT BNK    02-Mar-15   888.0       869.9       125    ₹-2,262       -2.0%     30d   
  REPCOHOME   83     FIN SVC    01-Jan-15   631.6       580.5       158    ₹-8,074       -8.1%     90d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  LUPIN       2      HEALTH     3.567    0.27   +116.3%   +42.7%    1,917.4     58     ₹111,212      +6.5%     
  SADBHAV     7      INFRA      2.782    0.39   +291.5%   +40.1%    342.4       325    ₹111,284      +1.8%     
  NCC         10     INFRA      2.434    0.40   +460.7%   +33.2%    97.5        1142   ₹111,358      +13.5%    
  BRITANNIA   11     FMCG       2.399    0.20   +157.9%   +22.1%    946.0       117    ₹110,676      +1.4%     
  AJANTPHARM  12     HEALTH     2.335    0.35   +208.0%   +30.3%    737.1       151    ₹111,303      +2.9%     
  EMAMILTD    17     FMCG       2.160    0.26   +126.1%   +26.4%    420.8       264    ₹111,097      -1.9%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BHARATFORG  4      DEFENCE    01-Jan-15   431.4       610.1       231    ₹41,259       +41.4%      +4.3%     
  BOSCHLTD    8      AUTO       01-Jan-15   17,477.5    22,972.6    5      ₹27,476       +31.4%      -1.7%     
  HONAUT      6      MFG        01-Jan-15   6,711.0     8,387.7     14     ₹23,474       +25.0%      -0.9%     
  SPARC       1      HEALTH     02-Mar-15   417.9       500.0       266    ₹21,836       +19.6%      +5.2%     
  BEL         14     DEFENCE    01-Jan-15   24.5        29.1        4089   ₹19,084       +19.1%      +4.8%     
  PAGEIND     24     MFG        01-Jan-15   11,008.1    12,831.4    9      ₹16,410       +16.6%      +6.0%     
  SRF         19     MFG        01-Jan-15   165.2       189.4       605    ₹14,644       +14.7%      +6.7%     
  WOCKPHARMA  3      HEALTH     02-Mar-15   1,510.0     1,701.9     73     ₹14,014       +12.7%      +3.9%     
  ASHOKLEY    5      AUTO       02-Feb-15   26.9        30.2        4016   ₹13,088       +12.1%      +4.6%     
  WHIRLPOOL   16     CON DUR    01-Jan-15   637.1       709.4       156    ₹11,275       +11.3%      +0.7%     
  SIEMENS     9      ENERGY     02-Mar-15   731.1       737.5       152    ₹976          +0.9%       +3.3%     
  SHREECEM    18     INFRA      02-Feb-15   10,437.7    10,419.0    10     ₹-187         -0.2%       -0.1%     
  IBULHSGFIN  22     FIN SVC    02-Feb-15   382.5       368.0       282    ₹-4,105       -3.8%       +0.6%     

  AFTER: Invested ₹2,198,392 | Cash ₹29,595 | Total ₹2,227,986 | Positions 19/20 | Slot ₹111,439

========================================================================
  REBALANCE #05  —  04 May 2015
  NAV: ₹2,074,269  |  Slot: ₹103,713  |  Cash: ₹29,595
========================================================================
  [SECTOR CAP≤4] dropped: NATCOPHARM, PIIND

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BEL         44     DEFENCE    01-Jan-15   24.5        26.1        4089   ₹6,801        +6.8%     123d  
  IBULHSGFIN  43     FIN SVC    02-Feb-15   382.5       377.4       282    ₹-1,441       -1.3%     91d   
  SHREECEM    54     INFRA      02-Feb-15   10,437.7    9,898.1     10     ₹-5,397       -5.2%     91d   
  EMAMILTD    52     FMCG       01-Apr-15   420.8       383.0       264    ₹-9,974       -9.0%     33d   

  ENTRIES (4)
  [52w filter blocked 1: BBTC(-20.5%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  UPL         5      MFG        2.943    0.10   +133.9%   +41.8%    297.6       348    ₹103,581      +9.6%     
  GSKCONS     12     FMCG       2.282    -0.11  +56.7%    +13.7%    6,024.6     17     ₹102,417      +1.5%     
  CONCOR      14     PSE        2.265    0.06   +86.4%    +20.8%    484.4       214    ₹103,664      +5.0%     
  MARICO      16     FMCG       2.148    -0.23  +89.5%    +10.4%    166.1       624    ₹103,617      -1.6%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BHARATFORG  2      DEFENCE    01-Jan-15   431.4       584.5       231    ₹35,356       +35.5%      +2.1%     
  BOSCHLTD    31     AUTO       01-Jan-15   17,477.5    20,405.3    5      ₹14,639       +16.8%      -4.6%     
  PAGEIND     7      MFG        01-Jan-15   11,008.1    12,517.7    9      ₹13,587       +13.7%      +4.0%     
  HONAUT      13     MFG        01-Jan-15   6,711.0     7,527.8     14     ₹11,436       +12.2%      -3.5%     
  SRF         32     MFG        01-Jan-15   165.2       184.3       605    ₹11,543       +11.6%      -3.8%     
  WHIRLPOOL   10     CON DUR    01-Jan-15   637.1       693.4       156    ₹8,782        +8.8%       -1.7%     
  AJANTPHARM  1      HEALTH     01-Apr-15   737.1       799.4       151    ₹9,408        +8.5%       +1.0%     
  ASHOKLEY    11     AUTO       02-Feb-15   26.9        29.0        4016   ₹8,589        +7.9%       +1.0%     
  BRITANNIA   4      FMCG       01-Apr-15   946.0       976.7       117    ₹3,601        +3.3%       +3.2%     
  SIEMENS     9      ENERGY     02-Mar-15   731.1       732.9       152    ₹270          +0.2%       +4.5%     
  SPARC       38     HEALTH     02-Mar-15   417.9       402.5       266    ₹-4,108       -3.7%       -8.7%     
  LUPIN       6      HEALTH     01-Apr-15   1,917.4     1,707.9     58     ₹-12,153      -10.9%      -0.5%     
  SADBHAV     21     INFRA      01-Apr-15   342.4       298.7       325    ₹-14,202      -12.8%      -6.7%     
  WOCKPHARMA  25     HEALTH     02-Mar-15   1,510.0     1,303.2     73     ₹-15,093      -13.7%      -10.0%    
  NCC         15     INFRA      01-Apr-15   97.5        83.4        1142   ₹-16,117      -14.5%      -2.9%     

  AFTER: Invested ₹2,044,631 | Cash ₹29,147 | Total ₹2,073,778 | Positions 19/20 | Slot ₹103,713

========================================================================
  REBALANCE #06  —  01 Jun 2015
  NAV: ₹2,164,292  |  Slot: ₹108,215  |  Cash: ₹29,147
========================================================================
  [SECTOR CAP≤4] dropped: APLLTD, AUROPHARMA, ZYDUSLIFE

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BOSCHLTD    58     AUTO       01-Jan-15   17,477.5    20,961.6    5      ₹17,421       +19.9%    151d  
  HONAUT      97     MFG        01-Jan-15   6,711.0     7,173.1     14     ₹6,470        +6.9%     151d  
  GSKCONS     —      FMCG       04-May-15   6,024.6     5,941.3     17     ₹-1,415       -1.4%     28d   
  SIEMENS     63     ENERGY     02-Mar-15   731.1       713.3       152    ₹-2,710       -2.4%     91d   
  WOCKPHARMA  70     HEALTH     02-Mar-15   1,510.0     1,341.2     73     ₹-12,321      -11.2%    91d   
  SADBHAV     98     INFRA      01-Apr-15   342.4       280.9       325    ₹-19,994      -18.0%    61d   
  NCC         72     INFRA      01-Apr-15   97.5        72.8        1142   ₹-28,255      -25.4%    61d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  NATCOPHARM  4      HEALTH     3.504    0.69   +212.4%   +63.3%    407.7       265    ₹108,035      -0.5%     
  EICHERMOT   5      AUTO       3.072    0.15   +196.3%   +20.8%    1,740.7     62     ₹107,921      +5.2%     
  EMAMILTD    8      FMCG       2.779    0.26   +167.6%   +15.0%    479.2       225    ₹107,814      +9.5%     
  BBTC        10     FMCG       2.696    0.41   +358.5%   +21.0%    500.5       216    ₹108,106      +10.3%    
  INDUSTOWER  12     INFRA      2.608    -0.02  +107.5%   +26.6%    327.7       330    ₹108,127      +9.6%     
  DCBBANK     17     PVT BNK    2.272    0.11   +99.8%    +22.6%    124.9       866    ₹108,173      +3.6%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  PAGEIND     1      MFG        01-Jan-15   11,008.1    15,087.9    9      ₹36,718       +37.1%      +14.7%    
  AJANTPHARM  2      HEALTH     01-Apr-15   737.1       976.8       151    ₹36,194       +32.5%      +15.1%    
  SRF         14     MFG        01-Jan-15   165.2       216.0       605    ₹30,716       +30.7%      +12.8%    
  BHARATFORG  28     DEFENCE    01-Jan-15   431.4       560.0       231    ₹29,698       +29.8%      -2.3%     
  BRITANNIA   3      FMCG       01-Apr-15   946.0       1,115.5     117    ₹19,840       +17.9%      +6.8%     
  MARICO      6      FMCG       04-May-15   166.1       190.3       624    ₹15,151       +14.6%      +8.3%     
  WHIRLPOOL   13     CON DUR    01-Jan-15   637.1       729.2       156    ₹14,362       +14.5%      +2.7%     
  UPL         15     MFG        04-May-15   297.6       325.1       348    ₹9,563        +9.2%       +4.3%     
  ASHOKLEY    26     AUTO       02-Feb-15   26.9        29.3        4016   ₹9,652        +8.9%       +2.1%     
  CONCOR      35     PSE        04-May-15   484.4       501.2       214    ₹3,589        +3.5%       +2.3%     
  SPARC       36     HEALTH     02-Mar-15   417.9       398.2       266    ₹-5,231       -4.7%       -2.5%     
  LUPIN       16     HEALTH     01-Apr-15   1,917.4     1,686.5     58     ₹-13,397      -12.0%      +1.8%     

  AFTER: Invested ₹2,096,369 | Cash ₹67,153 | Total ₹2,163,522 | Positions 18/20 | Slot ₹108,215

========================================================================
  REBALANCE #07  —  01 Jul 2015
  NAV: ₹2,170,541  |  Slot: ₹108,527  |  Cash: ₹67,153
========================================================================
  [SECTOR CAP≤4] dropped: TORNTPHARM

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BHARATFORG  71     DEFENCE    01-Jan-15   431.4       511.3       231    ₹18,446       +18.5%    181d  
  ASHOKLEY    37     AUTO       02-Feb-15   26.9        30.4        4016   ₹14,016       +13.0%    149d  
  CONCOR      43     PSE        04-May-15   484.4       485.3       214    ₹193          +0.2%     58d   
  DCBBANK     29     PVT BNK    01-Jun-15   124.9       122.6       866    ₹-2,036       -1.9%     30d   
  SPARC       73     HEALTH     02-Mar-15   417.9       394.2       266    ₹-6,314       -5.7%     121d  
  LUPIN       42     HEALTH     01-Apr-15   1,917.4     1,764.1     58     ₹-8,892       -8.0%     91d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BAJFINANCE  2      FIN SVC    3.679    0.25   +162.1%   +37.5%    52.7        2058   ₹108,481      +8.7%     
  APLLTD      3      HEALTH     3.447    0.22   +145.7%   +47.6%    588.6       184    ₹108,307      +7.0%     
  RAJESHEXPO  4      CON DUR    3.336    0.56   +141.1%   +79.5%    349.9       310    ₹108,469      +26.7%    
  GLENMARK    9      HEALTH     2.618    0.51   +77.6%    +29.9%    979.1       110    ₹107,701      +7.1%     
  OFSS        10     IT         2.485    0.08   +50.7%    +18.2%    2,319.9     46     ₹106,714      +4.3%     
  MARUTI      14     AUTO       2.300    0.18   +66.3%    +10.1%    3,659.1     29     ₹106,115      +3.1%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  SRF         8      MFG        01-Jan-15   165.2       234.3       605    ₹41,842       +41.9%      +10.4%    
  AJANTPHARM  6      HEALTH     01-Apr-15   737.1       958.1       151    ₹33,377       +30.0%      +5.0%     
  BRITANNIA   1      FMCG       01-Apr-15   946.0       1,186.6     117    ₹28,158       +25.4%      +2.5%     
  PAGEIND     12     MFG        01-Jan-15   11,008.1    13,490.9    9      ₹22,345       +22.6%      +1.2%     
  MARICO      11     FMCG       04-May-15   166.1       191.3       624    ₹15,765       +15.2%      +2.9%     
  WHIRLPOOL   22     CON DUR    01-Jan-15   637.1       728.9       156    ₹14,324       +14.4%      +1.9%     
  UPL         18     MFG        04-May-15   297.6       327.6       348    ₹10,428       +10.1%      +1.9%     
  EICHERMOT   5      AUTO       01-Jun-15   1,740.7     1,886.6     62     ₹9,049        +8.4%       +7.4%     
  NATCOPHARM  24     HEALTH     01-Jun-15   407.7       432.1       265    ₹6,484        +6.0%       +7.7%     
  EMAMILTD    7      FMCG       01-Jun-15   479.2       500.2       225    ₹4,723        +4.4%       +8.3%     
  BBTC        16     FMCG       01-Jun-15   500.5       495.7       216    ₹-1,036       -1.0%       +4.3%     
  INDUSTOWER  23     INFRA      01-Jun-15   327.7       306.5       330    ₹-6,995       -6.5%       -2.3%     

  AFTER: Invested ₹2,091,828 | Cash ₹77,947 | Total ₹2,169,774 | Positions 18/20 | Slot ₹108,527

========================================================================
  REBALANCE #08  —  03 Aug 2015
  NAV: ₹2,321,714  |  Slot: ₹116,086  |  Cash: ₹77,947
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PAGEIND     81     MFG        01-Jan-15   11,008.1    12,198.5    9      ₹10,713       +10.8%    214d  
  UPL         60     MFG        04-May-15   297.6       327.1       348    ₹10,235       +9.9%     91d   
  NATCOPHARM  50     HEALTH     01-Jun-15   407.7       425.0       265    ₹4,601        +4.3%     63d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HINDPETRO   4      OIL&GAS    3.048    0.44   +142.5%   +53.0%    81.6        1422   ₹116,029      +3.8%     
  IBULHSGFIN  10     FIN SVC    2.521    0.26   +106.7%   +32.1%    498.5       232    ₹115,659      +9.1%     
  BAJAJFINSV  11     FIN SVC    2.481    0.19   +100.6%   +32.5%    189.4       612    ₹115,920      +9.3%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RAJESHEXPO  1      CON DUR    01-Jul-15   349.9       561.3       310    ₹65,528       +60.4%      +23.9%    
  SRF         9      MFG        01-Jan-15   165.2       262.4       605    ₹58,843       +58.9%      +5.9%     
  BRITANNIA   2      FMCG       01-Apr-15   946.0       1,370.3     117    ₹49,643       +44.9%      +7.7%     
  AJANTPHARM  23     HEALTH     01-Apr-15   737.1       938.5       151    ₹30,408       +27.3%      -1.9%     
  EMAMILTD    3      FMCG       01-Jun-15   479.2       562.9       225    ₹18,845       +17.5%      +9.9%     
  BBTC        8      FMCG       01-Jun-15   500.5       585.5       216    ₹18,359       +17.0%      +11.6%    
  WHIRLPOOL   39     CON DUR    01-Jan-15   637.1       738.6       156    ₹15,830       +15.9%      +2.4%     
  APLLTD      6      HEALTH     01-Jul-15   588.6       657.5       184    ₹12,681       +11.7%      +0.6%     
  MARICO      30     FMCG       04-May-15   166.1       184.9       624    ₹11,737       +11.3%      -0.3%     
  MARUTI      7      AUTO       01-Jul-15   3,659.1     4,018.0     29     ₹10,406       +9.8%       +5.9%     
  OFSS        15     IT         01-Jul-15   2,319.9     2,481.5     46     ₹7,436        +7.0%       +5.4%     
  BAJFINANCE  5      FIN SVC    01-Jul-15   52.7        54.7        2058   ₹4,021        +3.7%       +7.3%     
  INDUSTOWER  29     INFRA      01-Jun-15   327.7       334.8       330    ₹2,361        +2.2%       +7.3%     
  EICHERMOT   13     AUTO       01-Jun-15   1,740.7     1,769.0     62     ₹1,754        +1.6%       -4.1%     
  GLENMARK    40     HEALTH     01-Jul-15   979.1       951.8       110    ₹-2,999       -2.8%       -2.4%     

  AFTER: Invested ₹2,255,137 | Cash ₹66,164 | Total ₹2,321,301 | Positions 18/20 | Slot ₹116,086

========================================================================
  REBALANCE #09  —  01 Sep 2015
  NAV: ₹2,125,148  |  Slot: ₹106,257  |  Cash: ₹66,164
========================================================================
  [SECTOR CAP≤4] dropped: ABBOTINDIA, DIVISLAB, DRREDDY

  [REGIME OFF] Nifty 200 4,097.9 < SMA200 4,382.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RAJESHEXPO  1      CON DUR    01-Jul-15   349.9       474.0       310    ₹38,461       +35.5%      -8.2%     
  BRITANNIA   3      FMCG       01-Apr-15   946.0       1,275.9     117    ₹38,607       +34.9%      -4.7%     
  SRF         35     MFG        01-Jan-15   165.2       217.0       605    ₹31,345       +31.4%      -8.3%     
  AJANTPHARM  23     HEALTH     01-Apr-15   737.1       921.5       151    ₹27,841       +25.0%      -1.8%     
  GLENMARK    9      HEALTH     01-Jul-15   979.1       1,081.4     110    ₹11,256       +10.5%      -0.6%     
  OFSS        15     IT         01-Jul-15   2,319.9     2,471.8     46     ₹6,987        +6.5%       -1.0%     
  APLLTD      12     HEALTH     01-Jul-15   588.6       624.2       184    ₹6,538        +6.0%       -1.2%     
  MARICO      51     FMCG       04-May-15   166.1       173.1       624    ₹4,375        +4.2%       -3.0%     
  EMAMILTD    18     FMCG       01-Jun-15   479.2       496.6       225    ₹3,925        +3.6%       -5.3%     
  WHIRLPOOL   71     CON DUR    01-Jan-15   637.1       649.8       156    ₹1,984        +2.0%       -6.7%     
  MARUTI      16     AUTO       01-Jul-15   3,659.1     3,708.2     29     ₹1,423        +1.3%       -5.6%     
  EICHERMOT   25     AUTO       01-Jun-15   1,740.7     1,692.6     62     ₹-2,981       -2.8%       -6.6%     
  IBULHSGFIN  6      FIN SVC    03-Aug-15   498.5       479.8       232    ₹-4,353       -3.8%       -4.1%     
  BAJAJFINSV  13     FIN SVC    03-Aug-15   189.4       176.2       612    ₹-8,059       -7.0%       -2.9%     
  BAJFINANCE  10     FIN SVC    01-Jul-15   52.7        47.5        2058   ₹-10,775      -9.9%       -4.7%     
  HINDPETRO   17     OIL&GAS    03-Aug-15   81.6        73.1        1422   ₹-12,035      -10.4%      -4.5%     
  BBTC        40     FMCG       01-Jun-15   500.5       447.1       216    ₹-11,527      -10.7%      -11.6%    
  INDUSTOWER  68     INFRA      01-Jun-15   327.7       284.3       330    ₹-14,314      -13.2%      -1.9%     

  AFTER: Invested ₹2,058,984 | Cash ₹66,164 | Total ₹2,125,148 | Positions 18/20 | Slot ₹106,257

========================================================================
  REBALANCE #10  —  01 Oct 2015
  NAV: ₹2,177,399  |  Slot: ₹108,870  |  Cash: ₹66,164
========================================================================
  [SECTOR CAP≤4] dropped: TORNTPHARM, ZYDUSLIFE, DRREDDY

  [REGIME OFF] Nifty 200 4,179.2 < SMA200 4,359.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RAJESHEXPO  2      CON DUR    01-Jul-15   349.9       548.4       310    ₹61,549       +56.7%      +12.6%    
  BRITANNIA   3      FMCG       01-Apr-15   946.0       1,364.1     117    ₹48,925       +44.2%      +4.4%     
  SRF         66     MFG        01-Jan-15   165.2       215.0       605    ₹30,125       +30.1%      -1.2%     
  AJANTPHARM  23     HEALTH     01-Apr-15   737.1       913.9       151    ₹26,700       +24.0%      +3.4%     
  MARUTI      6      AUTO       01-Jul-15   3,659.1     4,182.0     29     ₹15,163       +14.3%      +3.1%     
  OFSS        45     IT         01-Jul-15   2,319.9     2,482.2     46     ₹7,467        +7.0%       +2.3%     
  APLLTD      24     HEALTH     01-Jul-15   588.6       626.0       184    ₹6,883        +6.4%       +2.5%     
  IBULHSGFIN  5      FIN SVC    03-Aug-15   498.5       524.9       232    ₹6,124        +5.3%       +7.0%     
  WHIRLPOOL   55     CON DUR    01-Jan-15   637.1       664.0       156    ₹4,189        +4.2%       +4.3%     
  EMAMILTD    35     FMCG       01-Jun-15   479.2       496.1       225    ₹3,820        +3.5%       +0.6%     
  MARICO      69     FMCG       04-May-15   166.1       170.9       624    ₹3,041        +2.9%       -0.5%     
  GLENMARK    46     HEALTH     01-Jul-15   979.1       999.3       110    ₹2,221        +2.1%       +0.3%     
  EICHERMOT   65     AUTO       01-Jun-15   1,740.7     1,695.1     62     ₹-2,825       -2.6%       +0.2%     
  BBTC        49     FMCG       01-Jun-15   500.5       468.5       216    ₹-6,901       -6.4%       +2.3%     
  BAJFINANCE  25     FIN SVC    01-Jul-15   52.7        49.1        2058   ₹-7,359       -6.8%       +1.3%     
  BAJAJFINSV  19     FIN SVC    03-Aug-15   189.4       175.2       612    ₹-8,668       -7.5%       -1.1%     
  HINDPETRO   31     OIL&GAS    03-Aug-15   81.6        71.8        1422   ₹-13,864      -11.9%      -1.0%     
  INDUSTOWER  102    INFRA      01-Jun-15   327.7       280.3       330    ₹-15,640      -14.5% ⚠    +1.8%     
  ⚠  WAZ < 0 (momentum below universe mean): INDUSTOWER

  AFTER: Invested ₹2,111,235 | Cash ₹66,164 | Total ₹2,177,399 | Positions 18/20 | Slot ₹108,870

========================================================================
  REBALANCE #11  —  02 Nov 2015
  NAV: ₹2,200,548  |  Slot: ₹110,027  |  Cash: ₹66,164
========================================================================
  [SECTOR CAP≤4] dropped: LUPIN, TORNTPHARM, DRREDDY

  [REGIME OFF] Nifty 200 4,227.9 < SMA200 4,364.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RAJESHEXPO  1      CON DUR    01-Jul-15   349.9       634.2       310    ₹88,144       +81.3%      +3.9%     
  SRF         29     MFG        01-Jan-15   165.2       252.7       605    ₹52,930       +53.0%      +3.7%     
  BRITANNIA   3      FMCG       01-Apr-15   946.0       1,392.0     117    ₹52,184       +47.1%      +0.0%     
  AJANTPHARM  20     HEALTH     01-Apr-15   737.1       924.3       151    ₹28,266       +25.4%      -0.4%     
  MARUTI      16     AUTO       01-Jul-15   3,659.1     4,104.7     29     ₹12,923       +12.2%      +1.5%     
  OFSS        26     IT         01-Jul-15   2,319.9     2,489.5     46     ₹7,804        +7.3%       +0.6%     
  BAJAJFINSV  7      FIN SVC    03-Aug-15   189.4       196.4       612    ₹4,281        +3.7%       +5.6%     
  APLLTD      48     HEALTH     01-Jul-15   588.6       606.7       184    ₹3,327        +3.1%       -2.1%     
  MARICO      100    FMCG       04-May-15   166.1       165.0       624    ₹-667         -0.6% ⚠     -2.0%     
  WHIRLPOOL   91     CON DUR    01-Jan-15   637.1       619.9       156    ₹-2,676       -2.7% ⚠     -4.3%     
  IBULHSGFIN  34     FIN SVC    03-Aug-15   498.5       482.8       232    ₹-3,640       -3.1%       -1.3%     
  GLENMARK    40     HEALTH     01-Jul-15   979.1       948.2       110    ₹-3,402       -3.2%       -2.4%     
  BBTC        59     FMCG       01-Jun-15   500.5       480.2       216    ₹-4,372       -4.0%       -1.2%     
  EICHERMOT   51     AUTO       01-Jun-15   1,740.7     1,666.3     62     ₹-4,612       -4.3%       -0.2%     
  BAJFINANCE  21     FIN SVC    01-Jul-15   52.7        50.2        2058   ₹-5,070       -4.7%       +1.2%     
  EMAMILTD    141    FMCG       01-Jun-15   479.2       433.1       225    ₹-10,371      -9.6% ⚠     -6.9%     
  HINDPETRO   81     OIL&GAS    03-Aug-15   81.6        72.5        1422   ₹-12,969      -11.2%      +0.1%     
  INDUSTOWER  92     INFRA      01-Jun-15   327.7       273.2       330    ₹-17,979      -16.6% ⚠    -1.9%     
  ⚠  WAZ < 0 (momentum below universe mean): WHIRLPOOL, INDUSTOWER, MARICO, EMAMILTD

  AFTER: Invested ₹2,134,384 | Cash ₹66,164 | Total ₹2,200,548 | Positions 18/20 | Slot ₹110,027

========================================================================
  REBALANCE #12  —  01 Dec 2015
  NAV: ₹2,205,819  |  Slot: ₹110,291  |  Cash: ₹66,164
========================================================================

  [REGIME OFF] Nifty 200 4,198.9 < SMA200 4,335.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RAJESHEXPO  1      CON DUR    01-Jul-15   349.9       698.5       310    ₹108,055      +99.6%      +4.8%     
  SRF         37     MFG        01-Jan-15   165.2       240.6       605    ₹45,649       +45.7%      -0.3%     
  BRITANNIA   14     FMCG       01-Apr-15   946.0       1,298.0     117    ₹41,190       +37.2%      -0.8%     
  MARUTI      16     AUTO       01-Jul-15   3,659.1     4,159.9     29     ₹14,523       +13.7%      -0.5%     
  AJANTPHARM  83     HEALTH     01-Apr-15   737.1       814.9       151    ₹11,752       +10.6%      -3.0%     
  MARICO      38     FMCG       04-May-15   166.1       182.2       624    ₹10,097       +9.7%       +2.7%     
  APLLTD      27     HEALTH     01-Jul-15   588.6       629.3       184    ₹7,491        +6.9%       +3.8%     
  OFSS        58     IT         01-Jul-15   2,319.9     2,470.1     46     ₹6,908        +6.5%       +0.6%     
  BAJAJFINSV  5      FIN SVC    03-Aug-15   189.4       198.0       612    ₹5,239        +4.5%       +2.7%     
  WHIRLPOOL   112    CON DUR    01-Jan-15   637.1       656.8       156    ₹3,079        +3.1% ⚠     +0.7%     
  BAJFINANCE  7      FIN SVC    01-Jul-15   52.7        53.9        2058   ₹2,466        +2.3%       +3.7%     
  GLENMARK    138    HEALTH     01-Jul-15   979.1       947.6       110    ₹-3,461       -3.2% ⚠     +0.9%     
  IBULHSGFIN  25     FIN SVC    03-Aug-15   498.5       480.1       232    ₹-4,284       -3.7%       +4.9%     
  HINDPETRO   22     OIL&GAS    03-Aug-15   81.6        78.4        1422   ₹-4,548       -3.9%       +6.0%     
  BBTC        87     FMCG       01-Jun-15   500.5       475.3       216    ₹-5,432       -5.0%       -0.8%     
  EICHERMOT   137    AUTO       01-Jun-15   1,740.7     1,505.2     62     ₹-14,601      -13.5% ⚠    -2.6%     
  INDUSTOWER  86     INFRA      01-Jun-15   327.7       275.7       330    ₹-17,142      -15.9%      +0.8%     
  EMAMILTD    145    FMCG       01-Jun-15   479.2       400.9       225    ₹-17,612      -16.3% ⚠    -4.8%     
  ⚠  WAZ < 0 (momentum below universe mean): WHIRLPOOL, EICHERMOT, GLENMARK, EMAMILTD

  AFTER: Invested ₹2,139,655 | Cash ₹66,164 | Total ₹2,205,819 | Positions 18/20 | Slot ₹110,291

========================================================================
  REBALANCE #13  —  01 Jan 2016
  NAV: ₹2,223,555  |  Slot: ₹111,178  |  Cash: ₹66,164
========================================================================

  [REGIME OFF] Nifty 200 4,219.0 < SMA200 4,290.6 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RAJESHEXPO  1      CON DUR    01-Jul-15   349.9       671.0       310    ₹99,548       +91.8%      -0.2%     
  SRF         31     MFG        01-Jan-15   165.2       240.3       605    ₹45,470       +45.5%      +1.5%     
  BRITANNIA   14     FMCG       01-Apr-15   946.0       1,304.5     117    ₹41,954       +37.9%      +1.9%     
  MARICO      17     FMCG       04-May-15   166.1       195.1       624    ₹18,094       +17.5%      +2.8%     
  MARUTI      30     AUTO       01-Jul-15   3,659.1     4,235.1     29     ₹16,702       +15.7%      +0.7%     
  BAJFINANCE  4      FIN SVC    01-Jul-15   52.7        58.6        2058   ₹12,207       +11.3%      +5.2%     
  AJANTPHARM  87     HEALTH     01-Apr-15   737.1       805.7       151    ₹10,355       +9.3%       +0.7%     
  APLLTD      35     HEALTH     01-Jul-15   588.6       614.0       184    ₹4,666        +4.3%       +0.1%     
  BAJAJFINSV  13     FIN SVC    03-Aug-15   189.4       197.2       612    ₹4,792        +4.1%       +2.1%     
  OFSS        84     IT         01-Jul-15   2,319.9     2,399.2     46     ₹3,648        +3.4%       +0.2%     
  WHIRLPOOL   111    CON DUR    01-Jan-15   637.1       633.6       156    ₹-540         -0.5% ⚠     -0.6%     
  IBULHSGFIN  45     FIN SVC    03-Aug-15   498.5       493.4       232    ₹-1,195       -1.0%       +4.7%     
  HINDPETRO   26     OIL&GAS    03-Aug-15   81.6        79.0        1422   ₹-3,680       -3.2%       +2.6%     
  BBTC        77     FMCG       01-Jun-15   500.5       478.6       216    ₹-4,729       -4.4%       +3.3%     
  EICHERMOT   90     AUTO       01-Jun-15   1,740.7     1,610.2     62     ₹-8,087       -7.5% ⚠     +5.6%     
  INDUSTOWER  52     INFRA      01-Jun-15   327.7       301.2       330    ₹-8,740       -8.1%       +4.6%     
  GLENMARK    120    HEALTH     01-Jul-15   979.1       890.1       110    ₹-9,788       -9.1% ⚠     -0.1%     
  EMAMILTD    117    FMCG       01-Jun-15   479.2       418.9       225    ₹-13,572      -12.6% ⚠    -0.1%     
  ⚠  WAZ < 0 (momentum below universe mean): EICHERMOT, WHIRLPOOL, EMAMILTD, GLENMARK

  AFTER: Invested ₹2,157,390 | Cash ₹66,164 | Total ₹2,223,555 | Positions 18/20 | Slot ₹111,178

========================================================================
  REBALANCE #14  —  01 Feb 2016
  NAV: ₹2,124,062  |  Slot: ₹106,203  |  Cash: ₹66,164
========================================================================

  [REGIME OFF] Nifty 200 3,969.2 < SMA200 4,240.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RAJESHEXPO  1      CON DUR    01-Jul-15   349.9       720.9       310    ₹114,998      +106.0%     +2.5%     
  SRF         64     MFG        01-Jan-15   165.2       221.6       605    ₹34,102       +34.1%      -1.6%     
  BRITANNIA   36     FMCG       01-Apr-15   946.0       1,223.3     117    ₹32,453       +29.3%      +1.0%     
  MARICO      6      FMCG       04-May-15   166.1       194.1       624    ₹17,478       +16.9%      +2.2%     
  BAJFINANCE  3      FIN SVC    01-Jul-15   52.7        57.8        2058   ₹10,372       +9.6%       +2.0%     
  AJANTPHARM  65     HEALTH     01-Apr-15   737.1       786.7       151    ₹7,482        +6.7%       +4.6%     
  WHIRLPOOL   63     CON DUR    01-Jan-15   637.1       636.4       156    ₹-114         -0.1%       +4.5%     
  OFSS        71     IT         01-Jul-15   2,319.9     2,302.4     46     ₹-805         -0.8%       -0.3%     
  MARUTI      97     AUTO       01-Jul-15   3,659.1     3,604.9     29     ₹-1,574       -1.5% ⚠     -6.0%     
  BAJAJFINSV  24     FIN SVC    03-Aug-15   189.4       185.3       612    ₹-2,525       -2.2%       -1.9%     
  IBULHSGFIN  26     FIN SVC    03-Aug-15   498.5       480.1       232    ₹-4,285       -3.7%       +1.6%     
  APLLTD      38     HEALTH     01-Jul-15   588.6       560.0       184    ₹-5,262       -4.9%       -1.1%     
  HINDPETRO   13     OIL&GAS    03-Aug-15   81.6        74.6        1422   ₹-9,903       -8.5%       -2.4%     
  EMAMILTD    44     FMCG       01-Jun-15   479.2       437.8       225    ₹-9,318       -8.6%       +3.4%     
  EICHERMOT   50     AUTO       01-Jun-15   1,740.7     1,582.7     62     ₹-9,794       -9.1%       +3.8%     
  BBTC        110    FMCG       01-Jun-15   500.5       406.7       216    ₹-20,265      -18.7% ⚠    -2.2%     
  INDUSTOWER  69     INFRA      01-Jun-15   327.7       260.7       330    ₹-22,099      -20.4%      -3.0%     
  GLENMARK    106    HEALTH     01-Jul-15   979.1       767.0       110    ₹-23,329      -21.7% ⚠    -1.7%     
  ⚠  WAZ < 0 (momentum below universe mean): MARUTI, GLENMARK, BBTC

  AFTER: Invested ₹2,057,897 | Cash ₹66,164 | Total ₹2,124,062 | Positions 18/20 | Slot ₹106,203

========================================================================
  REBALANCE #15  —  01 Mar 2016
  NAV: ₹2,053,448  |  Slot: ₹102,672  |  Cash: ₹66,164
========================================================================

  [REGIME OFF] Nifty 200 3,779.9 < SMA200 4,179.7 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RAJESHEXPO  1      CON DUR    01-Jul-15   349.9       697.5       310    ₹107,748      +99.3%      -1.1%     
  BRITANNIA   12     FMCG       01-Apr-15   946.0       1,241.3     117    ₹34,559       +31.2%      +3.4%     
  SRF         39     MFG        01-Jan-15   165.2       210.5       605    ₹27,414       +27.4%      +0.3%     
  MARICO      4      FMCG       04-May-15   166.1       203.8       624    ₹23,549       +22.7%      +1.6%     
  BAJFINANCE  2      FIN SVC    01-Jul-15   52.7        59.7        2058   ₹14,445       +13.3%      +2.9%     
  AJANTPHARM  21     HEALTH     01-Apr-15   737.1       800.7       151    ₹9,597        +8.6%       +2.6%     
  EICHERMOT   6      AUTO       01-Jun-15   1,740.7     1,778.3     62     ₹2,336        +2.2%       +4.3%     
  WHIRLPOOL   66     CON DUR    01-Jan-15   637.1       608.8       156    ₹-4,410       -4.4%       +2.6%     
  APLLTD      18     HEALTH     01-Jul-15   588.6       555.0       184    ₹-6,182       -5.7%       +1.0%     
  OFSS        99     IT         01-Jul-15   2,319.9     2,068.8     46     ₹-11,549      -10.8% ⚠    -1.7%     
  BAJAJFINSV  52     FIN SVC    03-Aug-15   189.4       168.1       612    ₹-13,067      -11.3%      -0.8%     
  EMAMILTD    23     FMCG       01-Jun-15   479.2       423.4       225    ₹-12,544      -11.6%      -0.1%     
  MARUTI      136    AUTO       01-Jul-15   3,659.1     3,190.5     29     ₹-13,591      -12.8% ⚠    -3.3%     
  IBULHSGFIN  48     FIN SVC    03-Aug-15   498.5       422.7       232    ₹-17,594      -15.2%      +1.8%     
  HINDPETRO   36     OIL&GAS    03-Aug-15   81.6        67.8        1422   ₹-19,586      -16.9%      +2.7%     
  INDUSTOWER  34     INFRA      01-Jun-15   327.7       264.2       330    ₹-20,947      -19.4%      +1.8%     
  GLENMARK    106    HEALTH     01-Jul-15   979.1       733.3       110    ₹-27,034      -25.1% ⚠    +2.1%     
  BBTC        132    FMCG       01-Jun-15   500.5       333.1       216    ₹-36,146      -33.4% ⚠    -4.0%     
  ⚠  WAZ < 0 (momentum below universe mean): OFSS, GLENMARK, BBTC, MARUTI

  AFTER: Invested ₹1,987,284 | Cash ₹66,164 | Total ₹2,053,448 | Positions 18/20 | Slot ₹102,672

========================================================================
  REBALANCE #16  —  01 Apr 2016
  NAV: ₹2,110,521  |  Slot: ₹105,526  |  Cash: ₹66,164
========================================================================

  [REGIME OFF] Nifty 200 4,043.2 < SMA200 4,144.1 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RAJESHEXPO  4      CON DUR    01-Jul-15   349.9       615.3       310    ₹82,270       +75.8%      -1.8%     
  SRF         27     MFG        01-Jan-15   165.2       244.6       605    ₹48,023       +48.1%      +4.6%     
  MARICO      16     FMCG       04-May-15   166.1       210.1       624    ₹27,467       +26.5%      +0.4%     
  BAJFINANCE  3      FIN SVC    01-Jul-15   52.7        66.3        2058   ₹28,062       +25.9%      +4.6%     
  BRITANNIA   61     FMCG       01-Apr-15   946.0       1,170.3     117    ₹26,246       +23.7%      -1.3%     
  AJANTPHARM  29     HEALTH     01-Apr-15   737.1       847.5       151    ₹16,675       +15.0%      +1.7%     
  WHIRLPOOL   39     CON DUR    01-Jan-15   637.1       681.6       156    ₹6,949        +7.0%       +5.2%     
  EICHERMOT   13     AUTO       01-Jun-15   1,740.7     1,794.6     62     ₹3,343        +3.1%       +2.5%     
  OFSS        60     IT         01-Jul-15   2,319.9     2,225.8     46     ₹-4,326       -4.1%       +2.1%     
  MARUTI      146    AUTO       01-Jul-15   3,659.1     3,399.4     29     ₹-7,533       -7.1% ⚠     +1.5%     
  HINDPETRO   34     OIL&GAS    03-Aug-15   81.6        75.4        1422   ₹-8,864       -7.6%       +5.7%     
  BAJAJFINSV  52     FIN SVC    03-Aug-15   189.4       173.9       612    ₹-9,520       -8.2%       +3.5%     
  APLLTD      75     HEALTH     01-Jul-15   588.6       529.0       184    ₹-10,970      -10.1%      -3.0%     
  IBULHSGFIN  65     FIN SVC    03-Aug-15   498.5       443.6       232    ₹-12,740      -11.0%      +1.7%     
  INDUSTOWER  91     INFRA      01-Jun-15   327.7       271.9       330    ₹-18,387      -17.0% ⚠    +2.0%     
  EMAMILTD    106    FMCG       01-Jun-15   479.2       391.6       225    ₹-19,694      -18.3% ⚠    -2.4%     
  GLENMARK    124    HEALTH     01-Jul-15   979.1       763.4       110    ₹-23,727      -22.0% ⚠    -0.5%     
  BBTC        145    FMCG       01-Jun-15   500.5       365.3       216    ₹-29,202      -27.0% ⚠    +2.2%     
  ⚠  WAZ < 0 (momentum below universe mean): INDUSTOWER, EMAMILTD, GLENMARK, BBTC, MARUTI

  AFTER: Invested ₹2,044,357 | Cash ₹66,164 | Total ₹2,110,521 | Positions 18/20 | Slot ₹105,526

========================================================================
  REBALANCE #17  —  02 May 2016
  NAV: ₹2,200,565  |  Slot: ₹110,028  |  Cash: ₹66,164
========================================================================

  [REGIME OFF] Nifty 200 4,119.9 < SMA200 4,124.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RAJESHEXPO  9      CON DUR    01-Jul-15   349.9       587.1       310    ₹73,535       +67.8%      -0.2%     
  SRF         30     MFG        01-Jan-15   165.2       259.8       605    ₹57,234       +57.3%      +2.7%     
  MARICO      7      FMCG       04-May-15   166.1       232.1       624    ₹41,225       +39.8%      +5.7%     
  BRITANNIA   23     FMCG       01-Apr-15   946.0       1,248.6     117    ₹35,412       +32.0%      +2.5%     
  BAJFINANCE  4      FIN SVC    01-Jul-15   52.7        68.0        2058   ₹31,562       +29.1%      +1.5%     
  AJANTPHARM  27     HEALTH     01-Apr-15   737.1       915.7       151    ₹26,969       +24.2%      +2.0%     
  WHIRLPOOL   40     CON DUR    01-Jan-15   637.1       711.4       156    ₹11,595       +11.7%      +0.8%     
  EICHERMOT   16     AUTO       01-Jun-15   1,740.7     1,850.5     62     ₹6,812        +6.3%       +0.5%     
  BAJAJFINSV  36     FIN SVC    03-Aug-15   189.4       193.2       612    ₹2,306        +2.0%       +6.5%     
  OFSS        84     IT         01-Jul-15   2,319.9     2,293.9     46     ₹-1,194       -1.1% ⚠     +1.9%     
  HINDPETRO   44     OIL&GAS    03-Aug-15   81.6        79.9        1422   ₹-2,472       -2.1%       +1.7%     
  MARUTI      117    AUTO       01-Jul-15   3,659.1     3,497.7     29     ₹-4,682       -4.4% ⚠     +2.9%     
  IBULHSGFIN  74     FIN SVC    03-Aug-15   498.5       475.6       232    ₹-5,324       -4.6%       +4.6%     
  APLLTD      102    HEALTH     01-Jul-15   588.6       531.7       184    ₹-10,477      -9.7% ⚠     -0.4%     
  EMAMILTD    123    FMCG       01-Jun-15   479.2       414.9       225    ₹-14,454      -13.4% ⚠    -0.4%     
  GLENMARK    91     HEALTH     01-Jul-15   979.1       799.1       110    ₹-19,805      -18.4% ⚠    +2.8%     
  INDUSTOWER  104    INFRA      01-Jun-15   327.7       264.9       330    ₹-20,702      -19.1% ⚠    -0.0%     
  BBTC        124    FMCG       01-Jun-15   500.5       392.1       216    ₹-23,422      -21.7% ⚠    +2.3%     
  ⚠  WAZ < 0 (momentum below universe mean): OFSS, GLENMARK, APLLTD, INDUSTOWER, MARUTI, EMAMILTD, BBTC

  AFTER: Invested ₹2,134,401 | Cash ₹66,164 | Total ₹2,200,565 | Positions 18/20 | Slot ₹110,028

========================================================================
  REBALANCE #18  —  01 Jun 2016
  NAV: ₹2,189,331  |  Slot: ₹109,467  |  Cash: ₹66,164
========================================================================
  [SECTOR CAP≤4] dropped: BERGEPAINT

  EXITS (12)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MARICO      57     FMCG       04-May-15   166.1       218.4       624    ₹32,639       +31.5%    394d  
  BRITANNIA   72     FMCG       01-Apr-15   946.0       1,207.9     117    ₹30,644       +27.7%    427d  
  AJANTPHARM  48     HEALTH     01-Apr-15   737.1       930.8       151    ₹29,242       +26.3%    427d  
  EICHERMOT   88     AUTO       01-Jun-15   1,740.7     1,744.2     62     ₹218          +0.2%     366d  
  IBULHSGFIN  44     FIN SVC    03-Aug-15   498.5       498.5       232    ₹-14          -0.0%     303d  
  OFSS        —      IT         01-Jul-15   2,319.9     2,199.6     46     ₹-5,532       -5.2%     336d  
  BAJAJFINSV  43     FIN SVC    03-Aug-15   189.4       179.0       612    ₹-6,365       -5.5%     303d  
  EMAMILTD    —      FMCG       01-Jun-15   479.2       437.7       225    ₹-9,328       -8.7%     366d  
  GLENMARK    68     HEALTH     01-Jul-15   979.1       819.1       110    ₹-17,606      -16.3%    336d  
  INDUSTOWER  85     INFRA      01-Jun-15   327.7       271.5       330    ₹-18,538      -17.1%    366d  
  APLLTD      101    HEALTH     01-Jul-15   588.6       485.3       184    ₹-19,009      -17.6%    336d  
  BBTC        78     FMCG       01-Jun-15   500.5       370.2       216    ₹-28,153      -26.0%    366d  

  ENTRIES (12)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BIOCON      1      HEALTH     3.831    0.19   +60.9%    +57.9%    117.0       935    ₹109,436      +10.2%    
  VGUARD      2      CON DUR    3.607    0.26   +46.9%    +66.5%    93.1        1175   ₹109,414      +13.7%    
  RAMCOCEM    3      MFG        2.950    0.28   +62.3%    +35.7%    473.9       231    ₹109,461      +0.7%     
  INDUSINDBK  5      PVT BNK    2.645    0.41   +29.8%    +36.6%    1,042.1     105    ₹109,416      +2.9%     
  HDFC        7      PVT BNK    2.391    0.28   +15.7%    +24.7%    265.8       411    ₹109,228      +1.9%     
  HDFCBANK    8      PVT BNK    2.391    0.28   +15.7%    +24.7%    265.8       411    ₹109,228      +1.9%     
  SHRIRAMFIN  9      FIN SVC    2.237    0.68   +41.9%    +46.0%    197.6       553    ₹109,281      +3.9%     
  MUTHOOTFIN  10     FIN SVC    2.142    0.24   +35.5%    +40.6%    205.5       532    ₹109,310      +12.8%    
  WELSPUNIND  11     CONSUMP    2.085    0.54   +94.1%    +26.3%    97.9        1118   ₹109,401      +6.1%     
  PIDILITIND  12     MFG        2.056    0.41   +31.7%    +20.3%    342.9       319    ₹109,375      +9.8%     
  HAVELLS     13     CON DUR    2.052    0.46   +36.4%    +31.7%    331.1       330    ₹109,253      -0.7%     
  BPCL        14     OIL&GAS    1.896    0.37   +30.4%    +29.8%    89.0        1230   ₹109,439      +3.9%     

  HOLDS (6)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RAJESHEXPO  19     CON DUR    01-Jul-15   349.9       558.5       310    ₹64,677       +59.6%      -0.4%     
  SRF         33     MFG        01-Jan-15   165.2       246.2       605    ₹49,000       +49.0%      +0.5%     
  BAJFINANCE  4      FIN SVC    01-Jul-15   52.7        73.7        2058   ₹43,249       +39.9%      +1.7%     
  WHIRLPOOL   38     CON DUR    01-Jan-15   637.1       748.2       156    ₹17,335       +17.4%      +1.8%     
  HINDPETRO   6      OIL&GAS    03-Aug-15   81.6        86.0        1422   ₹6,249        +5.4%       +5.4%     
  MARUTI      27     AUTO       01-Jul-15   3,659.1     3,803.1     29     ₹4,174        +3.9%       +4.7%     

  AFTER: Invested ₹2,135,346 | Cash ₹52,427 | Total ₹2,187,773 | Positions 18/20 | Slot ₹109,467

========================================================================
  REBALANCE #19  —  01 Jul 2016
  NAV: ₹2,262,587  |  Slot: ₹113,129  |  Cash: ₹52,427
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SRF         63     MFG        01-Jan-15   165.2       251.2       605    ₹52,065       +52.1%    547d  
  WHIRLPOOL   36     CON DUR    01-Jan-15   637.1       804.4       156    ₹26,094       +26.3%    547d  
  RAJESHEXPO  77     CON DUR    01-Jul-15   349.9       439.5       310    ₹27,788       +25.6%    366d  
  MARUTI      44     AUTO       01-Jul-15   3,659.1     3,803.1     29     ₹4,176        +3.9%     366d  

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BAJAJFINSV  5      FIN SVC    2.932    0.51   +57.1%    +37.6%    234.4       482    ₹112,969      +13.5%    
  PETRONET    6      OIL&GAS    2.520    0.46   +63.7%    +19.1%    101.7       1112   ₹113,113      +4.8%     
  IGL         11     OIL&GAS    2.059    0.49   +58.0%    +8.1%     53.3        2123   ₹113,097      +1.8%     
  ASIANPAINT  13     CONSUMP    1.981    0.36   +34.1%    +16.1%    921.2       122    ₹112,387      +1.4%     
  NATCOPHARM  14     HEALTH     1.956    0.36   +31.0%    +37.8%    523.6       216    ₹113,094      +5.6%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  8      FIN SVC    01-Jul-15   52.7        78.8        2058   ₹53,668       +49.5%      +5.9%     
  HINDPETRO   9      OIL&GAS    03-Aug-15   81.6        97.0        1422   ₹21,841       +18.8%      +8.2%     
  MUTHOOTFIN  4      FIN SVC    01-Jun-16   205.5       241.6       532    ₹19,232       +17.6%      +6.8%     
  RAMCOCEM    3      MFG        01-Jun-16   473.9       552.2       231    ₹18,090       +16.5%      +4.7%     
  BPCL        12     OIL&GAS    01-Jun-16   89.0        99.9        1230   ₹13,445       +12.3%      +7.0%     
  SHRIRAMFIN  16     FIN SVC    01-Jun-16   197.6       208.5       553    ₹6,040        +5.5%       +5.5%     
  HAVELLS     28     CON DUR    01-Jun-16   331.1       343.4       330    ₹4,060        +3.7%       +2.2%     
  BIOCON      1      HEALTH     01-Jun-16   117.0       121.2       935    ₹3,852        +3.5%       +2.6%     
  VGUARD      2      CON DUR    01-Jun-16   93.1        95.9        1175   ₹3,274        +3.0%       +3.9%     
  INDUSINDBK  10     PVT BNK    01-Jun-16   1,042.1     1,065.0     105    ₹2,404        +2.2%       +2.3%     
  WELSPUNIND  24     CONSUMP    01-Jun-16   97.9        99.1        1118   ₹1,447        +1.3%       +0.7%     
  HDFC        25     PVT BNK    01-Jun-16   265.8       267.8       411    ₹836          +0.8%       +1.2%     
  HDFCBANK    26     PVT BNK    01-Jun-16   265.8       267.8       411    ₹836          +0.8%       +1.2%     
  PIDILITIND  7      MFG        01-Jun-16   342.9       344.0       319    ₹351          +0.3%       +3.0%     

  AFTER: Invested ₹2,250,788 | Cash ₹11,128 | Total ₹2,261,916 | Positions 19/20 | Slot ₹113,129

========================================================================
  REBALANCE #20  —  01 Aug 2016
  NAV: ₹2,528,388  |  Slot: ₹126,419  |  Cash: ₹11,128
========================================================================
  [SECTOR CAP≤4] dropped: IOC

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  NATCOPHARM  24     HEALTH     01-Jul-16   523.6       562.8       216    ₹8,461        +7.5%     31d   
  HDFC        26     PVT BNK    01-Jun-16   265.8       283.3       411    ₹7,189        +6.6%     61d   
  HDFCBANK    27     PVT BNK    01-Jun-16   265.8       283.3       411    ₹7,189        +6.6%     61d   
  WELSPUNIND  43     CONSUMP    01-Jun-16   97.9        98.3        1118   ₹517          +0.5%     61d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SHREECEM    10     INFRA      2.284    0.74   +51.9%    +31.1%    15,652.0    8      ₹125,216      +3.3%     
  BERGEPAINT  11     CON DUR    2.124    0.44   +53.4%    +24.9%    186.1       679    ₹126,361      +3.1%     
  SUNTV       12     MEDIA      2.115    0.71   +88.3%    +29.8%    367.7       343    ₹126,130      +14.6%    

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  1      FIN SVC    01-Jul-15   52.7        108.4       2058   ₹114,583      +105.6%     +22.0%    
  HINDPETRO   6      OIL&GAS    03-Aug-15   81.6        121.9       1422   ₹57,374       +49.4%      +12.3%    
  MUTHOOTFIN  3      FIN SVC    01-Jun-16   205.5       298.8       532    ₹49,668       +45.4%      +20.6%    
  VGUARD      2      CON DUR    01-Jun-16   93.1        113.9       1175   ₹24,364       +22.3%      +11.9%    
  BPCL        17     OIL&GAS    01-Jun-16   89.0        108.8       1230   ₹24,389       +22.3%      +4.5%     
  BAJAJFINSV  4      FIN SVC    01-Jul-16   234.4       281.5       482    ₹22,737       +20.1%      +12.3%    
  BIOCON      5      HEALTH     01-Jun-16   117.0       135.3       935    ₹17,084       +15.6%      +7.7%     
  RAMCOCEM    8      MFG        01-Jun-16   473.9       544.0       231    ₹16,197       +14.8%      +0.2%     
  HAVELLS     18     CON DUR    01-Jun-16   331.1       378.3       330    ₹15,583       +14.3%      +8.5%     
  SHRIRAMFIN  19     FIN SVC    01-Jun-16   197.6       221.8       553    ₹13,379       +12.2%      +5.8%     
  ASIANPAINT  7      CONSUMP    01-Jul-16   921.2       1,030.0     122    ₹13,273       +11.8%      +6.5%     
  INDUSINDBK  21     PVT BNK    01-Jun-16   1,042.1     1,138.1     105    ₹10,089       +9.2%       +5.1%     
  IGL         20     OIL&GAS    01-Jul-16   53.3        56.2        2123   ₹6,255        +5.5%       +4.5%     
  PETRONET    13     OIL&GAS    01-Jul-16   101.7       106.3       1112   ₹5,115        +4.5%       +5.9%     
  PIDILITIND  9      MFG        01-Jun-16   342.9       350.4       319    ₹2,400        +2.2%       +1.2%     

  AFTER: Invested ₹2,430,660 | Cash ₹97,280 | Total ₹2,527,940 | Positions 18/20 | Slot ₹126,419

========================================================================
  REBALANCE #21  —  01 Sep 2016
  NAV: ₹2,618,479  |  Slot: ₹130,924  |  Cash: ₹97,280
========================================================================
  [SECTOR CAP≤4] dropped: IOC

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  INDUSINDBK  41     PVT BNK    01-Jun-16   1,042.1     1,122.6     105    ₹8,461        +7.7%     92d   
  SHRIRAMFIN  53     FIN SVC    01-Jun-16   197.6       205.2       553    ₹4,216        +3.9%     92d   
  PIDILITIND  71     MFG        01-Jun-16   342.9       332.7       319    ₹-3,232       -3.0%     92d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HINDZINC    8      METAL      2.655    0.12   +93.4%    +34.2%    101.0       1296   ₹130,855      +1.8%     
  DHFL        11     FIN SVC    2.233    0.16   +37.0%    +46.6%    279.3       468    ₹130,715      +6.6%     
  POWERGRID   12     ENERGY     2.121    0.04   +40.4%    +22.7%    65.5        1998   ₹130,908      +2.7%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  3      FIN SVC    01-Jul-15   52.7        107.8       2058   ₹113,283      +104.4%     +8.1%     
  MUTHOOTFIN  4      FIN SVC    01-Jun-16   205.5       305.5       532    ₹53,224       +48.7%      +4.6%     
  HINDPETRO   14     OIL&GAS    03-Aug-15   81.6        116.4       1422   ₹49,497       +42.7%      +1.2%     
  VGUARD      2      CON DUR    01-Jun-16   93.1        130.7       1175   ₹44,198       +40.4%      +8.8%     
  BAJAJFINSV  1      FIN SVC    01-Jul-16   234.4       301.6       482    ₹32,418       +28.7%      +9.7%     
  IGL         5      OIL&GAS    01-Jul-16   53.3        67.9        2123   ₹30,954       +27.4%      +7.8%     
  BIOCON      7      HEALTH     01-Jun-16   117.0       149.1       935    ₹29,965       +27.4%      +5.1%     
  BPCL        26     OIL&GAS    01-Jun-16   89.0        107.4       1230   ₹22,634       +20.7%      -0.6%     
  HAVELLS     18     CON DUR    01-Jun-16   331.1       395.8       330    ₹21,375       +19.6%      +3.3%     
  PETRONET    6      OIL&GAS    01-Jul-16   101.7       119.2       1112   ₹19,398       +17.1%      +6.0%     
  RAMCOCEM    17     MFG        01-Jun-16   473.9       547.1       231    ₹16,925       +15.5%      +0.7%     
  ASIANPAINT  30     CONSUMP    01-Jul-16   921.2       1,063.1     122    ₹17,315       +15.4%      +3.1%     
  BERGEPAINT  10     CON DUR    01-Aug-16   186.1       210.5       679    ₹16,560       +13.1%      +7.3%     
  SHREECEM    15     INFRA      01-Aug-16   15,652.0    16,594.9    8      ₹7,543        +6.0%       +2.2%     
  SUNTV       33     MEDIA      01-Aug-16   367.7       362.8       343    ₹-1,706       -1.4%       +1.2%     

  AFTER: Invested ₹2,576,159 | Cash ₹41,854 | Total ₹2,618,013 | Positions 18/20 | Slot ₹130,924

========================================================================
  REBALANCE #22  —  03 Oct 2016
  NAV: ₹2,703,901  |  Slot: ₹135,195  |  Cash: ₹41,854
========================================================================
  [SECTOR CAP≤4] dropped: CHOLAFIN

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BPCL        33     OIL&GAS    01-Jun-16   89.0        114.9       1230   ₹31,850       +29.1%    124d  
  SHREECEM    32     INFRA      01-Aug-16   15,652.0    17,211.1    8      ₹12,473       +10.0%    63d   
  POWERGRID   44     ENERGY     01-Sep-16   65.5        64.9        1998   ₹-1,141       -0.9%     32d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MRF         5      MFG        2.569    0.19   +25.7%    +57.3%    51,226.4    2      ₹102,453      +16.3%    
  IOC         6      OIL&GAS    2.485    0.08   +60.9%    +38.9%    53.5        2526   ₹135,195      +4.8%     
  SRF         9      MFG        2.293    0.24   +68.7%    +45.9%    366.2       369    ₹135,122      +10.8%    

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  3      FIN SVC    01-Jul-15   52.7        104.9       2058   ₹107,445      +99.0%      +0.1%     
  HINDPETRO   15     OIL&GAS    03-Aug-15   81.6        124.8       1422   ₹61,408       +52.9%      +4.3%     
  MUTHOOTFIN  16     FIN SVC    01-Jun-16   205.5       295.2       532    ₹47,723       +43.7%      -1.6%     
  VGUARD      4      CON DUR    01-Jun-16   93.1        127.4       1175   ₹40,235       +36.8%      +0.4%     
  BAJAJFINSV  7      FIN SVC    01-Jul-16   234.4       318.0       482    ₹40,291       +35.7%      +3.5%     
  BIOCON      1      HEALTH     01-Jun-16   117.0       157.9       935    ₹38,212       +34.9%      +3.1%     
  IGL         12     OIL&GAS    01-Jul-16   53.3        69.0        2123   ₹33,333       +29.5%      +3.8%     
  RAMCOCEM    14     MFG        01-Jun-16   473.9       596.8       231    ₹28,407       +26.0%      +4.3%     
  HAVELLS     21     CON DUR    01-Jun-16   331.1       403.7       330    ₹23,955       +21.9%      +4.1%     
  PETRONET    10     OIL&GAS    01-Jul-16   101.7       121.2       1112   ₹21,612       +19.1%      +3.5%     
  ASIANPAINT  24     CONSUMP    01-Jul-16   921.2       1,096.3     122    ₹21,356       +19.0%      +1.9%     
  HINDZINC    2      METAL      01-Sep-16   101.0       113.8       1296   ₹16,603       +12.7%      +11.7%    
  BERGEPAINT  13     CON DUR    01-Aug-16   186.1       209.6       679    ₹15,940       +12.6%      +1.5%     
  SUNTV       22     MEDIA      01-Aug-16   367.7       412.8       343    ₹15,473       +12.3%      +8.0%     
  DHFL        25     FIN SVC    01-Sep-16   279.3       288.5       468    ₹4,298        +3.3%       +3.3%     

  AFTER: Invested ₹2,626,072 | Cash ₹77,386 | Total ₹2,703,458 | Positions 18/20 | Slot ₹135,195

========================================================================
  REBALANCE #23  —  01 Nov 2016
  NAV: ₹2,783,767  |  Slot: ₹139,188  |  Cash: ₹77,386
========================================================================

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HAVELLS     28     CON DUR    01-Jun-16   331.1       381.3       330    ₹16,571       +15.2%    153d  
  ASIANPAINT  59     CONSUMP    01-Jul-16   921.2       991.8       122    ₹8,618        +7.7%     123d  

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  WHIRLPOOL   6      CON DUR    2.626    0.32   +65.4%    +35.0%    1,116.4     124    ₹138,432      +3.8%     
  TVSMOTOR    7      AUTO       2.539    0.33   +64.9%    +40.1%    382.5       363    ₹138,832      +3.9%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  15     FIN SVC    01-Jul-15   52.7        102.4       2058   ₹102,298      +94.3%      -3.4%     
  HINDPETRO   20     OIL&GAS    03-Aug-15   81.6        134.1       1422   ₹74,680       +64.4%      +4.3%     
  VGUARD      1      CON DUR    01-Jun-16   93.1        147.8       1175   ₹64,259       +58.7%      +9.6%     
  MUTHOOTFIN  18     FIN SVC    01-Jun-16   205.5       304.1       532    ₹52,486       +48.0%      +1.8%     
  BAJAJFINSV  4      FIN SVC    01-Jul-16   234.4       344.3       482    ₹52,968       +46.9%      +6.3%     
  IGL         2      OIL&GAS    01-Jul-16   53.3        75.8        2123   ₹47,878       +42.3%      +3.5%     
  BIOCON      9      HEALTH     01-Jun-16   117.0       151.5       935    ₹32,239       +29.5%      -2.2%     
  PETRONET    3      OIL&GAS    01-Jul-16   101.7       131.4       1112   ₹32,965       +29.1%      +1.4%     
  RAMCOCEM    14     MFG        01-Jun-16   473.9       609.4       231    ₹31,319       +28.6%      +1.0%     
  HINDZINC    5      METAL      01-Sep-16   101.0       121.9       1296   ₹27,092       +20.7%      +9.6%     
  DHFL        11     FIN SVC    01-Sep-16   279.3       322.3       468    ₹20,100       +15.4%      +4.3%     
  SUNTV       26     MEDIA      01-Aug-16   367.7       422.0       343    ₹18,605       +14.8%      +2.3%     
  BERGEPAINT  24     CON DUR    01-Aug-16   186.1       207.2       679    ₹14,349       +11.4%      -0.3%     
  IOC         13     OIL&GAS    03-Oct-16   53.5        57.7        2526   ₹10,436       +7.7%       +2.9%     
  MRF         12     MFG        03-Oct-16   51,226.4    49,450.1    2      ₹-3,553       -3.5%       +1.7%     
  SRF         21     MFG        03-Oct-16   366.2       348.0       369    ₹-6,712       -5.0%       -1.3%     

  AFTER: Invested ₹2,736,816 | Cash ₹46,622 | Total ₹2,783,438 | Positions 18/20 | Slot ₹139,188

========================================================================
  REBALANCE #24  —  01 Dec 2016
  NAV: ₹2,509,118  |  Slot: ₹125,456  |  Cash: ₹46,622
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SUNTV       38     MEDIA      01-Aug-16   367.7       358.9       343    ₹-3,039       -2.4%     122d  
  DHFL        68     FIN SVC    01-Sep-16   279.3       243.8       468    ₹-16,612      -12.7%    91d   
  SRF         41     MFG        03-Oct-16   366.2       303.7       369    ₹-23,057      -17.1%    59d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  VAKRANGEE   2      IT         3.620    0.06   +83.6%    +42.4%    123.4       1016   ₹125,400      +4.8%     
  SYNGENE     3      HEALTH     3.076    0.20   +62.3%    +30.9%    297.8       421    ₹125,367      +9.6%     
  AIAENG      11     MFG        2.188    0.49   +44.9%    +12.2%    1,251.1     100    ₹125,113      +4.0%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  30     FIN SVC    01-Jul-15   52.7        88.3        2058   ₹73,169       +67.4%      -0.3%     
  HINDPETRO   8      OIL&GAS    03-Aug-15   81.6        129.0       1422   ₹67,357       +58.1%      -1.9%     
  IGL         6      OIL&GAS    01-Jul-16   53.3        73.2        2123   ₹42,286       +37.4%      +0.2%     
  BIOCON      4      HEALTH     01-Jun-16   117.0       151.6       935    ₹32,269       +29.5%      +3.4%     
  PETRONET    7      OIL&GAS    01-Jul-16   101.7       130.9       1112   ₹32,469       +28.7%      +2.5%     
  VGUARD      12     CON DUR    01-Jun-16   93.1        116.3       1175   ₹27,211       +24.9%      -6.0%     
  BAJAJFINSV  18     FIN SVC    01-Jul-16   234.4       292.6       482    ₹28,041       +24.8%      -2.1%     
  MUTHOOTFIN  24     FIN SVC    01-Jun-16   205.5       253.8       532    ₹25,717       +23.5%      -3.8%     
  HINDZINC    1      METAL      01-Sep-16   101.0       123.8       1296   ₹29,654       +22.7%      +4.4%     
  RAMCOCEM    10     MFG        01-Jun-16   473.9       581.0       231    ₹24,744       +22.6%      +2.6%     
  BERGEPAINT  32     CON DUR    01-Aug-16   186.1       186.0       679    ₹-81          -0.1%       +4.3%     
  IOC         15     OIL&GAS    03-Oct-16   53.5        52.8        2526   ₹-1,792       -1.3%       -1.5%     
  MRF         5      MFG        03-Oct-16   51,226.4    48,256.4    2      ₹-5,940       -5.8%       -0.3%     
  TVSMOTOR    9      AUTO       01-Nov-16   382.5       349.2       363    ₹-12,069      -8.7%       +0.3%     
  WHIRLPOOL   21     CON DUR    01-Nov-16   1,116.4     929.0       124    ₹-23,237      -16.8%      -2.0%     

  AFTER: Invested ₹2,489,117 | Cash ₹19,555 | Total ₹2,508,672 | Positions 18/20 | Slot ₹125,456

========================================================================
  REBALANCE #25  —  02 Jan 2017
  NAV: ₹2,472,976  |  Slot: ₹123,649  |  Cash: ₹19,555
========================================================================
  [SECTOR CAP≤4] dropped: BPCL

  [REGIME OFF] Nifty 200 4,364.9 < SMA200 4,376.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  43     FIN SVC    01-Jul-15   52.7        84.0        2058   ₹64,397       +59.4%      +3.2%     
  HINDPETRO   7      OIL&GAS    03-Aug-15   81.6        129.8       1422   ₹68,588       +59.1%      +3.9%     
  IGL         1      OIL&GAS    01-Jul-16   53.3        80.5        2123   ₹57,850       +51.2%      +4.0%     
  BIOCON      3      HEALTH     01-Jun-16   117.0       155.9       935    ₹36,359       +33.2%      +1.9%     
  BAJAJFINSV  20     FIN SVC    01-Jul-16   234.4       293.6       482    ₹28,537       +25.3%      +4.5%     
  PETRONET    8      OIL&GAS    01-Jul-16   101.7       127.3       1112   ₹28,404       +25.1%      +0.5%     
  VGUARD      18     CON DUR    01-Jun-16   93.1        112.4       1175   ₹22,641       +20.7%      -1.5%     
  MUTHOOTFIN  37     FIN SVC    01-Jun-16   205.5       239.8       532    ₹18,248       +16.7%      +0.9%     
  HINDZINC    2      METAL      01-Sep-16   101.0       116.9       1296   ₹20,636       +15.8%      -1.3%     
  RAMCOCEM    17     MFG        01-Jun-16   473.9       541.4       231    ₹15,604       +14.3%      +3.2%     
  IOC         4      OIL&GAS    03-Oct-16   53.5        58.1        2526   ₹11,466       +8.5%       +5.3%     
  AIAENG      11     MFG        01-Dec-16   1,251.1     1,250.3     100    ₹-81          -0.1%       +2.9%     
  VAKRANGEE   6      IT         01-Dec-16   123.4       122.9       1016   ₹-546         -0.4%       +0.8%     
  MRF         25     MFG        03-Oct-16   51,226.4    48,454.9    2      ₹-5,543       -5.4%       -1.0%     
  SYNGENE     5      HEALTH     01-Dec-16   297.8       279.2       421    ₹-7,826       -6.2%       +0.6%     
  TVSMOTOR    14     AUTO       01-Nov-16   382.5       349.4       363    ₹-12,018      -8.7%       +0.5%     
  BERGEPAINT  73     CON DUR    01-Aug-16   186.1       168.1       679    ₹-12,245      -9.7% ⚠     +2.1%     
  WHIRLPOOL   41     CON DUR    01-Nov-16   1,116.4     871.5       124    ₹-30,368      -21.9%      +0.4%     
  ⚠  WAZ < 0 (momentum below universe mean): BERGEPAINT

  AFTER: Invested ₹2,453,421 | Cash ₹19,555 | Total ₹2,472,976 | Positions 18/20 | Slot ₹123,649

========================================================================
  REBALANCE #26  —  01 Feb 2017
  NAV: ₹2,762,427  |  Slot: ₹138,121  |  Cash: ₹19,555
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MUTHOOTFIN  35     FIN SVC    01-Jun-16   205.5       272.1       532    ₹35,468       +32.4%    245d  
  TVSMOTOR    40     AUTO       01-Nov-16   382.5       374.6       363    ₹-2,843       -2.0%     92d   
  BERGEPAINT  97     CON DUR    01-Aug-16   186.1       169.1       679    ₹-11,571      -9.2%     184d  
  SYNGENE     54     HEALTH     01-Dec-16   297.8       265.9       421    ₹-13,434      -10.7%    62d   
  WHIRLPOOL   65     CON DUR    01-Nov-16   1,116.4     940.0       124    ₹-21,871      -15.8%    92d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  POWERGRID   5      ENERGY     2.888    0.34   +56.8%    +17.5%    73.8        1872   ₹138,089      +3.7%     
  BEL         9      DEFENCE    2.355    0.63   +33.2%    +20.5%    40.0        3453   ₹138,093      +4.1%     
  RECLTD      10     FIN SVC    2.354    0.84   +77.4%    +15.8%    54.0        2559   ₹138,093      +6.0%     
  PFC         12     FIN SVC    2.197    0.70   +76.0%    +14.8%    62.2        2219   ₹138,075      +2.1%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  21     FIN SVC    01-Jul-15   52.7        102.9       2058   ₹103,361      +95.3%      +12.1%    
  HINDPETRO   2      OIL&GAS    03-Aug-15   81.6        155.5       1422   ₹105,039      +90.5%      +7.6%     
  IGL         8      OIL&GAS    01-Jul-16   53.3        84.2        2123   ₹65,594       +58.0%      +3.1%     
  VGUARD      11     CON DUR    01-Jun-16   93.1        141.1       1175   ₹56,413       +51.6%      +12.8%    
  RAMCOCEM    7      MFG        01-Jun-16   473.9       693.0       231    ₹50,631       +46.3%      +13.1%    
  BIOCON      6      HEALTH     01-Jun-16   117.0       166.6       935    ₹46,352       +42.4%      +2.4%     
  BAJAJFINSV  24     FIN SVC    01-Jul-16   234.4       331.3       482    ₹46,707       +41.3%      +7.6%     
  HINDZINC    1      METAL      01-Sep-16   101.0       138.1       1296   ₹48,156       +36.8%      +6.3%     
  PETRONET    23     OIL&GAS    01-Jul-16   101.7       132.7       1112   ₹34,472       +30.5%      +3.6%     
  IOC         3      OIL&GAS    03-Oct-16   53.5        66.6        2526   ₹32,965       +24.4%      +5.1%     
  VAKRANGEE   4      IT         01-Dec-16   123.4       137.7       1016   ₹14,461       +11.5%      +4.5%     
  AIAENG      17     MFG        01-Dec-16   1,251.1     1,274.0     100    ₹2,284        +1.8%       -0.2%     
  MRF         25     MFG        03-Oct-16   51,226.4    51,910.7    2      ₹1,369        +1.3%       +0.7%     

  AFTER: Invested ₹2,671,172 | Cash ₹90,599 | Total ₹2,761,771 | Positions 17/20 | Slot ₹138,121

========================================================================
  REBALANCE #27  —  01 Mar 2017
  NAV: ₹2,874,163  |  Slot: ₹143,708  |  Cash: ₹90,599
========================================================================
  [SECTOR CAP≤4] dropped: DHFL

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PETRONET    42     OIL&GAS    01-Jul-16   101.7       137.4       1112   ₹39,662       +35.1%    243d  
  MRF         41     MFG        03-Oct-16   51,226.4    52,145.1    2      ₹1,837        +1.8%     149d  
  BEL         48     DEFENCE    01-Feb-17   40.0        38.9        3453   ₹-3,904       -2.8%     28d   
  POWERGRID   53     ENERGY     01-Feb-17   73.8        69.4        1872   ₹-8,179       -5.9%     28d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SUNTV       6      MEDIA      2.706    0.95   +126.7%   +54.9%    566.5       253    ₹143,334      +6.7%     
  HDFC        9      PVT BNK    2.559    0.39   +41.8%    +16.0%    317.3       452    ₹143,422      +3.2%     
  HDFCBANK    10     PVT BNK    2.559    0.39   +41.8%    +16.0%    317.3       452    ₹143,422      +3.2%     
  INDUSINDBK  11     PVT BNK    2.306    0.74   +58.3%    +22.0%    1,257.3     114    ₹143,328      +1.4%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  32     FIN SVC    01-Jul-15   52.7        103.9       2058   ₹105,289      +97.1%      +1.6%     
  HINDPETRO   9      OIL&GAS    03-Aug-15   81.6        156.1       1422   ₹106,001      +91.4%      -0.7%     
  IGL         2      OIL&GAS    01-Jul-16   53.3        92.0        2123   ₹82,171       +72.7%      +3.0%     
  VGUARD      1      CON DUR    01-Jun-16   93.1        158.9       1175   ₹77,283       +70.6%      +9.3%     
  BAJAJFINSV  4      FIN SVC    01-Jul-16   234.4       390.5       482    ₹75,274       +66.6%      +7.8%     
  BIOCON      6      HEALTH     01-Jun-16   117.0       182.5       935    ₹61,216       +55.9%      +2.3%     
  HINDZINC    14     METAL      01-Sep-16   101.0       143.3       1296   ₹54,868       +41.9%      +4.4%     
  RAMCOCEM    28     MFG        01-Jun-16   473.9       648.2       231    ₹40,269       +36.8%      +0.4%     
  IOC         3      OIL&GAS    03-Oct-16   53.5        70.1        2526   ₹41,802       +30.9%      +1.2%     
  VAKRANGEE   25     IT         01-Dec-16   123.4       141.2       1016   ₹18,054       +14.4%      +2.1%     
  AIAENG      18     MFG        01-Dec-16   1,251.1     1,404.8     100    ₹15,367       +12.3%      +2.9%     
  RECLTD      15     FIN SVC    01-Feb-17   54.0        58.5        2559   ₹11,610       +8.4%       +6.7%     
  PFC         39     FIN SVC    01-Feb-17   62.2        62.9        2219   ₹1,578        +1.1%       +2.1%     

  AFTER: Invested ₹2,835,907 | Cash ₹37,575 | Total ₹2,873,482 | Positions 17/20 | Slot ₹143,708

========================================================================
  REBALANCE #28  —  03 Apr 2017
  NAV: ₹3,006,900  |  Slot: ₹150,345  |  Cash: ₹37,575
========================================================================
  [SECTOR CAP≤4] dropped: DHFL

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJFINANCE  30     FIN SVC    01-Jul-15   52.7        114.1       2058   ₹126,331      +116.5%   642d  
  IGL         28     OIL&GAS    01-Jul-16   53.3        88.9        2123   ₹75,570       +66.8%    276d  
  RAMCOCEM    39     MFG        01-Jun-16   473.9       654.3       231    ₹41,681       +38.1%    306d  
  AIAENG      34     MFG        01-Dec-16   1,251.1     1,509.8     100    ₹25,863       +20.7%    123d  
  VAKRANGEE   27     IT         01-Dec-16   123.4       147.3       1016   ₹24,284       +19.4%    123d  
  PFC         33     FIN SVC    01-Feb-17   62.2        70.1        2219   ₹17,550       +12.7%    61d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  EDELWEISS   4      FIN SVC    2.685    1.18   +176.5%   +67.6%    73.6        2042   ₹150,321      +12.3%    
  IBULHSGFIN  5      FIN SVC    2.523    0.30   +63.1%    +54.9%    707.9       212    ₹150,068      +6.5%     
  DALMIABHA   6      MFG        2.502    0.29   +147.5%   +45.8%    1,969.1     76     ₹149,650      +2.4%     
  BBTC        7      FMCG       2.372    1.19   +122.7%   +65.1%    828.5       181    ₹149,963      +6.8%     
  GUJGASLTD   10     OIL&GAS    2.305    0.42   +44.0%    +44.8%    142.3       1056   ₹150,318      +4.2%     
  NATCOPHARM  12     HEALTH     2.250    0.28   +85.9%    +50.4%    803.3       187    ₹150,211      +8.1%     
  KARURVYSYA  13     PVT BNK    2.245    0.36   +36.7%    +42.4%    73.3        2050   ₹150,331      +10.0%    

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDPETRO   15     OIL&GAS    03-Aug-15   81.6        158.1       1422   ₹108,780      +93.8%      +0.7%     
  VGUARD      1      CON DUR    01-Jun-16   93.1        168.4       1175   ₹88,486       +80.9%      +2.5%     
  BAJAJFINSV  3      FIN SVC    01-Jul-16   234.4       407.9       482    ₹83,640       +74.0%      +2.7%     
  BIOCON      9      HEALTH     01-Jun-16   117.0       185.3       935    ₹63,812       +58.3%      +0.9%     
  HINDZINC    21     METAL      01-Sep-16   101.0       143.4       1296   ₹55,022       +42.0%      +2.0%     
  IOC         19     OIL&GAS    03-Oct-16   53.5        70.8        2526   ₹43,683       +32.3%      +2.1%     
  RECLTD      2      FIN SVC    01-Feb-17   54.0        68.6        2559   ₹37,431       +27.1%      +6.9%     
  SUNTV       8      MEDIA      01-Mar-17   566.5       626.4       253    ₹15,152       +10.6%      +5.2%     
  INDUSINDBK  22     PVT BNK    01-Mar-17   1,257.3     1,330.5     114    ₹8,349        +5.8%       +1.8%     
  HDFC        17     PVT BNK    01-Mar-17   317.3       326.8       452    ₹4,282        +3.0%       +1.1%     
  HDFCBANK    18     PVT BNK    01-Mar-17   317.3       326.8       452    ₹4,282        +3.0%       +1.1%     

  AFTER: Invested ₹2,989,279 | Cash ₹16,373 | Total ₹3,005,652 | Positions 18/20 | Slot ₹150,345

========================================================================
  REBALANCE #29  —  02 May 2017
  NAV: ₹3,231,373  |  Slot: ₹161,569  |  Cash: ₹16,373
========================================================================
  [SECTOR CAP≤4] dropped: IGL

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HINDZINC    61     METAL      01-Sep-16   101.0       132.2       1296   ₹40,489       +30.9%    243d  
  BBTC        7      FMCG       03-Apr-17   828.5       914.5       181    ₹15,553       +10.4%    29d   
  EDELWEISS   8      FIN SVC    03-Apr-17   73.6        75.3        2042   ₹3,361        +2.2%     29d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  DHFL        4      FIN SVC    2.895    0.02   +122.5%   +57.2%    438.4       368    ₹161,341      +9.7%     
  MRF         13     MFG        2.170    0.73   +93.7%    +28.9%    67,499.7    2      ₹134,999      +7.2%     
  RELIANCE    14     OIL&GAS    2.125    0.48   +34.8%    +34.7%    298.9       540    ₹161,426      -0.2%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDPETRO   31     OIL&GAS    03-Aug-15   81.6        168.5       1422   ₹123,519      +106.5%     +1.9%     
  VGUARD      6      CON DUR    01-Jun-16   93.1        183.5       1175   ₹106,210      +97.1%      +3.7%     
  BAJAJFINSV  2      FIN SVC    01-Jul-16   234.4       454.5       482    ₹106,087      +93.9%      +4.3%     
  IOC         11     OIL&GAS    03-Oct-16   53.5        82.3        2526   ₹72,733       +53.8%      +5.6%     
  BIOCON      40     HEALTH     01-Jun-16   117.0       178.0       935    ₹57,012       +52.1%      -2.2%     
  RECLTD      1      FIN SVC    01-Feb-17   54.0        81.2        2559   ₹69,597       +50.4%      +8.0%     
  SUNTV       3      MEDIA      01-Mar-17   566.5       727.8       253    ₹40,801       +28.5%      +7.5%     
  DALMIABHA   19     MFG        03-Apr-17   1,969.1     2,189.3     76     ₹16,733       +11.2%      +4.5%     
  GUJGASLTD   5      OIL&GAS    03-Apr-17   142.3       157.8       1056   ₹16,313       +10.9%      +3.3%     
  HDFC        9      PVT BNK    01-Mar-17   317.3       351.3       452    ₹15,351       +10.7%      +3.5%     
  HDFCBANK    10     PVT BNK    01-Mar-17   317.3       351.3       452    ₹15,351       +10.7%      +3.5%     
  INDUSINDBK  36     PVT BNK    01-Mar-17   1,257.3     1,370.6     114    ₹12,926       +9.0%       +1.0%     
  IBULHSGFIN  14     FIN SVC    03-Apr-17   707.9       761.4       212    ₹11,345       +7.6%       +8.2%     
  NATCOPHARM  12     HEALTH     03-Apr-17   803.3       855.2       187    ₹9,710        +6.5%       +2.0%     
  KARURVYSYA  13     PVT BNK    03-Apr-17   73.3        76.0        2050   ₹5,551        +3.7%       +3.9%     

  AFTER: Invested ₹3,182,225 | Cash ₹48,604 | Total ₹3,230,829 | Positions 18/20 | Slot ₹161,569

========================================================================
  REBALANCE #30  —  01 Jun 2017
  NAV: ₹3,141,277  |  Slot: ₹157,064  |  Cash: ₹48,604
========================================================================
  [SECTOR CAP≤4] dropped: KOTAKBANK

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BIOCON      102    HEALTH     01-Jun-16   117.0       156.7       935    ₹37,109       +33.9%    365d  
  RECLTD      3      FIN SVC    01-Feb-17   54.0        72.3        2559   ₹46,802       +33.9%    120d  
  SUNTV       36     MEDIA      01-Mar-17   566.5       633.1       253    ₹16,836       +11.7%    92d   
  MRF         —      MFG        02-May-17   67,499.7    67,346.4    2      ₹-306         -0.2%     30d   
  RELIANCE    56     OIL&GAS    02-May-17   298.9       289.7       540    ₹-4,982       -3.1%     30d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIENSOL  1      ENERGY     4.624    1.04   +324.2%   +105.6%   127.1       1236   ₹157,034      +22.7%    
  VAKRANGEE   2      IT         3.001    0.16   +124.2%   +22.6%    174.1       902    ₹157,052      +9.6%     
  TVSMOTOR    6      AUTO       2.315    0.80   +81.3%    +24.9%    508.7       308    ₹156,673      +2.1%     
  MARUTI      7      AUTO       2.289    1.00   +77.2%    +20.7%    6,570.4     23     ₹151,120      +3.7%     
  HINDUNILVR  8      FMCG       2.255    0.50   +32.9%    +26.5%    940.4       167    ₹157,055      +7.3%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDPETRO   35     OIL&GAS    03-Aug-15   81.6        165.0       1422   ₹118,577      +102.2%     +1.0%     
  VGUARD      23     CON DUR    01-Jun-16   93.1        176.8       1175   ₹98,282       +89.8%      -3.8%     
  BAJAJFINSV  21     FIN SVC    01-Jul-16   234.4       418.6       482    ₹88,790       +78.6%      -0.1%     
  IOC         16     OIL&GAS    03-Oct-16   53.5        76.9        2526   ₹59,029       +43.7%      -3.3%     
  DALMIABHA   5      MFG        03-Apr-17   1,969.1     2,419.4     76     ₹34,222       +22.9%      +1.8%     
  HDFC        6      PVT BNK    01-Mar-17   317.3       371.3       452    ₹24,420       +17.0%      +2.9%     
  HDFCBANK    7      PVT BNK    01-Mar-17   317.3       371.3       452    ₹24,420       +17.0%      +2.9%     
  IBULHSGFIN  14     FIN SVC    03-Apr-17   707.9       817.2       212    ₹23,189       +15.5%      +6.4%     
  INDUSINDBK  43     PVT BNK    01-Mar-17   1,257.3     1,408.4     114    ₹17,227       +12.0%      +2.9%     
  NATCOPHARM  17     HEALTH     03-Apr-17   803.3       877.3       187    ₹13,835       +9.2%       +3.1%     
  KARURVYSYA  44     PVT BNK    03-Apr-17   73.3        74.6        2050   ₹2,646        +1.8%       +0.9%     
  GUJGASLTD   25     OIL&GAS    03-Apr-17   142.3       142.8       1056   ₹496          +0.3%       -1.6%     
  DHFL        11     FIN SVC    02-May-17   438.4       408.8       368    ₹-10,909      -6.8%       +0.4%     

  AFTER: Invested ₹3,088,858 | Cash ₹51,494 | Total ₹3,140,352 | Positions 18/20 | Slot ₹157,064

========================================================================
  REBALANCE #31  —  03 Jul 2017
  NAV: ₹3,194,678  |  Slot: ₹159,734  |  Cash: ₹51,494
========================================================================

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  INDUSINDBK  54     PVT BNK    01-Mar-17   1,257.3     1,413.5     114    ₹17,805       +12.4%    124d  
  GUJGASLTD   74     OIL&GAS    03-Apr-17   142.3       140.2       1056   ₹-2,282       -1.5%     91d   

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  GODREJIND   7      CONSUMP    2.388    0.82   +73.0%    +28.6%    640.8       249    ₹159,558      +3.3%     
  COROMANDEL  8      MFG        2.231    0.97   +83.3%    +38.0%    383.2       416    ₹159,416      +2.4%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDPETRO   51     OIL&GAS    03-Aug-15   81.6        157.6       1422   ₹108,065      +93.1%      -1.1%     
  VGUARD      49     CON DUR    01-Jun-16   93.1        168.1       1175   ₹88,147       +80.6%      -3.4%     
  BAJAJFINSV  25     FIN SVC    01-Jul-16   234.4       411.1       482    ₹85,197       +75.4%      -0.8%     
  IOC         24     OIL&GAS    03-Oct-16   53.5        71.9        2526   ₹46,381       +34.3%      -3.9%     
  DALMIABHA   12     MFG        03-Apr-17   1,969.1     2,523.8     76     ₹42,162       +28.2%      +2.7%     
  KARURVYSYA  10     PVT BNK    03-Apr-17   73.3        91.1        2050   ₹36,405       +24.2%      +4.1%     
  HDFC        5      PVT BNK    01-Mar-17   317.3       380.7       452    ₹28,647       +20.0%      +0.5%     
  HDFCBANK    6      PVT BNK    01-Mar-17   317.3       380.7       452    ₹28,647       +20.0%      +0.5%     
  NATCOPHARM  15     HEALTH     03-Apr-17   803.3       945.4       187    ₹26,575       +17.7%      +4.5%     
  VAKRANGEE   2      IT         01-Jun-17   174.1       196.8       902    ₹20,489       +13.0%      +5.6%     
  IBULHSGFIN  35     FIN SVC    03-Apr-17   707.9       769.8       212    ₹13,138       +8.8%       -2.6%     
  TVSMOTOR    3      AUTO       01-Jun-17   508.7       526.4       308    ₹5,460        +3.5%       +1.9%     
  MARUTI      4      AUTO       01-Jun-17   6,570.4     6,761.1     23     ₹4,386        +2.9%       +1.9%     
  HINDUNILVR  13     FMCG       01-Jun-17   940.4       951.0       167    ₹1,758        +1.1%       +1.5%     
  ADANIENSOL  1      ENERGY     01-Jun-17   127.1       126.5       1236   ₹-680         -0.4%       +6.4%     
  DHFL        11     FIN SVC    02-May-17   438.4       433.7       368    ₹-1,749       -1.1%       +0.9%     

  AFTER: Invested ₹3,152,990 | Cash ₹41,309 | Total ₹3,194,299 | Positions 18/20 | Slot ₹159,734

========================================================================
  REBALANCE #32  —  01 Aug 2017
  NAV: ₹3,369,464  |  Slot: ₹168,473  |  Cash: ₹41,309
========================================================================
  [SECTOR CAP≤4] dropped: KOTAKBANK

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HINDPETRO   40     OIL&GAS    03-Aug-15   81.6        177.2       1422   ₹135,991      +117.2%   729d  
  BAJAJFINSV  19     FIN SVC    01-Jul-16   234.4       504.6       482    ₹130,230      +115.3%   396d  
  VGUARD      49     CON DUR    01-Jun-16   93.1        174.9       1175   ₹96,130       +87.9%    426d  
  IOC         91     OIL&GAS    03-Oct-16   53.5        68.8        2526   ₹38,614       +28.6%    302d  
  COROMANDEL  13     MFG        03-Jul-17   383.2       403.3       416    ₹8,337        +5.2%     29d   
  ADANIENSOL  2      ENERGY     01-Jun-17   127.1       125.8       1236   ₹-1,545       -1.0%     61d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RAJESHEXPO  6      CON DUR    2.610    0.79   +64.8%    +16.7%    714.7       235    ₹167,949      +4.4%     
  IGL         8      OIL&GAS    2.542    0.77   +93.2%    +12.9%    104.2       1616   ₹168,423      +4.8%     
  RELIANCE    9      OIL&GAS    2.205    0.71   +58.0%    +18.0%    352.5       477    ₹168,166      +3.9%     
  INDUSINDBK  10     PVT BNK    2.022    1.14   +44.6%    +16.2%    1,585.1     106    ₹168,024      +5.7%     
  PGHH        11     FMCG       1.963    0.36   +33.7%    +15.5%    7,228.0     23     ₹166,244      +1.0%     
  UPL         18     MFG        1.635    1.13   +55.5%    +10.9%    548.9       306    ₹167,977      +3.3%     
  HONAUT      19     MFG        1.633    0.51   +24.8%    +12.5%    12,237.3    13     ₹159,085      +2.6%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  DALMIABHA   28     MFG        03-Apr-17   1,969.1     2,588.4     76     ₹47,067       +31.5%      -1.1%     
  HDFC        3      PVT BNK    01-Mar-17   317.3       412.5       452    ₹43,026       +30.0%      +4.2%     
  HDFCBANK    4      PVT BNK    01-Mar-17   317.3       412.5       452    ₹43,026       +30.0%      +4.2%     
  IBULHSGFIN  22     FIN SVC    03-Apr-17   707.9       876.7       212    ₹35,783       +23.8%      +6.3%     
  KARURVYSYA  26     PVT BNK    03-Apr-17   73.3        90.0        2050   ₹34,234       +22.8%      +1.0%     
  VAKRANGEE   1      IT         01-Jun-17   174.1       199.9       902    ₹23,235       +14.8%      +1.3%     
  NATCOPHARM  33     HEALTH     03-Apr-17   803.3       899.3       187    ₹17,960       +12.0%      -1.8%     
  TVSMOTOR    5      AUTO       01-Jun-17   508.7       568.4       308    ₹18,390       +11.7%      +4.4%     
  MARUTI      9      AUTO       01-Jun-17   6,570.4     7,222.7     23     ₹15,003       +9.9%       +4.1%     
  HINDUNILVR  7      FMCG       01-Jun-17   940.4       1,017.0     167    ₹12,782       +8.1%       +2.8%     
  DHFL        25     FIN SVC    02-May-17   438.4       459.5       368    ₹7,767        +4.8%       +2.8%     
  GODREJIND   24     CONSUMP    03-Jul-17   640.8       649.5       249    ₹2,162        +1.4%       -1.3%     

  AFTER: Invested ₹3,296,207 | Cash ₹71,872 | Total ₹3,368,079 | Positions 19/20 | Slot ₹168,473

========================================================================
  REBALANCE #33  —  01 Sep 2017
  NAV: ₹3,405,144  |  Slot: ₹170,257  |  Cash: ₹71,872
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MARUTI      16     AUTO       01-Jun-17   6,570.4     7,254.3     23     ₹15,729       +10.4%    92d   
  UPL         73     MFG        01-Aug-17   548.9       511.2       306    ₹-11,535      -6.9%     31d   
  NATCOPHARM  116    HEALTH     03-Apr-17   803.3       667.2       187    ₹-25,452      -16.9%    151d  

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MGL         11     OIL&GAS    2.057    0.81   +83.1%    +9.4%     844.5       201    ₹169,742      +2.7%     
  BEL         14     DEFENCE    1.971    1.06   +60.1%    +13.0%    50.1        3395   ₹170,226      +5.9%     
  BRITANNIA   15     FMCG       1.937    0.86   +24.3%    +18.4%    1,877.6     90     ₹168,980      +2.5%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  DALMIABHA   35     MFG        03-Apr-17   1,969.1     2,690.0     76     ₹54,786       +36.6%      +2.3%     
  VAKRANGEE   1      IT         01-Jun-17   174.1       226.7       902    ₹47,398       +30.2%      +7.9%     
  KARURVYSYA  14     PVT BNK    03-Apr-17   73.3        94.3        2050   ₹43,045       +28.6%      +5.1%     
  HDFC        12     PVT BNK    01-Mar-17   317.3       405.7       452    ₹39,956       +27.9%      +0.5%     
  HDFCBANK    13     PVT BNK    01-Mar-17   317.3       405.7       452    ₹39,956       +27.9%      +0.5%     
  IBULHSGFIN  47     FIN SVC    03-Apr-17   707.9       880.4       212    ₹36,586       +24.4%      +1.9%     
  DHFL        23     FIN SVC    02-May-17   438.4       508.3       368    ₹25,723       +15.9%      +9.1%     
  TVSMOTOR    9      AUTO       01-Jun-17   508.7       579.7       308    ₹21,868       +14.0%      +3.2%     
  HINDUNILVR  22     FMCG       01-Jun-17   940.4       1,053.5     167    ₹18,887       +12.0%      +2.4%     
  IGL         7      OIL&GAS    01-Aug-17   104.2       111.7       1616   ₹12,038       +7.1%       +2.9%     
  HONAUT      17     MFG        01-Aug-17   12,237.3    13,107.3    13     ₹11,309       +7.1%       +3.6%     
  RAJESHEXPO  4      CON DUR    01-Aug-17   714.7       735.0       235    ₹4,772        +2.8%       +2.4%     
  PGHH        33     FMCG       01-Aug-17   7,228.0     7,401.7     23     ₹3,997        +2.4%       +1.7%     
  INDUSINDBK  21     PVT BNK    01-Aug-17   1,585.1     1,610.4     106    ₹2,678        +1.6%       +3.1%     
  RELIANCE    10     OIL&GAS    01-Aug-17   352.5       354.0       477    ₹682          +0.4%       +2.3%     
  GODREJIND   49     CONSUMP    03-Jul-17   640.8       621.0       249    ₹-4,923       -3.1%       +1.7%     

  AFTER: Invested ₹3,394,171 | Cash ₹10,369 | Total ₹3,404,540 | Positions 19/20 | Slot ₹170,257

========================================================================
  REBALANCE #34  —  03 Oct 2017
  NAV: ₹3,499,054  |  Slot: ₹174,953  |  Cash: ₹10,369
========================================================================
  [SECTOR CAP≤4] dropped: GUJGASLTD

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BEL         32     DEFENCE    01-Sep-17   50.1        46.5        3395   ₹-12,375      -7.3%     32d   
  GODREJIND   74     CONSUMP    03-Jul-17   640.8       588.7       249    ₹-12,973      -8.1%     92d   

  ENTRIES (1)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  GAIL        9      OIL&GAS    2.200    0.84   +58.0%    +19.5%    76.7        2280   ₹174,904      +8.2%     

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  DALMIABHA   42     MFG        03-Apr-17   1,969.1     2,715.0     76     ₹56,690       +37.9%      +1.4%     
  HDFC        10     PVT BNK    01-Mar-17   317.3       415.2       452    ₹44,235       +30.8%      +0.2%     
  HDFCBANK    11     PVT BNK    01-Mar-17   317.3       415.2       452    ₹44,235       +30.8%      +0.2%     
  VAKRANGEE   6      IT         01-Jun-17   174.1       222.4       902    ₹43,594       +27.8%      +0.6%     
  HONAUT      5      MFG        01-Aug-17   12,237.3    15,425.8    13     ₹41,450       +26.1%      +3.6%     
  IBULHSGFIN  25     FIN SVC    03-Apr-17   707.9       890.5       212    ₹38,720       +25.8%      -0.5%     
  KARURVYSYA  46     PVT BNK    03-Apr-17   73.3        91.7        2050   ₹37,719       +25.1%      -3.0%     
  IGL         1      OIL&GAS    01-Aug-17   104.2       129.7       1616   ₹41,151       +24.4%      +4.3%     
  DHFL        15     FIN SVC    02-May-17   438.4       543.3       368    ₹38,596       +23.9%      +0.8%     
  TVSMOTOR    9      AUTO       01-Jun-17   508.7       622.8       308    ₹35,150       +22.4%      +2.2%     
  RAJESHEXPO  3      CON DUR    01-Aug-17   714.7       815.3       235    ₹23,652       +14.1%      +6.7%     
  HINDUNILVR  27     FMCG       01-Jun-17   940.4       1,027.7     167    ₹14,576       +9.3%       -2.7%     
  MGL         17     OIL&GAS    01-Sep-17   844.5       877.9       201    ₹6,725        +4.0%       -0.5%     
  PGHH        37     FMCG       01-Aug-17   7,228.0     7,500.9     23     ₹6,276        +3.8%       +0.8%     
  BRITANNIA   21     FMCG       01-Sep-17   1,877.6     1,920.1     90     ₹3,826        +2.3%       +0.9%     
  INDUSINDBK  22     PVT BNK    01-Aug-17   1,585.1     1,611.1     106    ₹2,749        +1.6%       -0.2%     
  RELIANCE    18     OIL&GAS    01-Aug-17   352.5       351.0       477    ₹-724         -0.4%       -1.7%     

  AFTER: Invested ₹3,359,153 | Cash ₹139,694 | Total ₹3,498,847 | Positions 18/20 | Slot ₹174,953

========================================================================
  REBALANCE #35  —  01 Nov 2017
  NAV: ₹3,699,671  |  Slot: ₹184,984  |  Cash: ₹139,694
========================================================================
  [SECTOR CAP≤4] dropped: GUJGASLTD, PETRONET

  EXITS (1)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  KARURVYSYA  63     PVT BNK    03-Apr-17   73.3        89.3        2050   ₹32,670       +21.7%    212d  

  ENTRIES (1)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BEL         14     DEFENCE    1.943    1.19   +64.5%    +15.0%    52.8        3501   ₹184,981      +7.2%     

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  DALMIABHA   47     MFG        03-Apr-17   1,969.1     2,976.8     76     ₹76,583       +51.2%      +6.1%     
  DHFL        7      FIN SVC    02-May-17   438.4       640.9       368    ₹74,519       +46.2%      +9.2%     
  VAKRANGEE   3      IT         01-Jun-17   174.1       247.3       902    ₹65,981       +42.0%      +3.7%     
  IGL         4      OIL&GAS    01-Aug-17   104.2       139.1       1616   ₹56,384       +33.5%      +2.6%     
  TVSMOTOR    10     AUTO       01-Jun-17   508.7       675.8       308    ₹51,486       +32.9%      +2.4%     
  HDFC        25     PVT BNK    01-Mar-17   317.3       418.0       452    ₹45,516       +31.7%      +0.1%     
  HDFCBANK    26     PVT BNK    01-Mar-17   317.3       418.0       452    ₹45,516       +31.7%      +0.1%     
  IBULHSGFIN  48     FIN SVC    03-Apr-17   707.9       905.2       212    ₹41,835       +27.9%      -2.1%     
  HONAUT      5      MFG        01-Aug-17   12,237.3    15,600.8    13     ₹43,724       +27.5%      +1.2%     
  MGL         8      OIL&GAS    01-Sep-17   844.5       1,016.2     201    ₹34,509       +20.3%      +6.1%     
  RELIANCE    9      OIL&GAS    01-Aug-17   352.5       418.8       477    ₹31,618       +18.8%      +5.6%     
  HINDUNILVR  17     FMCG       01-Jun-17   940.4       1,096.9     167    ₹26,125       +16.6%      +1.5%     
  BRITANNIA   13     FMCG       01-Sep-17   1,877.6     2,078.1     90     ₹18,049       +10.7%      +2.9%     
  GAIL        14     OIL&GAS    03-Oct-17   76.7        82.5        2280   ₹13,116       +7.5%       +4.6%     
  PGHH        31     FMCG       01-Aug-17   7,228.0     7,761.5     23     ₹12,271       +7.4%       +1.3%     
  RAJESHEXPO  18     CON DUR    01-Aug-17   714.7       765.2       235    ₹11,878       +7.1%       -3.3%     
  INDUSINDBK  57     PVT BNK    01-Aug-17   1,585.1     1,563.1     106    ₹-2,335       -1.4%       -1.4%     

  AFTER: Invested ₹3,561,957 | Cash ₹137,494 | Total ₹3,699,451 | Positions 18/20 | Slot ₹184,984

========================================================================
  REBALANCE #36  —  01 Dec 2017
  NAV: ₹3,746,887  |  Slot: ₹187,344  |  Cash: ₹137,494
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  IBULHSGFIN  72     FIN SVC    03-Apr-17   707.9       831.7       212    ₹26,244       +17.5%    242d  
  MGL         51     OIL&GAS    01-Sep-17   844.5       888.1       201    ₹8,762        +5.2%     91d   
  INDUSINDBK  48     PVT BNK    01-Aug-17   1,585.1     1,580.8     106    ₹-464         -0.3%     122d  
  BEL         54     DEFENCE    01-Nov-17   52.8        52.5        3501   ₹-1,294       -0.7%     30d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BBTC        2      FMCG       3.353    0.99   +197.5%   +48.2%    1,478.3     126    ₹186,271      -0.9%     
  FRETAIL     3      CONSUMP    2.538    1.11   +348.9%   +1.6%     544.8       343    ₹186,849      +2.1%     
  BALKRISIND  7      MFG        2.244    0.53   +130.1%   +28.2%    978.3       191    ₹186,860      +4.4%     
  MARUTI      10     AUTO       2.111    1.09   +70.9%    +10.2%    7,994.1     23     ₹183,864      +2.5%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VAKRANGEE   1      IT         01-Jun-17   174.1       320.5       902    ₹132,025      +84.1%      +7.4%     
  DALMIABHA   26     MFG        03-Apr-17   1,969.1     3,120.1     76     ₹87,482       +58.5%      +2.4%     
  TVSMOTOR    4      AUTO       01-Jun-17   508.7       692.6       308    ₹56,638       +36.2%      +0.9%     
  HONAUT      5      MFG        01-Aug-17   12,237.3    16,639.1    13     ₹57,223       +36.0%      +1.7%     
  DHFL        10     FIN SVC    02-May-17   438.4       595.4       368    ₹57,764       +35.8%      -2.9%     
  IGL         6      OIL&GAS    01-Aug-17   104.2       139.5       1616   ₹57,051       +33.9%      +1.1%     
  HDFC        15     PVT BNK    01-Mar-17   317.3       424.2       452    ₹48,317       +33.7%      +0.4%     
  HDFCBANK    16     PVT BNK    01-Mar-17   317.3       424.2       452    ₹48,317       +33.7%      +0.4%     
  PGHH        9      FMCG       01-Aug-17   7,228.0     8,492.5     23     ₹29,083       +17.5%      +4.5%     
  HINDUNILVR  37     FMCG       01-Jun-17   940.4       1,090.3     167    ₹25,033       +15.9%      -1.0%     
  RELIANCE    20     OIL&GAS    01-Aug-17   352.5       400.2       477    ₹22,714       +13.5%      -1.3%     
  BRITANNIA   13     FMCG       01-Sep-17   1,877.6     2,126.2     90     ₹22,377       +13.2%      +1.0%     
  RAJESHEXPO  32     CON DUR    01-Aug-17   714.7       751.9       235    ₹8,743        +5.2%       -1.5%     
  GAIL        28     OIL&GAS    03-Oct-17   76.7        80.3        2280   ₹8,195        +4.7%       -0.8%     

  AFTER: Invested ₹3,647,175 | Cash ₹98,828 | Total ₹3,746,003 | Positions 18/20 | Slot ₹187,344

========================================================================
  REBALANCE #37  —  01 Jan 2018
  NAV: ₹3,954,918  |  Slot: ₹197,746  |  Cash: ₹98,828
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HDFC        32     PVT BNK    01-Mar-17   317.3       425.6       452    ₹48,971       +34.1%    306d  
  HDFCBANK    33     PVT BNK    01-Mar-17   317.3       425.6       452    ₹48,971       +34.1%    306d  
  DHFL        35     FIN SVC    02-May-17   438.4       578.1       368    ₹51,413       +31.9%    244d  
  GAIL        37     OIL&GAS    03-Oct-17   76.7        87.9        2280   ₹25,408       +14.5%    90d   
  RAJESHEXPO  47     CON DUR    01-Aug-17   714.7       798.3       235    ₹19,653       +11.7%    153d  

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIENSOL  4      ENERGY     2.602    0.03   +299.5%   +43.8%    219.7       900    ₹197,730      +3.9%     
  TITAN       5      CON DUR    2.587    0.12   +170.8%   +45.8%    828.8       238    ₹197,247      +2.1%     
  PAGEIND     7      MFG        2.325    -0.09  +96.1%    +36.9%    22,945.6    8      ₹183,565      +5.1%     
  DLF         8      REALTY     2.302    -0.25  +139.6%   +56.2%    240.2       823    ₹197,687      +6.0%     
  BIOCON      11     HEALTH     2.144    0.18   +76.4%    +62.0%    265.0       746    ₹197,698      +4.4%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  VAKRANGEE   1      IT         01-Jun-17   174.1       377.9       902    ₹183,854      +117.1%     +8.0%     
  DALMIABHA   13     MFG        03-Apr-17   1,969.1     3,208.1     76     ₹94,162       +62.9%      +0.9%     
  HONAUT      15     MFG        01-Aug-17   12,237.3    19,020.8    13     ₹88,185       +55.4%      +1.3%     
  TVSMOTOR    9      AUTO       01-Jun-17   508.7       732.5       308    ₹68,942       +44.0%      +1.1%     
  IGL         25     OIL&GAS    01-Aug-17   104.2       149.3       1616   ₹72,852       +43.3%      +3.7%     
  HINDUNILVR  17     FMCG       01-Jun-17   940.4       1,172.3     167    ₹38,721       +24.7%      +1.1%     
  PGHH        27     FMCG       01-Aug-17   7,228.0     8,460.5     23     ₹28,347       +17.1%      +0.7%     
  RELIANCE    29     OIL&GAS    01-Aug-17   352.5       400.0       477    ₹22,641       +13.5%      -1.1%     
  BALKRISIND  6      MFG        01-Dec-17   978.3       1,106.8     191    ₹24,536       +13.1%      +1.9%     
  MARUTI      3      AUTO       01-Dec-17   7,994.1     8,962.5     23     ₹22,273       +12.1%      +3.4%     
  BRITANNIA   28     FMCG       01-Sep-17   1,877.6     2,096.2     90     ₹19,681       +11.6%      -0.2%     
  BBTC        2      FMCG       01-Dec-17   1,478.3     1,618.0     126    ₹17,595       +9.4%       +5.6%     
  FRETAIL     10     CONSUMP    01-Dec-17   544.8       526.3       343    ₹-6,328       -3.4%       -0.3%     

  AFTER: Invested ₹3,844,561 | Cash ₹109,201 | Total ₹3,953,762 | Positions 18/20 | Slot ₹197,746

========================================================================
  REBALANCE #38  —  01 Feb 2018
  NAV: ₹3,689,408  |  Slot: ₹184,470  |  Cash: ₹109,201
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DALMIABHA   44     MFG        03-Apr-17   1,969.1     2,972.0     76     ₹76,222       +50.9%    304d  
  VAKRANGEE   41     IT         01-Jun-17   174.1       262.4       902    ₹79,628       +50.7%    245d  
  IGL         56     OIL&GAS    01-Aug-17   104.2       133.1       1616   ₹46,683       +27.7%    184d  
  TVSMOTOR    40     AUTO       01-Jun-17   508.7       641.5       308    ₹40,904       +26.1%    245d  
  DLF         34     REALTY     01-Jan-18   240.2       229.3       823    ₹-9,010       -4.6%     31d   
  BBTC        37     FMCG       01-Dec-17   1,478.3     1,383.9     126    ₹-11,902      -6.4%     62d   
  PAGEIND     52     MFG        01-Jan-18   22,945.6    19,379.6    8      ₹-28,528      -15.5%    31d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  PCJEWELLER  2      CON DUR    2.775    0.86   +149.8%   +39.3%    48.1        3832   ₹184,441      -8.4%     
  HDFC        4      PVT BNK    2.745    0.08   +55.1%    +9.3%     457.0       403    ₹184,176      +2.7%     
  HDFCBANK    5      PVT BNK    2.745    0.08   +55.1%    +9.3%     457.0       403    ₹184,176      +2.7%     
  LT          7      INFRA      2.543    0.15   +55.4%    +20.2%    1,278.9     144    ₹184,157      +6.1%     
  MPHASIS     8      IT         2.476    -0.04  +61.5%    +26.3%    722.3       255    ₹184,185      +9.6%     
  IBULHSGFIN  12     FIN SVC    2.130    0.42   +87.9%    +12.5%    1,010.8     182    ₹183,962      +5.3%     
  RAJESHEXPO  13     CON DUR    2.055    -0.23  +63.5%    +5.8%     814.1       226    ₹183,983      -0.1%     
  ABBOTINDIA  14     HEALTH     2.048    0.04   +26.5%    +28.4%    5,020.8     36     ₹180,749      +1.3%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HONAUT      29     MFG        01-Aug-17   12,237.3    16,698.3    13     ₹57,993       +36.5%      -4.5%     
  HINDUNILVR  10     FMCG       01-Jun-17   940.4       1,195.6     167    ₹42,616       +27.1%      +0.3%     
  RELIANCE    27     OIL&GAS    01-Aug-17   352.5       415.0       477    ₹29,793       +17.7%      -0.3%     
  BIOCON      1      HEALTH     01-Jan-18   265.0       305.0       746    ₹29,802       +15.1%      +6.0%     
  PGHH        26     FMCG       01-Aug-17   7,228.0     8,267.8     23     ₹23,916       +14.4%      -0.6%     
  BRITANNIA   33     FMCG       01-Sep-17   1,877.6     2,097.2     90     ₹19,771       +11.7%      +0.8%     
  MARUTI      6      AUTO       01-Dec-17   7,994.1     8,730.3     23     ₹16,933       +9.2%       -0.1%     
  BALKRISIND  3      MFG        01-Dec-17   978.3       1,047.3     191    ₹13,165       +7.0%       -1.5%     
  FRETAIL     11     CONSUMP    01-Dec-17   544.8       542.5       343    ₹-755         -0.4%       -1.3%     
  TITAN       9      CON DUR    01-Jan-18   828.8       807.9       238    ₹-4,963       -2.5%       -5.6%     
  ADANIENSOL  19     ENERGY     01-Jan-18   219.7       207.3       900    ₹-11,160      -5.6%       -4.1%     

  AFTER: Invested ₹3,656,718 | Cash ₹30,945 | Total ₹3,687,663 | Positions 19/20 | Slot ₹184,470

========================================================================
  REBALANCE #39  —  01 Mar 2018
  NAV: ₹3,540,460  |  Slot: ₹177,023  |  Cash: ₹30,945
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HONAUT      34     MFG        01-Aug-17   12,237.3    16,813.1    13     ₹59,485       +37.4%    212d  
  ABBOTINDIA  25     HEALTH     01-Feb-18   5,020.8     5,192.3     36     ₹6,175        +3.4%     28d   
  BALKRISIND  26     MFG        01-Dec-17   978.3       982.9       191    ₹865          +0.5%     90d   
  FRETAIL     36     CONSUMP    01-Dec-17   544.8       517.8       343    ₹-9,261       -5.0%     90d   
  TITAN       29     CON DUR    01-Jan-18   828.8       787.7       238    ₹-9,776       -5.0%     59d   
  IBULHSGFIN  28     FIN SVC    01-Feb-18   1,010.8     923.2       182    ₹-15,946      -8.7%     28d   
  PCJEWELLER  69     CON DUR    01-Feb-18   48.1        33.1        3832   ₹-57,526      -31.2%    28d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  VBL         2      FMCG       2.987    0.44   +65.9%    +27.9%    37.3        4745   ₹176,993      -1.6%     
  INDIGO      4      CONSUMP    2.470    0.34   +62.9%    +15.5%    1,317.1     134    ₹176,498      +4.3%     
  ASHOKLEY    7      AUTO       2.353    0.36   +58.0%    +16.6%    59.3        2985   ₹176,986      +5.4%     
  TCS         8      IT         2.208    0.14   +24.7%    +14.5%    1,230.7     143    ₹175,984      +1.1%     
  INFY        11     IT         2.123    0.10   +18.5%    +18.1%    456.5       387    ₹176,654      +1.9%     
  KOTAKBANK   12     PVT BNK    2.089    0.37   +35.2%    +6.8%     218.0       812    ₹177,016      +2.4%     
  BHARATFORG  13     DEFENCE    2.053    0.13   +47.4%    +12.1%    740.2       239    ₹176,904      +4.0%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  6      FMCG       01-Jun-17   940.4       1,154.7     167    ₹35,772       +22.8%      -1.1%     
  RELIANCE    20     OIL&GAS    01-Aug-17   352.5       417.0       477    ₹30,747       +18.3%      +1.6%     
  BIOCON      1      HEALTH     01-Jan-18   265.0       308.8       746    ₹32,649       +16.5%      +3.1%     
  BRITANNIA   9      FMCG       01-Sep-17   1,877.6     2,187.2     90     ₹27,868       +16.5%      +2.6%     
  PGHH        19     FMCG       01-Aug-17   7,228.0     8,405.9     23     ₹27,091       +16.3%      +0.8%     
  MARUTI      14     AUTO       01-Dec-17   7,994.1     8,239.6     23     ₹5,648        +3.1%       -0.8%     
  MPHASIS     5      IT         01-Feb-18   722.3       701.9       255    ₹-5,193       -2.8%       -0.9%     
  RAJESHEXPO  10     CON DUR    01-Feb-18   814.1       790.3       226    ₹-5,370       -2.9%       -1.3%     
  HDFC        22     PVT BNK    01-Feb-18   457.0       430.2       403    ₹-10,804      -5.9%       -0.8%     
  HDFCBANK    23     PVT BNK    01-Feb-18   457.0       430.2       403    ₹-10,804      -5.9%       -0.8%     
  ADANIENSOL  3      ENERGY     01-Jan-18   219.7       204.2       900    ₹-13,950      -7.1%       -0.7%     
  LT          17     INFRA      01-Feb-18   1,278.9     1,155.5     144    ₹-17,763      -9.6%       -1.2%     

  AFTER: Invested ₹3,493,341 | Cash ₹45,650 | Total ₹3,538,991 | Positions 19/20 | Slot ₹177,023

========================================================================
  REBALANCE #40  —  02 Apr 2018
  NAV: ₹3,501,042  |  Slot: ₹175,052  |  Cash: ₹45,650
========================================================================
  [SECTOR CAP≤4] dropped: GODREJCP

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  RELIANCE    36     OIL&GAS    01-Aug-17   352.5       392.6       477    ₹19,117       +11.4%    244d  
  BIOCON      23     HEALTH     01-Jan-18   265.0       294.8       746    ₹22,232       +11.2%    91d   
  MARUTI      30     AUTO       01-Dec-17   7,994.1     8,364.8     23     ₹8,526        +4.6%     122d  
  INFY        26     IT         01-Mar-18   456.5       447.2       387    ₹-3,592       -2.0%     32d   
  BHARATFORG  31     DEFENCE    01-Mar-18   740.2       680.7       239    ₹-14,215      -8.0%     32d   
  RAJESHEXPO  63     CON DUR    01-Feb-18   814.1       733.3       226    ₹-18,261      -9.9%     60d   

  ENTRIES (6)
  [52w filter blocked 1: STRTECH(-21.2%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  DBL         1      INFRA      3.529    1.01   +219.8%   +10.8%    1,099.5     159    ₹174,825      +12.7%    
  TITAN       4      CON DUR    2.740    0.45   +111.8%   +11.2%    917.9       190    ₹174,396      +6.9%     
  CHOLAFIN    7      FIN SVC    2.479    0.43   +49.0%    +14.9%    287.4       609    ₹175,019      +2.6%     
  INDUSINDBK  11     PVT BNK    2.348    0.31   +30.4%    +9.3%     1,717.3     101    ₹173,449      +3.7%     
  FRETAIL     13     CONSUMP    2.225    0.52   +121.5%   +4.7%     549.3       318    ₹174,677      +2.6%     
  HCLTECH     20     IT         1.967    0.16   +13.7%    +11.1%    380.7       459    ₹174,735      +2.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  14     FMCG       01-Jun-17   940.4       1,178.2     167    ₹39,711       +25.3%      +2.2%     
  BRITANNIA   3      FMCG       01-Sep-17   1,877.6     2,259.4     90     ₹34,366       +20.3%      +4.5%     
  PGHH        18     FMCG       01-Aug-17   7,228.0     8,386.1     23     ₹26,637       +16.0%      +0.5%     
  ASHOKLEY    2      AUTO       01-Mar-18   59.3        62.3        2985   ₹8,978        +5.1%       +3.0%     
  KOTAKBANK   17     PVT BNK    01-Mar-18   218.0       218.2       812    ₹137          +0.1%       +3.0%     
  INDIGO      22     CONSUMP    01-Mar-18   1,317.1     1,313.7     134    ₹-457         -0.3%       +4.0%     
  VBL         10     FMCG       01-Mar-18   37.3        37.0        4745   ₹-1,220       -0.7%       +2.3%     
  HDFC        8      PVT BNK    01-Feb-18   457.0       443.3       403    ₹-5,545       -3.0%       +2.9%     
  HDFCBANK    9      PVT BNK    01-Feb-18   457.0       443.3       403    ₹-5,545       -3.0%       +2.9%     
  MPHASIS     6      IT         01-Feb-18   722.3       695.3       255    ₹-6,878       -3.7%       +0.0%     
  TCS         16     IT         01-Mar-18   1,230.7     1,178.8     143    ₹-7,418       -4.2%       +0.4%     
  LT          19     INFRA      01-Feb-18   1,278.9     1,173.7     144    ₹-15,138      -8.2%       +2.5%     
  ADANIENSOL  12     ENERGY     01-Jan-18   219.7       193.6       900    ₹-23,490      -11.9%      +0.0%     

  AFTER: Invested ₹3,401,417 | Cash ₹98,381 | Total ₹3,499,798 | Positions 19/20 | Slot ₹175,052

========================================================================
  REBALANCE #41  —  02 May 2018
  NAV: ₹3,707,352  |  Slot: ₹185,368  |  Cash: ₹98,381
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VBL         58     FMCG       01-Mar-18   37.3        38.6        4745   ₹6,101        +3.4%     62d   
  HCLTECH     52     IT         02-Apr-18   380.7       388.8       459    ₹3,746        +2.1%     30d   
  LT          57     INFRA      01-Feb-18   1,278.9     1,232.8     144    ₹-6,636       -3.6%     90d   
  ADANIENSOL  51     ENERGY     01-Jan-18   219.7       167.1       900    ₹-47,340      -23.9%    121d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  DMART       2      FMCG       3.302    0.28   +100.0%   +29.0%    1,495.8     123    ₹183,990      +2.6%     
  PIDILITIND  3      MFG        3.257    0.48   +52.7%    +25.7%    530.4       349    ₹185,093      +6.3%     
  SRF         11     MFG        2.397    0.74   +35.3%    +28.1%    455.9       406    ₹185,102      +7.5%     
  M&M         14     AUTO       2.160    0.62   +28.8%    +13.9%    793.7       233    ₹184,926      +5.6%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  8      FMCG       01-Jun-17   940.4       1,283.4     167    ₹57,279       +36.5%      +2.4%     
  BRITANNIA   5      FMCG       01-Sep-17   1,877.6     2,400.1     90     ₹47,027       +27.8%      +2.7%     
  PGHH        13     FMCG       01-Aug-17   7,228.0     8,648.1     23     ₹32,662       +19.6%      +0.6%     
  CHOLAFIN    7      FIN SVC    02-Apr-18   287.4       333.6       609    ₹28,144       +16.1%      +5.8%     
  TCS         10     IT         01-Mar-18   1,230.7     1,417.8     143    ₹26,768       +15.2%      +6.9%     
  KOTAKBANK   9      PVT BNK    01-Mar-18   218.0       250.1       812    ₹26,035       +14.7%      +8.3%     
  ASHOKLEY    1      AUTO       01-Mar-18   59.3        67.9        2985   ₹25,741       +14.5%      +4.2%     
  MPHASIS     4      IT         01-Feb-18   722.3       824.0       255    ₹25,943       +14.1%      +6.5%     
  FRETAIL     22     CONSUMP    02-Apr-18   549.3       596.6       318    ₹15,041       +8.6%       +0.9%     
  INDUSINDBK  20     PVT BNK    02-Apr-18   1,717.3     1,777.9     101    ₹6,123        +3.5%       +1.1%     
  DBL         6      INFRA      02-Apr-18   1,099.5     1,123.9     159    ₹3,875        +2.2%       +0.8%     
  TITAN       12     CON DUR    02-Apr-18   917.9       937.7       190    ₹3,768        +2.2%       +0.8%     
  INDIGO      38     CONSUMP    01-Mar-18   1,317.1     1,327.7     134    ₹1,418        +0.8%       -6.1%     
  HDFC        29     PVT BNK    01-Feb-18   457.0       452.0       403    ₹-2,003       -1.1%       +2.1%     
  HDFCBANK    30     PVT BNK    01-Feb-18   457.0       452.0       403    ₹-2,003       -1.1%       +2.1%     

  AFTER: Invested ₹3,658,595 | Cash ₹47,880 | Total ₹3,706,475 | Positions 19/20 | Slot ₹185,368

========================================================================
  REBALANCE #42  —  01 Jun 2018
  NAV: ₹3,653,875  |  Slot: ₹182,694  |  Cash: ₹47,880
========================================================================
  [SECTOR CAP≤4] dropped: COLPAL

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PGHH        44     FMCG       01-Aug-17   7,228.0     8,342.6     23     ₹25,637       +15.4%    304d  
  CHOLAFIN    28     FIN SVC    02-Apr-18   287.4       308.5       609    ₹12,840       +7.3%     60d   
  INDIGO      87     CONSUMP    01-Mar-18   1,317.1     1,163.5     134    ₹-20,582      -11.7%    92d   
  SRF         —      MFG        02-May-18   455.9       362.8       406    ₹-37,825      -20.4%    30d   
  DBL         42     INFRA      02-Apr-18   1,099.5     847.7       159    ₹-40,033      -22.9%    60d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    1      CONSUMP    3.392    0.45   +166.5%   +23.0%    245.3       744    ₹182,500      -1.3%     
  COFORGE     4      IT         2.740    0.72   +120.6%   +32.7%    201.6       906    ₹182,606      +3.3%     
  TECHM       7      IT         2.562    0.03   +89.2%    +14.4%    516.6       353    ₹182,364      +2.4%     
  DABUR       9      FMCG       2.551    0.17   +39.8%    +19.0%    354.9       514    ₹182,443      +3.3%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  3      FMCG       01-Jun-17   940.4       1,385.4     167    ₹74,301       +47.3%      +2.3%     
  BRITANNIA   2      FMCG       01-Sep-17   1,877.6     2,571.5     90     ₹62,452       +37.0%      +2.9%     
  KOTAKBANK   10     PVT BNK    01-Mar-18   218.0       262.2       812    ₹35,914       +20.3%      +3.5%     
  MPHASIS     8      IT         01-Feb-18   722.3       868.5       255    ₹37,270       +20.2%      +0.6%     
  TCS         19     IT         01-Mar-18   1,230.7     1,415.4     143    ₹26,420       +15.0%      +0.3%     
  ASHOKLEY    27     AUTO       01-Mar-18   59.3        63.5        2985   ₹12,431       +7.0%       +0.5%     
  FRETAIL     25     CONSUMP    02-Apr-18   549.3       586.9       318    ₹11,957       +6.8%       +1.4%     
  HDFC        13     PVT BNK    01-Feb-18   457.0       487.5       403    ₹12,296       +6.7%       +4.9%     
  HDFCBANK    14     PVT BNK    01-Feb-18   457.0       487.5       403    ₹12,296       +6.7%       +4.9%     
  INDUSINDBK  22     PVT BNK    02-Apr-18   1,717.3     1,822.8     101    ₹10,650       +6.1%       +0.9%     
  M&M         11     AUTO       02-May-18   793.7       829.0       233    ₹8,238        +4.5%       +4.7%     
  DMART       6      FMCG       02-May-18   1,495.8     1,531.1     123    ₹4,336        +2.4%       +3.3%     
  PIDILITIND  5      MFG        02-May-18   530.4       538.3       349    ₹2,769        +1.5%       +0.5%     
  TITAN       18     CON DUR    02-Apr-18   917.9       875.0       190    ₹-8,146       -4.7%       -2.9%     

  AFTER: Invested ₹3,518,185 | Cash ₹134,823 | Total ₹3,653,008 | Positions 18/20 | Slot ₹182,694

========================================================================
  REBALANCE #43  —  02 Jul 2018
  NAV: ₹3,664,011  |  Slot: ₹183,201  |  Cash: ₹134,823
========================================================================
  [SECTOR CAP≤4] dropped: INFY

  [REGIME OFF] Nifty 200 5,679.7 < SMA200 5,680.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  3      FMCG       01-Jun-17   940.4       1,443.8     167    ₹84,057       +53.5%      +2.6%     
  BRITANNIA   2      FMCG       01-Sep-17   1,877.6     2,757.5     90     ₹79,195       +46.9%      +3.8%     
  MPHASIS     6      IT         01-Feb-18   722.3       903.3       255    ₹46,169       +25.1%      +1.5%     
  TCS         4      IT         01-Mar-18   1,230.7     1,512.6     143    ₹40,311       +22.9%      +2.2%     
  KOTAKBANK   7      PVT BNK    01-Mar-18   218.0       266.2       812    ₹39,137       +22.1%      +1.3%     
  JUBLFOOD    1      CONSUMP    01-Jun-18   245.3       276.3       744    ₹23,072       +12.6%      +3.4%     
  INDUSINDBK  24     PVT BNK    02-Apr-18   1,717.3     1,861.7     101    ₹14,581       +8.4%       +0.3%     
  HDFC        17     PVT BNK    01-Feb-18   457.0       478.9       403    ₹8,819        +4.8%       +0.3%     
  HDFCBANK    18     PVT BNK    01-Feb-18   457.0       478.9       403    ₹8,819        +4.8%       +0.3%     
  FRETAIL     36     CONSUMP    02-Apr-18   549.3       573.2       318    ₹7,584        +4.3%       +1.6%     
  M&M         15     AUTO       02-May-18   793.7       815.3       233    ₹5,035        +2.7%       -1.2%     
  DMART       12     FMCG       02-May-18   1,495.8     1,525.9     123    ₹3,696        +2.0%       +1.8%     
  COFORGE     11     IT         01-Jun-18   201.6       201.1       906    ₹-412         -0.2%       +0.8%     
  DABUR       13     FMCG       01-Jun-18   354.9       352.1       514    ₹-1,439       -0.8%       -0.2%     
  PIDILITIND  20     MFG        02-May-18   530.4       509.6       349    ₹-7,242       -3.9%       -0.6%     
  TITAN       31     CON DUR    02-Apr-18   917.9       872.5       190    ₹-8,626       -4.9%       +1.2%     
  TECHM       16     IT         01-Jun-18   516.6       484.2       353    ₹-11,431      -6.3%       -4.4%     
  ASHOKLEY    61     AUTO       01-Mar-18   59.3        53.5        2985   ₹-17,140      -9.7% ⚠     -6.5%     
  ⚠  WAZ < 0 (momentum below universe mean): ASHOKLEY

  AFTER: Invested ₹3,529,188 | Cash ₹134,823 | Total ₹3,664,011 | Positions 18/20 | Slot ₹183,201

========================================================================
  REBALANCE #44  —  01 Aug 2018
  NAV: ₹3,832,928  |  Slot: ₹191,646  |  Cash: ₹134,823
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  KOTAKBANK   28     PVT BNK    01-Mar-18   218.0       261.3       812    ₹35,166       +19.9%    153d  
  INDUSINDBK  34     PVT BNK    02-Apr-18   1,717.3     1,919.4     101    ₹20,415       +11.8%    121d  
  FRETAIL     61     CONSUMP    02-Apr-18   549.3       540.8       318    ₹-2,703       -1.5%     121d  
  TITAN       36     CON DUR    02-Apr-18   917.9       895.8       190    ₹-4,200       -2.4%     121d  
  ASHOKLEY    104    AUTO       01-Mar-18   59.3        51.0        2985   ₹-24,894      -14.1%    153d  

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BAJFINANCE  2      FIN SVC    3.090    0.74   +60.3%    +46.8%    264.0       725    ₹191,411      +5.8%     
  ABBOTINDIA  4      HEALTH     2.803    0.29   +80.6%    +23.2%    6,983.6     27     ₹188,558      +4.1%     
  RELIANCE    6      OIL&GAS    2.762    0.79   +50.6%    +25.8%    527.8       363    ₹191,584      +8.4%     
  PAGEIND     9      MFG        2.515    0.68   +82.6%    +26.9%    27,338.4    7      ₹191,369      +4.9%     
  BAJAJFINSV  10     FIN SVC    2.484    0.82   +41.9%    +30.7%    696.2       275    ₹191,464      +5.8%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  3      FMCG       01-Jun-17   940.4       1,523.6     167    ₹97,386       +62.0%      +3.2%     
  BRITANNIA   1      FMCG       01-Sep-17   1,877.6     2,882.3     90     ₹90,429       +53.5%      +2.0%     
  MPHASIS     5      IT         01-Feb-18   722.3       1,006.5     255    ₹72,471       +39.3%      +6.6%     
  TCS         8      IT         01-Mar-18   1,230.7     1,618.1     143    ₹55,397       +31.5%      +1.6%     
  DABUR       13     FMCG       01-Jun-18   354.9       403.3       514    ₹24,831       +13.6%      +11.8%    
  COFORGE     7      IT         01-Jun-18   201.6       227.3       906    ₹23,370       +12.8%      +6.2%     
  JUBLFOOD    11     CONSUMP    01-Jun-18   245.3       273.8       744    ₹21,183       +11.6%      -1.1%     
  DMART       15     FMCG       02-May-18   1,495.8     1,669.7     123    ₹21,384       +11.6%      +6.2%     
  HDFC        20     PVT BNK    01-Feb-18   457.0       498.6       403    ₹16,773       +9.1%       +0.0%     
  HDFCBANK    21     PVT BNK    01-Feb-18   457.0       498.6       403    ₹16,773       +9.1%       +0.0%     
  M&M         22     AUTO       02-May-18   793.7       861.9       233    ₹15,886       +8.6%       +1.6%     
  PIDILITIND  23     MFG        02-May-18   530.4       543.6       349    ₹4,632        +2.5%       +3.8%     
  TECHM       17     IT         01-Jun-18   516.6       513.2       353    ₹-1,198       -0.7%       +5.1%     

  AFTER: Invested ₹3,752,183 | Cash ₹79,612 | Total ₹3,831,795 | Positions 18/20 | Slot ₹191,646

========================================================================
  REBALANCE #45  —  03 Sep 2018
  NAV: ₹3,950,244  |  Slot: ₹197,512  |  Cash: ₹79,612
========================================================================
  [SECTOR CAP≤4] dropped: INFY

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DMART       38     FMCG       02-May-18   1,495.8     1,604.7     123    ₹13,382       +7.3%     124d  
  HDFC        51     PVT BNK    01-Feb-18   457.0       479.3       403    ₹8,986        +4.9%     214d  
  HDFCBANK    52     PVT BNK    01-Feb-18   457.0       479.3       403    ₹8,986        +4.9%     214d  

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BATAINDIA   3      CON DUR    2.845    0.84   +55.6%    +42.2%    993.8       198    ₹196,766      +3.7%     
  GODREJCP    7      FMCG       2.565    0.30   +56.4%    +26.5%    894.1       220    ₹196,705      +3.6%     
  HAVELLS     13     CON DUR    2.381    0.97   +48.7%    +34.5%    680.5       290    ₹197,355      +4.0%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  21     FMCG       01-Jun-17   940.4       1,492.7     167    ₹92,222       +58.7%      -2.8%     
  BRITANNIA   11     FMCG       01-Sep-17   1,877.6     2,916.1     90     ₹93,467       +55.3%      -0.8%     
  MPHASIS     12     IT         01-Feb-18   722.3       1,042.5     255    ₹81,641       +44.3%      +2.9%     
  TCS         10     IT         01-Mar-18   1,230.7     1,680.6     143    ₹64,341       +36.6%      +1.3%     
  COFORGE     2      IT         01-Jun-18   201.6       251.5       906    ₹45,215       +24.8%      +4.5%     
  DABUR       9      FMCG       01-Jun-18   354.9       435.1       514    ₹41,183       +22.6%      +2.1%     
  JUBLFOOD    5      CONSUMP    01-Jun-18   245.3       299.4       744    ₹40,234       +22.0%      +1.0%     
  PAGEIND     6      MFG        01-Aug-18   27,338.4    31,091.6    7      ₹26,272       +13.7%      +1.8%     
  M&M         34     AUTO       02-May-18   793.7       878.6       233    ₹19,785       +10.7%      -0.9%     
  ABBOTINDIA  1      HEALTH     01-Aug-18   6,983.6     7,640.6     27     ₹17,738       +9.4%       +6.4%     
  TECHM       20     IT         01-Jun-18   516.6       561.9       353    ₹15,972       +8.8%       +5.5%     
  PIDILITIND  37     MFG        02-May-18   530.4       563.5       349    ₹11,579       +6.3%       +2.7%     
  RELIANCE    4      OIL&GAS    01-Aug-18   527.8       544.1       363    ₹5,913        +3.1%       -0.1%     
  BAJFINANCE  14     FIN SVC    01-Aug-18   264.0       264.1       725    ₹88           +0.0%       -3.5%     
  BAJAJFINSV  36     FIN SVC    01-Aug-18   696.2       662.6       275    ₹-9,237       -4.8%       -3.8%     

  AFTER: Invested ₹3,877,760 | Cash ₹71,783 | Total ₹3,949,542 | Positions 18/20 | Slot ₹197,512

========================================================================
  REBALANCE #46  —  01 Oct 2018
  NAV: ₹3,627,281  |  Slot: ₹181,364  |  Cash: ₹71,783
========================================================================
  [SECTOR CAP≤4] dropped: INFY, WIPRO, HCLTECH

  [REGIME OFF] Nifty 200 5,772.3 < SMA200 5,812.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  23     FMCG       01-Jun-17   940.4       1,442.8     167    ₹83,889       +53.4%      -0.1%     
  TCS         1      IT         01-Mar-18   1,230.7     1,846.5     143    ₹88,065       +50.0%      +6.3%     
  BRITANNIA   29     FMCG       01-Sep-17   1,877.6     2,595.3     90     ₹64,596       +38.2%      -2.7%     
  MPHASIS     8      IT         01-Feb-18   722.3       968.9       255    ₹62,885       +34.1%      -3.5%     
  DABUR       11     FMCG       01-Jun-18   354.9       409.8       514    ₹28,188       +15.5%      -2.0%     
  PAGEIND     5      MFG        01-Aug-18   27,338.4    30,462.0    7      ₹21,865       +11.4%      +0.7%     
  TECHM       6      IT         01-Jun-18   516.6       572.3       353    ₹19,669       +10.8%      +1.7%     
  COFORGE     7      IT         01-Jun-18   201.6       214.9       906    ₹12,100       +6.6%       -6.0%     
  RELIANCE    3      OIL&GAS    01-Aug-18   527.8       545.2       363    ₹6,323        +3.3%       -0.5%     
  ABBOTINDIA  10     HEALTH     01-Aug-18   6,983.6     6,960.2     27     ₹-632         -0.3%       -5.4%     
  JUBLFOOD    22     CONSUMP    01-Jun-18   245.3       243.8       744    ₹-1,096       -0.6%       -7.2%     
  M&M         36     AUTO       02-May-18   793.7       785.7       233    ₹-1,857       -1.0%       -7.5%     
  PIDILITIND  27     MFG        02-May-18   530.4       503.3       349    ₹-9,426       -5.1%       -5.4%     
  BATAINDIA   14     CON DUR    03-Sep-18   993.8       903.9       198    ₹-17,789      -9.0%       -2.9%     
  BAJAJFINSV  45     FIN SVC    01-Aug-18   696.2       585.8       275    ₹-30,357      -15.9%      -7.3%     
  GODREJCP    34     FMCG       03-Sep-18   894.1       745.4       220    ₹-32,716      -16.6%      -4.4%     
  HAVELLS     24     CON DUR    03-Sep-18   680.5       565.4       290    ₹-33,402      -16.9%      -7.2%     
  BAJFINANCE  50     FIN SVC    01-Aug-18   264.0       214.1       725    ₹-36,155      -18.9%      -10.5%    

  AFTER: Invested ₹3,555,498 | Cash ₹71,783 | Total ₹3,627,281 | Positions 18/20 | Slot ₹181,364

========================================================================
  REBALANCE #47  —  01 Nov 2018
  NAV: ₹3,393,109  |  Slot: ₹169,655  |  Cash: ₹71,783
========================================================================
  [SECTOR CAP≤4] dropped: INFY

  [REGIME OFF] Nifty 200 5,505.5 < SMA200 5,780.7 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  19     FMCG       01-Jun-17   940.4       1,420.4     167    ₹80,151       +51.0%      +1.9%     
  BRITANNIA   48     FMCG       01-Sep-17   1,877.6     2,479.6     90     ₹54,183       +32.1%      +0.1%     
  TCS         8      IT         01-Mar-18   1,230.7     1,588.0     143    ₹51,104       +29.0%      -0.3%     
  MPHASIS     25     IT         01-Feb-18   722.3       812.7       255    ₹23,052       +12.5%      -7.1%     
  COFORGE     9      IT         01-Jun-18   201.6       220.4       906    ₹17,063       +9.3%       -0.2%     
  TECHM       4      IT         01-Jun-18   516.6       543.1       353    ₹9,336        +5.1%       +3.1%     
  ABBOTINDIA  2      HEALTH     01-Aug-18   6,983.6     6,916.3     27     ₹-1,818       -1.0%       +3.1%     
  DABUR       28     FMCG       01-Jun-18   354.9       345.8       514    ₹-4,708       -2.6%       -8.2%     
  PAGEIND     18     MFG        01-Aug-18   27,338.4    26,175.8    7      ₹-8,138       -4.3%       -2.8%     
  BATAINDIA   11     CON DUR    03-Sep-18   993.8       948.3       198    ₹-9,005       -4.6%       +8.7%     
  HAVELLS     15     CON DUR    03-Sep-18   680.5       612.7       290    ₹-19,669      -10.0%      +4.8%     
  PIDILITIND  43     MFG        02-May-18   530.4       472.5       349    ₹-20,199      -10.9%      +0.3%     
  BAJFINANCE  27     FIN SVC    01-Aug-18   264.0       234.5       725    ₹-21,420      -11.2%      +5.0%     
  RELIANCE    41     OIL&GAS    01-Aug-18   527.8       467.5       363    ₹-21,868      -11.4%      -3.7%     
  JUBLFOOD    54     CONSUMP    01-Jun-18   245.3       217.0       744    ₹-21,024      -11.5%      -5.0%     
  M&M         73     AUTO       02-May-18   793.7       700.8       233    ₹-21,630      -11.7% ⚠    -1.9%     
  BAJAJFINSV  89     FIN SVC    01-Aug-18   696.2       536.5       275    ₹-43,925      -22.9%      -2.9%     
  GODREJCP    71     FMCG       03-Sep-18   894.1       660.0       220    ₹-51,507      -26.2% ⚠    -4.1%     
  ⚠  WAZ < 0 (momentum below universe mean): GODREJCP, M&M

  AFTER: Invested ₹3,321,326 | Cash ₹71,783 | Total ₹3,393,109 | Positions 18/20 | Slot ₹169,655

========================================================================
  REBALANCE #48  —  03 Dec 2018
  NAV: ₹3,594,532  |  Slot: ₹179,727  |  Cash: ₹71,783
========================================================================
  [SECTOR CAP≤4] dropped: WIPRO, INFY

  [REGIME OFF] Nifty 200 5,742.8 < SMA200 5,751.6 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  2      FMCG       01-Jun-17   940.4       1,613.0     167    ₹112,309      +71.5%      +7.1%     
  BRITANNIA   33     FMCG       01-Sep-17   1,877.6     2,748.7     90     ₹78,404       +46.4%      +4.2%     
  TCS         8      IT         01-Mar-18   1,230.7     1,626.3     143    ₹56,577       +32.1%      +3.4%     
  MPHASIS     42     IT         01-Feb-18   722.3       834.0       255    ₹28,495       +15.5%      +3.6%     
  DABUR       51     FMCG       01-Jun-18   354.9       386.6       514    ₹16,249       +8.9%       +2.7%     
  JUBLFOOD    28     CONSUMP    01-Jun-18   245.3       261.4       744    ₹11,945       +6.5%       +10.2%    
  PIDILITIND  11     MFG        02-May-18   530.4       556.8       349    ₹9,234        +5.0%       +3.7%     
  TECHM       12     IT         01-Jun-18   516.6       537.0       353    ₹7,208        +4.0%       +1.4%     
  ABBOTINDIA  3      HEALTH     01-Aug-18   6,983.6     7,082.9     27     ₹2,681        +1.4%       +4.3%     
  COFORGE     24     IT         01-Jun-18   201.6       203.2       906    ₹1,457        +0.8%       -1.8%     
  BATAINDIA   14     CON DUR    03-Sep-18   993.8       993.5       198    ₹-55          -0.0%       +6.7%     
  RELIANCE    45     OIL&GAS    01-Aug-18   527.8       511.9       363    ₹-5,768       -3.0%       +2.7%     
  HAVELLS     19     CON DUR    03-Sep-18   680.5       657.5       290    ₹-6,694       -3.4%       +4.1%     
  BAJFINANCE  37     FIN SVC    01-Aug-18   264.0       243.3       725    ₹-15,055      -7.9%       +4.3%     
  PAGEIND     73     MFG        01-Aug-18   27,338.4    24,355.1    7      ₹-20,883      -10.9% ⚠    -3.0%     
  M&M         96     AUTO       02-May-18   793.7       705.5       233    ₹-20,550      -11.1% ⚠    -1.0%     
  BAJAJFINSV  56     FIN SVC    01-Aug-18   696.2       597.5       275    ₹-27,140      -14.2%      +4.0%     
  GODREJCP    64     FMCG       03-Sep-18   894.1       725.9       220    ₹-37,011      -18.8% ⚠    +5.4%     
  ⚠  WAZ < 0 (momentum below universe mean): GODREJCP, PAGEIND, M&M

  AFTER: Invested ₹3,522,749 | Cash ₹71,783 | Total ₹3,594,532 | Positions 18/20 | Slot ₹179,727

========================================================================
  REBALANCE #49  —  01 Jan 2019
  NAV: ₹3,603,784  |  Slot: ₹180,189  |  Cash: ₹71,783
========================================================================
  [SECTOR CAP≤4] dropped: DMART, VBL, MARICO

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TCS         29     IT         01-Mar-18   1,230.7     1,561.0     143    ₹47,239       +26.8%    306d  
  MPHASIS     53     IT         01-Feb-18   722.3       840.0       255    ₹30,018       +16.3%    334d  
  DABUR       40     FMCG       01-Jun-18   354.9       394.9       514    ₹20,531       +11.3%    214d  
  PIDILITIND  27     MFG        02-May-18   530.4       532.7       349    ₹823          +0.4%     244d  
  RELIANCE    60     OIL&GAS    01-Aug-18   527.8       496.2       363    ₹-11,464      -6.0%     153d  
  M&M         92     AUTO       02-May-18   793.7       716.9       233    ₹-17,894      -9.7%     244d  
  PAGEIND     110    MFG        01-Aug-18   27,338.4    22,837.2    7      ₹-31,509      -16.5%    153d  

  ENTRIES (7)
  [52w filter blocked 1: INDIGO(-22.8%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  COLPAL      1      FMCG       3.054    0.13   +24.0%    +20.9%    1,115.6     161    ₹179,614      +2.9%     
  HDFC        7      PVT BNK    2.283    0.07   +15.7%    +8.7%     496.2       363    ₹180,115      +2.1%     
  HDFCBANK    8      PVT BNK    2.283    0.07   +15.7%    +8.7%     496.2       363    ₹180,115      +2.1%     
  KOTAKBANK   13     PVT BNK    2.090    -0.29  +23.8%    +9.5%     248.9       723    ₹179,966      +1.6%     
  LT          14     INFRA      2.049    0.06   +16.0%    +10.1%    1,283.0     140    ₹179,621      +1.7%     
  MUTHOOTFIN  16     FIN SVC    2.003    0.01   +11.2%    +23.2%    448.5       401    ₹179,839      +5.5%     
  ASIANPAINT  17     CONSUMP    1.975    -0.05  +21.4%    +6.8%     1,283.9     140    ₹179,742      +1.3%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  2      FMCG       01-Jun-17   940.4       1,591.3     167    ₹108,695      +69.2%      -0.1%     
  BRITANNIA   5      FMCG       01-Sep-17   1,877.6     2,756.3     90     ₹79,087       +46.8%      +0.5%     
  BATAINDIA   3      CON DUR    03-Sep-18   993.8       1,051.6     198    ₹11,455       +5.8%       +3.1%     
  COFORGE     10     IT         01-Jun-18   201.6       212.0       906    ₹9,490        +5.2%       +1.9%     
  TECHM       19     IT         01-Jun-18   516.6       541.9       353    ₹8,910        +4.9%       +1.6%     
  JUBLFOOD    23     CONSUMP    01-Jun-18   245.3       245.8       744    ₹380          +0.2%       -0.5%     
  ABBOTINDIA  22     HEALTH     01-Aug-18   6,983.6     6,904.8     27     ₹-2,127       -1.1%       +1.1%     
  BAJFINANCE  4      FIN SVC    01-Aug-18   264.0       257.6       725    ₹-4,636       -2.4%       +4.3%     
  HAVELLS     11     CON DUR    03-Sep-18   680.5       660.0       290    ₹-5,950       -3.0%       +1.0%     
  BAJAJFINSV  20     FIN SVC    01-Aug-18   696.2       649.6       275    ₹-12,824      -6.7%       +4.8%     
  GODREJCP    26     FMCG       03-Sep-18   894.1       759.7       220    ₹-29,569      -15.0%      +1.5%     

  AFTER: Invested ₹3,457,686 | Cash ₹144,604 | Total ₹3,602,289 | Positions 18/20 | Slot ₹180,189

========================================================================
  REBALANCE #50  —  01 Feb 2019
  NAV: ₹3,635,489  |  Slot: ₹181,774  |  Cash: ₹144,604
========================================================================
  [SECTOR CAP≤4] dropped: MARICO

  [REGIME OFF] Nifty 200 5,701.6 < SMA200 5,767.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  5      FMCG       01-Jun-17   940.4       1,589.7     167    ₹108,429      +69.0%      +1.8%     
  BRITANNIA   1      FMCG       01-Sep-17   1,877.6     2,887.7     90     ₹90,915       +53.8%      +2.4%     
  COFORGE     17     IT         01-Jun-18   201.6       243.0       906    ₹37,508       +20.5%      +5.6%     
  TECHM       30     IT         01-Jun-18   516.6       562.6       353    ₹16,225       +8.9%       +3.9%     
  JUBLFOOD    14     CONSUMP    01-Jun-18   245.3       266.9       744    ₹16,039       +8.8%       +10.6%    
  BATAINDIA   3      CON DUR    03-Sep-18   993.8       1,068.3     198    ₹14,765       +7.5%       +1.3%     
  ASIANPAINT  2      CONSUMP    01-Jan-19   1,283.9     1,364.0     140    ₹11,224       +6.2%       +4.3%     
  ABBOTINDIA  8      HEALTH     01-Aug-18   6,983.6     7,377.5     27     ₹10,634       +5.6%       +1.0%     
  HAVELLS     16     CON DUR    03-Sep-18   680.5       700.2       290    ₹5,716        +2.9%       +5.4%     
  KOTAKBANK   19     PVT BNK    01-Jan-19   248.9       250.0       723    ₹756          +0.4%       +0.7%     
  COLPAL      9      FMCG       01-Jan-19   1,115.6     1,090.0     161    ₹-4,131       -2.3%       -0.1%     
  HDFC        22     PVT BNK    01-Jan-19   496.2       482.9       363    ₹-4,834       -2.7%       -0.3%     
  HDFCBANK    23     PVT BNK    01-Jan-19   496.2       482.9       363    ₹-4,834       -2.7%       -0.3%     
  BAJFINANCE  13     FIN SVC    01-Aug-18   264.0       254.9       725    ₹-6,636       -3.5%       +2.2%     
  MUTHOOTFIN  27     FIN SVC    01-Jan-19   448.5       431.0       401    ₹-7,001       -3.9%       -4.3%     
  LT          63     INFRA      01-Jan-19   1,283.0     1,178.6     140    ₹-14,612      -8.1% ⚠     -0.3%     
  BAJAJFINSV  24     FIN SVC    01-Aug-18   696.2       607.3       275    ₹-24,465      -12.8%      -2.9%     
  GODREJCP    53     FMCG       03-Sep-18   894.1       668.7       220    ₹-49,588      -25.2%      -6.8%     
  ⚠  WAZ < 0 (momentum below universe mean): LT

  AFTER: Invested ₹3,490,885 | Cash ₹144,604 | Total ₹3,635,489 | Positions 18/20 | Slot ₹181,774

========================================================================
  REBALANCE #51  —  01 Mar 2019
  NAV: ₹3,618,236  |  Slot: ₹180,912  |  Cash: ₹144,604
========================================================================

  [REGIME OFF] Nifty 200 5,688.6 < SMA200 5,753.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HINDUNILVR  23     FMCG       01-Jun-17   940.4       1,532.6     167    ₹98,890       +63.0%      -1.6%     
  BRITANNIA   31     FMCG       01-Sep-17   1,877.6     2,733.4     90     ₹77,025       +45.6%      +0.8%     
  TECHM       5      IT         01-Jun-18   516.6       626.8       353    ₹38,887       +21.3%      +3.7%     
  COFORGE     4      IT         01-Jun-18   201.6       243.7       906    ₹38,166       +20.9%      +2.0%     
  BATAINDIA   1      CON DUR    03-Sep-18   993.8       1,195.4     198    ₹39,932       +20.3%      +2.5%     
  MUTHOOTFIN  7      FIN SVC    01-Jan-19   448.5       464.4       401    ₹6,386        +3.6%       +1.7%     
  JUBLFOOD    30     CONSUMP    01-Jun-18   245.3       250.9       744    ₹4,167        +2.3%       -0.7%     
  ASIANPAINT  20     CONSUMP    01-Jan-19   1,283.9     1,303.3     140    ₹2,726        +1.5%       -0.9%     
  HAVELLS     15     CON DUR    03-Sep-18   680.5       672.1       290    ₹-2,438       -1.2%       +0.6%     
  KOTAKBANK   38     PVT BNK    01-Jan-19   248.9       244.0       723    ₹-3,526       -2.0%       -2.2%     
  BAJFINANCE  8      FIN SVC    01-Aug-18   264.0       258.0       725    ₹-4,362       -2.3%       +1.3%     
  HDFC        39     PVT BNK    01-Jan-19   496.2       481.2       363    ₹-5,429       -3.0%       -0.9%     
  HDFCBANK    40     PVT BNK    01-Jan-19   496.2       481.2       363    ₹-5,429       -3.0%       -0.9%     
  ABBOTINDIA  27     HEALTH     01-Aug-18   6,983.6     6,680.2     27     ₹-8,194       -4.3%       -2.8%     
  COLPAL      24     FMCG       01-Jan-19   1,115.6     1,049.9     161    ₹-10,575      -5.9%       -1.3%     
  BAJAJFINSV  19     FIN SVC    01-Aug-18   696.2       641.5       275    ₹-15,053      -7.9%       +3.1%     
  LT          83     INFRA      01-Jan-19   1,283.0     1,164.4     140    ₹-16,612      -9.2% ⚠     +1.9%     
  GODREJCP    81     FMCG       03-Sep-18   894.1       640.9       220    ₹-55,706      -28.3% ⚠    -0.9%     
  ⚠  WAZ < 0 (momentum below universe mean): GODREJCP, LT

  AFTER: Invested ₹3,473,632 | Cash ₹144,604 | Total ₹3,618,236 | Positions 18/20 | Slot ₹180,912

========================================================================
  REBALANCE #52  —  01 Apr 2019
  NAV: ₹3,818,252  |  Slot: ₹190,913  |  Cash: ₹144,604
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HINDUNILVR  57     FMCG       01-Jun-17   940.4       1,493.2     167    ₹92,302       +58.8%    669d  
  BRITANNIA   49     FMCG       01-Sep-17   1,877.6     2,709.0     90     ₹74,835       +44.3%    577d  
  LT          73     INFRA      01-Jan-19   1,283.0     1,256.6     140    ₹-3,700       -2.1%     90d   
  ABBOTINDIA  42     HEALTH     01-Aug-18   6,983.6     6,646.9     27     ₹-9,093       -4.8%     243d  
  COLPAL      55     FMCG       01-Jan-19   1,115.6     1,060.2     161    ₹-8,913       -5.0%     90d   
  GODREJCP    112    FMCG       03-Sep-18   894.1       638.9       220    ₹-56,152      -28.5%    210d  

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RELIANCE    2      OIL&GAS    3.044    0.35   +56.2%    +24.3%    616.1       309    ₹190,370      +4.9%     
  TITAN       3      CON DUR    2.559    -0.06  +29.0%    +26.2%    1,094.0     174    ₹190,355      +2.9%     
  VBL         5      FMCG       2.483    0.29   +46.0%    +21.6%    52.0        3672   ₹190,879      +9.5%     
  RBLBANK     6      PVT BNK    2.475    0.45   +47.7%    +17.8%    659.7       289    ₹190,648      +6.6%     
  INFY        8      IT         2.362    0.08   +34.7%    +15.6%    618.5       308    ₹190,485      +3.3%     
  UPL         12     MFG        2.219    0.10   +32.3%    +24.6%    580.5       328    ₹190,405      +3.7%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   1      CON DUR    03-Sep-18   993.8       1,295.6     198    ₹59,773       +30.4%      +3.3%     
  COFORGE     23     IT         01-Jun-18   201.6       245.6       906    ₹39,926       +21.9%      +1.1%     
  MUTHOOTFIN  7      FIN SVC    01-Jan-19   448.5       537.4       401    ₹35,652       +19.8%      +5.2%     
  JUBLFOOD    21     CONSUMP    01-Jun-18   245.3       287.0       744    ₹31,003       +17.0%      +5.1%     
  TECHM       27     IT         01-Jun-18   516.6       591.9       353    ₹26,572       +14.6%      -0.5%     
  BAJFINANCE  4      FIN SVC    01-Aug-18   264.0       291.0       725    ₹19,589       +10.2%      +5.5%     
  ASIANPAINT  14     CONSUMP    01-Jan-19   1,283.9     1,397.3     140    ₹15,883       +8.8%       +3.0%     
  HAVELLS     11     CON DUR    03-Sep-18   680.5       736.8       290    ₹16,322       +8.3%       +4.1%     
  HDFC        9      PVT BNK    01-Jan-19   496.2       534.0       363    ₹13,726       +7.6%       +3.5%     
  HDFCBANK    10     PVT BNK    01-Jan-19   496.2       534.0       363    ₹13,726       +7.6%       +3.5%     
  KOTAKBANK   25     PVT BNK    01-Jan-19   248.9       266.6       723    ₹12,802       +7.1%       +2.6%     
  BAJAJFINSV  20     FIN SVC    01-Aug-18   696.2       713.0       275    ₹4,619        +2.4%       +5.0%     

  AFTER: Invested ₹3,656,979 | Cash ₹159,915 | Total ₹3,816,894 | Positions 18/20 | Slot ₹190,913

========================================================================
  REBALANCE #53  —  02 May 2019
  NAV: ₹3,843,505  |  Slot: ₹192,175  |  Cash: ₹159,915
========================================================================
  [SECTOR CAP≤4] dropped: HCLTECH

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COFORGE     38     IT         01-Jun-18   201.6       236.2       906    ₹31,360       +17.2%    335d  
  JUBLFOOD    34     CONSUMP    01-Jun-18   245.3       264.5       744    ₹14,277       +7.8%     335d  
  ASIANPAINT  31     CONSUMP    01-Jan-19   1,283.9     1,344.8     140    ₹8,525        +4.7%     121d  

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ICICIGI     2      FIN SVC    3.005    0.20   +43.7%    +32.7%    1,053.0     182    ₹191,653      +4.0%     
  TCS         3      IT         2.739    0.23   +40.9%    +18.1%    1,821.3     105    ₹191,241      +3.9%     
  SHREECEM    10     INFRA      2.317    0.48   +16.5%    +23.8%    19,375.8    9      ₹174,382      +3.5%     
  WIPRO       11     IT         2.305    0.16   +34.3%    +11.7%    134.8       1425   ₹192,094      +3.7%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   1      CON DUR    03-Sep-18   993.8       1,369.8     198    ₹74,455       +37.8%      +3.3%     
  TECHM       14     IT         01-Jun-18   516.6       630.7       353    ₹40,271       +22.1%      +3.8%     
  MUTHOOTFIN  20     FIN SVC    01-Jan-19   448.5       531.1       401    ₹33,121       +18.4%      +0.2%     
  BAJFINANCE  4      FIN SVC    01-Aug-18   264.0       303.7       725    ₹28,749       +15.0%      +3.4%     
  KOTAKBANK   22     PVT BNK    01-Jan-19   248.9       279.9       723    ₹22,437       +12.5%      +3.6%     
  HDFC        5      PVT BNK    01-Jan-19   496.2       544.2       363    ₹17,415       +9.7%       +3.3%     
  HDFCBANK    6      PVT BNK    01-Jan-19   496.2       544.2       363    ₹17,415       +9.7%       +3.3%     
  BAJAJFINSV  9      FIN SVC    01-Aug-18   696.2       754.9       275    ₹16,125       +8.4%       +1.9%     
  HAVELLS     16     CON DUR    03-Sep-18   680.5       732.4       290    ₹15,041       +7.6%       +1.2%     
  UPL         8      MFG        01-Apr-19   580.5       598.9       328    ₹6,032        +3.2%       +2.4%     
  TITAN       13     CON DUR    01-Apr-19   1,094.0     1,111.4     174    ₹3,024        +1.6%       +1.5%     
  RELIANCE    7      OIL&GAS    01-Apr-19   616.1       621.9       309    ₹1,805        +0.9%       +3.0%     
  VBL         17     FMCG       01-Apr-19   52.0        51.2        3672   ₹-2,853       -1.5%       +4.0%     
  RBLBANK     15     PVT BNK    01-Apr-19   659.7       642.8       289    ₹-4,866       -2.6%       -0.5%     
  INFY        26     IT         01-Apr-19   618.5       598.6       308    ₹-6,130       -3.2%       -0.8%     

  AFTER: Invested ₹3,833,951 | Cash ₹8,665 | Total ₹3,842,615 | Positions 19/20 | Slot ₹192,175

========================================================================
  REBALANCE #54  —  03 Jun 2019
  NAV: ₹3,991,943  |  Slot: ₹199,597  |  Cash: ₹8,665
========================================================================
  [SECTOR CAP≤4] dropped: AXISBANK

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TECHM       94     IT         01-Jun-18   516.6       570.9       353    ₹19,151       +10.5%    367d  
  SHREECEM    —      INFRA      02-May-19   19,375.8    21,301.5    9      ₹17,331       +9.9%     32d   
  UPL         —      MFG        01-Apr-19   580.5       630.5       328    ₹16,385       +8.6%     63d   
  HAVELLS     28     CON DUR    03-Sep-18   680.5       731.4       290    ₹14,752       +7.5%     273d  
  WIPRO       26     IT         02-May-19   134.8       133.7       1425   ₹-1,537       -0.8%     32d   
  INFY        —      IT         01-Apr-19   618.5       609.9       308    ₹-2,636       -1.4%     63d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  NAUKRI      2      IT         2.909    0.05   +97.5%    +39.0%    451.4       442    ₹199,513      +15.9%    
  HONAUT      3      MFG        2.766    0.33   +44.5%    +22.9%    26,282.6    7      ₹183,978      +7.2%     
  GUJGASLTD   4      OIL&GAS    2.694    0.57   +10.2%    +55.1%    177.2       1126   ₹199,503      +9.8%     
  PIIND       10     MFG        2.361    0.29   +35.4%    +23.7%    1,116.0     178    ₹198,648      +4.8%     
  SRF         11     MFG        2.351    0.42   +51.7%    +27.8%    558.0       357    ₹199,189      +2.6%     
  ULTRACEMCO  15     INFRA      2.227    0.58   +27.1%    +28.1%    4,607.3     43     ₹198,113      +2.8%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  5      FIN SVC    01-Aug-18   264.0       340.4       725    ₹55,414       +29.0%      +6.1%     
  BATAINDIA   12     CON DUR    03-Sep-18   993.8       1,272.7     198    ₹55,233       +28.1%      +0.7%     
  MUTHOOTFIN  14     FIN SVC    01-Jan-19   448.5       573.4       401    ₹50,105       +27.9%      +4.1%     
  KOTAKBANK   17     PVT BNK    01-Jan-19   248.9       304.8       723    ₹40,413       +22.5%      +3.8%     
  BAJAJFINSV  6      FIN SVC    01-Aug-18   696.2       831.5       275    ₹37,195       +19.4%      +4.8%     
  HDFC        7      PVT BNK    01-Jan-19   496.2       567.6       363    ₹25,909       +14.4%      +3.3%     
  HDFCBANK    8      PVT BNK    01-Jan-19   496.2       567.6       363    ₹25,909       +14.4%      +3.3%     
  TITAN       13     CON DUR    01-Apr-19   1,094.0     1,236.5     174    ₹24,789       +13.0%      +5.1%     
  ICICIGI     1      FIN SVC    02-May-19   1,053.0     1,164.0     182    ₹20,186       +10.5%      +8.1%     
  VBL         22     FMCG       01-Apr-19   52.0        54.8        3672   ₹10,347       +5.4%       +3.0%     
  RBLBANK     19     PVT BNK    01-Apr-19   659.7       675.0       289    ₹4,431        +2.3%       +2.7%     
  TCS         25     IT         02-May-19   1,821.3     1,843.5     105    ₹2,322        +1.2%       +5.4%     
  RELIANCE    23     OIL&GAS    01-Apr-19   616.1       602.1       309    ₹-4,329       -2.3%       +2.7%     

  AFTER: Invested ₹3,971,692 | Cash ₹18,851 | Total ₹3,990,543 | Positions 19/20 | Slot ₹199,597

========================================================================
  REBALANCE #55  —  01 Jul 2019
  NAV: ₹3,961,721  |  Slot: ₹198,086  |  Cash: ₹18,851
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  RBLBANK     56     PVT BNK    01-Apr-19   659.7       631.9       289    ₹-8,020       -4.2%     91d   
  GUJGASLTD   38     OIL&GAS    03-Jun-19   177.2       162.9       1126   ₹-16,123      -8.1%     28d   
  RELIANCE    61     OIL&GAS    01-Apr-19   616.1       561.6       309    ₹-16,823      -8.8%     91d   

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ABBOTINDIA  6      HEALTH     2.592    0.26   +32.8%    +20.3%    8,050.8     24     ₹193,219      +4.4%     
  TRENT       9      CONSUMP    2.529    0.45   +41.2%    +21.2%    450.5       439    ₹197,790      +9.2%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   5      CON DUR    03-Sep-18   993.8       1,351.4     198    ₹70,822       +36.0%      +2.5%     
  BAJFINANCE  1      FIN SVC    01-Aug-18   264.0       358.7       725    ₹68,626       +35.9%      +4.3%     
  MUTHOOTFIN  17     FIN SVC    01-Jan-19   448.5       563.4       401    ₹46,096       +25.6%      +0.2%     
  BAJAJFINSV  7      FIN SVC    01-Aug-18   696.2       848.7       275    ₹41,932       +21.9%      +2.6%     
  KOTAKBANK   37     PVT BNK    01-Jan-19   248.9       294.8       723    ₹33,210       +18.5%      -0.3%     
  TITAN       4      CON DUR    01-Apr-19   1,094.0     1,292.1     174    ₹34,466       +18.1%      +2.8%     
  HDFC        12     PVT BNK    01-Jan-19   496.2       577.7       363    ₹29,590       +16.4%      +2.4%     
  HDFCBANK    13     PVT BNK    01-Jan-19   496.2       577.7       363    ₹29,590       +16.4%      +2.4%     
  SRF         3      MFG        03-Jun-19   558.0       589.6       357    ₹11,281       +5.7%       +2.4%     
  PIIND       2      MFG        03-Jun-19   1,116.0     1,170.1     178    ₹9,637        +4.9%       +3.5%     
  VBL         28     FMCG       01-Apr-19   52.0        54.1        3672   ₹7,930        +4.2%       +0.8%     
  TCS         16     IT         02-May-19   1,821.3     1,856.1     105    ₹3,649        +1.9%       +0.5%     
  ICICIGI     15     FIN SVC    02-May-19   1,053.0     1,055.7     182    ₹481          +0.3%       -2.8%     
  NAUKRI      8      IT         03-Jun-19   451.4       438.3       442    ₹-5,763       -2.9%       +3.6%     
  ULTRACEMCO  22     INFRA      03-Jun-19   4,607.3     4,379.3     43     ₹-9,803       -4.9%       -1.0%     
  HONAUT      11     MFG        03-Jun-19   26,282.6    24,614.7    7      ₹-11,675      -6.3%       +0.5%     

  AFTER: Invested ₹3,794,322 | Cash ₹166,934 | Total ₹3,961,257 | Positions 18/20 | Slot ₹198,086

========================================================================
  REBALANCE #56  —  01 Aug 2019
  NAV: ₹3,682,990  |  Slot: ₹184,149  |  Cash: ₹166,934
========================================================================
  [SECTOR CAP≤4] dropped: HDFCLIFE

  [REGIME OFF] Nifty 200 5,661.5 < SMA200 5,821.1 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   13     CON DUR    03-Sep-18   993.8       1,221.7     198    ₹45,121       +22.9%      -2.1%     
  MUTHOOTFIN  9      FIN SVC    01-Jan-19   448.5       544.8       401    ₹38,615       +21.5%      +0.0%     
  KOTAKBANK   26     PVT BNK    01-Jan-19   248.9       297.9       723    ₹35,427       +19.7%      -0.1%     
  BAJFINANCE  23     FIN SVC    01-Aug-18   264.0       313.0       725    ₹35,498       +18.5%      -3.3%     
  ICICIGI     8      FIN SVC    02-May-19   1,053.0     1,101.5     182    ₹8,820        +4.6%       +2.9%     
  VBL         16     FMCG       01-Apr-19   52.0        54.2        3672   ₹8,196        +4.3%       -0.8%     
  HDFC        67     PVT BNK    01-Jan-19   496.2       517.5       363    ₹7,755        +4.3% ⚠     -4.1%     
  HDFCBANK    68     PVT BNK    01-Jan-19   496.2       517.5       363    ₹7,755        +4.3% ⚠     -4.1%     
  BAJAJFINSV  55     FIN SVC    01-Aug-18   696.2       705.6       275    ₹2,579        +1.3%       -4.9%     
  TCS         30     IT         02-May-19   1,821.3     1,811.0     105    ₹-1,085       -0.6%       +1.5%     
  PIIND       4      MFG        03-Jun-19   1,116.0     1,093.1     178    ₹-4,082       -2.1%       -0.1%     
  NAUKRI      1      IT         03-Jun-19   451.4       433.1       442    ₹-8,082       -4.1%       +0.1%     
  ABBOTINDIA  7      HEALTH     01-Jul-19   8,050.8     7,695.4     24     ₹-8,528       -4.4%       -1.5%     
  TITAN       47     CON DUR    01-Apr-19   1,094.0     1,037.1     174    ₹-9,897       -5.2%       -5.1%     
  TRENT       10     CONSUMP    01-Jul-19   450.5       418.6       439    ₹-14,017      -7.1%       -0.0%     
  SRF         3      MFG        03-Jun-19   558.0       515.0       357    ₹-15,326      -7.7%       -3.8%     
  ULTRACEMCO  58     INFRA      03-Jun-19   4,607.3     4,104.5     43     ₹-21,622      -10.9%      -5.1%     
  HONAUT      34     MFG        03-Jun-19   26,282.6    22,664.6    7      ₹-25,326      -13.8%      -0.7%     
  ⚠  WAZ < 0 (momentum below universe mean): HDFC, HDFCBANK

  AFTER: Invested ₹3,516,056 | Cash ₹166,934 | Total ₹3,682,990 | Positions 18/20 | Slot ₹184,149

========================================================================
  REBALANCE #57  —  03 Sep 2019
  NAV: ₹3,760,195  |  Slot: ₹188,010  |  Cash: ₹166,934
========================================================================

  [REGIME OFF] Nifty 200 5,576.0 < SMA200 5,840.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   1      CON DUR    03-Sep-18   993.8       1,444.0     198    ₹89,149       +45.3%      +5.2%     
  MUTHOOTFIN  15     FIN SVC    01-Jan-19   448.5       541.4       401    ₹37,255       +20.7%      -1.7%     
  BAJFINANCE  36     FIN SVC    01-Aug-18   264.0       316.5       725    ₹38,061       +19.9%      -1.0%     
  KOTAKBANK   44     PVT BNK    01-Jan-19   248.9       281.7       723    ₹23,691       +13.2%      -4.4%     
  ICICIGI     11     FIN SVC    02-May-19   1,053.0     1,138.3     182    ₹15,523       +8.1%       +0.6%     
  ABBOTINDIA  2      HEALTH     01-Jul-19   8,050.8     8,666.7     24     ₹14,783       +7.7%       +5.2%     
  VBL         22     FMCG       01-Apr-19   52.0        55.0        3672   ₹11,028       +5.8%       -0.9%     
  HDFC        56     PVT BNK    01-Jan-19   496.2       515.0       363    ₹6,837        +3.8%       -1.1%     
  HDFCBANK    57     PVT BNK    01-Jan-19   496.2       515.0       363    ₹6,837        +3.8%       -1.1%     
  TCS         21     IT         02-May-19   1,821.3     1,870.4     105    ₹5,156        +2.7%       +1.5%     
  PIIND       4      MFG        03-Jun-19   1,116.0     1,133.0     178    ₹3,028        +1.5%       +2.7%     
  BAJAJFINSV  68     FIN SVC    01-Aug-18   696.2       700.1       275    ₹1,053        +0.5% ⚠     -1.8%     
  TRENT       8      CONSUMP    01-Jul-19   450.5       448.1       439    ₹-1,058       -0.5%       -1.6%     
  SRF         20     MFG        03-Jun-19   558.0       532.9       357    ₹-8,940       -4.5%       -2.7%     
  TITAN       55     CON DUR    01-Apr-19   1,094.0     1,039.4     174    ₹-9,496       -5.0%       -2.6%     
  HONAUT      35     MFG        03-Jun-19   26,282.6    24,236.5    7      ₹-14,323      -7.8%       +3.6%     
  NAUKRI      19     IT         03-Jun-19   451.4       398.2       442    ₹-23,518      -11.8%      -2.2%     
  ULTRACEMCO  85     INFRA      03-Jun-19   4,607.3     3,768.7     43     ₹-36,059      -18.2% ⚠    -5.7%     
  ⚠  WAZ < 0 (momentum below universe mean): BAJAJFINSV, ULTRACEMCO

  AFTER: Invested ₹3,593,261 | Cash ₹166,934 | Total ₹3,760,195 | Positions 18/20 | Slot ₹188,010

========================================================================
  REBALANCE #58  —  01 Oct 2019
  NAV: ₹4,107,816  |  Slot: ₹205,391  |  Cash: ₹166,934
========================================================================
  [SECTOR CAP≤4] dropped: PGHH

  [REGIME OFF] Nifty 200 5,850.4 < SMA200 5,852.1 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   3      CON DUR    03-Sep-18   993.8       1,630.0     198    ₹125,976      +64.0%      +6.5%     
  BAJFINANCE  19     FIN SVC    01-Aug-18   264.0       387.8       725    ₹89,744       +46.9%      +8.0%     
  MUTHOOTFIN  22     FIN SVC    01-Jan-19   448.5       600.6       401    ₹60,987       +33.9%      +5.1%     
  KOTAKBANK   15     PVT BNK    01-Jan-19   248.9       328.3       723    ₹57,372       +31.9%      +6.1%     
  ABBOTINDIA  7      HEALTH     01-Jul-19   8,050.8     9,965.1     24     ₹45,944       +23.8%      +6.8%     
  BAJAJFINSV  36     FIN SVC    01-Aug-18   696.2       842.7       275    ₹40,290       +21.0%      +7.9%     
  HDFC        24     PVT BNK    01-Jan-19   496.2       581.8       363    ₹31,076       +17.3%      +5.4%     
  HDFCBANK    25     PVT BNK    01-Jan-19   496.2       581.8       363    ₹31,076       +17.3%      +5.4%     
  TITAN       21     CON DUR    01-Apr-19   1,094.0     1,256.2     174    ₹28,215       +14.8%      +6.3%     
  PIIND       9      MFG        03-Jun-19   1,116.0     1,235.2     178    ₹21,210       +10.7%      +0.5%     
  ICICIGI     26     FIN SVC    02-May-19   1,053.0     1,151.0     182    ₹17,830       +9.3%       +1.4%     
  HONAUT      16     MFG        03-Jun-19   26,282.6    27,751.2    7      ₹10,280       +5.6%       +4.4%     
  TRENT       23     CONSUMP    01-Jul-19   450.5       475.7       439    ₹11,049       +5.6%       -0.2%     
  VBL         57     FMCG       01-Apr-19   52.0        52.7        3672   ₹2,485        +1.3%       -3.6%     
  SRF         55     MFG        03-Jun-19   558.0       524.9       357    ₹-11,787      -5.9%       -3.4%     
  TCS         76     IT         02-May-19   1,821.3     1,711.2     105    ₹-11,561      -6.0% ⚠     -2.6%     
  NAUKRI      52     IT         03-Jun-19   451.4       403.6       442    ₹-21,141      -10.6%      +1.3%     
  ULTRACEMCO  72     INFRA      03-Jun-19   4,607.3     4,085.9     43     ₹-22,419      -11.3% ⚠    +0.9%     
  ⚠  WAZ < 0 (momentum below universe mean): ULTRACEMCO, TCS

  AFTER: Invested ₹3,940,882 | Cash ₹166,934 | Total ₹4,107,816 | Positions 18/20 | Slot ₹205,391

========================================================================
  REBALANCE #59  —  01 Nov 2019
  NAV: ₹4,295,038  |  Slot: ₹214,752  |  Cash: ₹166,934
========================================================================
  [SECTOR CAP≤4] dropped: HDFCAMC, NAM-INDIA, SBILIFE, HDFCLIFE

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  KOTAKBANK   55     PVT BNK    01-Jan-19   248.9       314.7       723    ₹47,537       +26.4%    304d  
  BAJAJFINSV  31     FIN SVC    01-Aug-18   696.2       835.4       275    ₹38,274       +20.0%    457d  
  HDFC        41     PVT BNK    01-Jan-19   496.2       577.7       363    ₹29,596       +16.4%    304d  
  HDFCBANK    42     PVT BNK    01-Jan-19   496.2       577.7       363    ₹29,596       +16.4%    304d  
  NAUKRI      34     IT         03-Jun-19   451.4       511.5       442    ₹26,589       +13.3%    151d  
  VBL         89     FMCG       01-Apr-19   52.0        54.1        3672   ₹7,662        +4.0%     214d  
  TCS         61     IT         02-May-19   1,821.3     1,869.4     105    ₹5,050        +2.6%     183d  
  SRF         40     MFG        03-Jun-19   558.0       567.9       357    ₹3,552        +1.8%     151d  
  ULTRACEMCO  96     INFRA      03-Jun-19   4,607.3     4,037.6     43     ₹-24,498      -12.4%    151d  

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MANAPPURAM  1      FIN SVC    3.304    0.18   +147.1%   +52.5%    152.5       1407   ₹214,613      +11.8%    
  BERGEPAINT  2      CON DUR    3.236    0.24   +74.7%    +55.1%    411.5       521    ₹214,381      +6.3%     
  NESTLEIND   6      FMCG       2.740    0.11   +58.0%    +31.0%    696.3       308    ₹214,468      +3.5%     
  WHIRLPOOL   7      CON DUR    2.714    0.23   +52.9%    +45.2%    2,199.4     97     ₹213,339      +6.4%     
  HINDUNILVR  9      FMCG       2.619    0.13   +42.1%    +26.6%    1,949.5     110    ₹214,445      +4.4%     
  COLPAL      11     FMCG       2.541    0.29   +44.3%    +32.2%    1,309.1     164    ₹214,694      +1.5%     
  IGL         14     OIL&GAS    2.306    0.15   +63.2%    +29.3%    175.0       1227   ₹214,725      +5.3%     
  DMART       18     FMCG       2.053    0.30   +50.3%    +38.0%    1,966.6     109    ₹214,359      +3.9%     
  MGL         20     OIL&GAS    1.977    0.05   +25.6%    +31.2%    854.5       251    ₹214,480      +5.2%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   4      CON DUR    03-Sep-18   993.8       1,634.0     198    ₹126,775      +64.4%      +0.1%     
  BAJFINANCE  16     FIN SVC    01-Aug-18   264.0       395.3       725    ₹95,198       +49.7%      +2.5%     
  MUTHOOTFIN  21     FIN SVC    01-Jan-19   448.5       633.0       401    ₹73,998       +41.1%      +4.7%     
  ABBOTINDIA  5      HEALTH     01-Jul-19   8,050.8     10,661.6    24     ₹62,661       +32.4%      +3.8%     
  ICICIGI     24     FIN SVC    02-May-19   1,053.0     1,288.4     182    ₹42,832       +22.3%      +4.6%     
  PIIND       12     MFG        03-Jun-19   1,116.0     1,362.2     178    ₹43,829       +22.1%      +2.8%     
  TRENT       13     CONSUMP    01-Jul-19   450.5       544.1       439    ₹41,067       +20.8%      +5.0%     
  TITAN       19     CON DUR    01-Apr-19   1,094.0     1,277.0     174    ₹31,851       +16.7%      +0.5%     
  HONAUT      17     MFG        03-Jun-19   26,282.6    28,037.0    7      ₹12,281       +6.7%       +0.7%     

  AFTER: Invested ₹4,183,654 | Cash ₹109,093 | Total ₹4,292,747 | Positions 18/20 | Slot ₹214,752

========================================================================
  REBALANCE #60  —  02 Dec 2019
  NAV: ₹4,201,628  |  Slot: ₹210,081  |  Cash: ₹109,093
========================================================================
  [SECTOR CAP≤4] dropped: NAM-INDIA, BAJAJFINSV

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MUTHOOTFIN  48     FIN SVC    01-Jan-19   448.5       604.4       401    ₹62,526       +34.8%    335d  
  TRENT       29     CONSUMP    01-Jul-19   450.5       523.2       439    ₹31,883       +16.1%    154d  
  TITAN       68     CON DUR    01-Apr-19   1,094.0     1,131.8     174    ₹6,573        +3.5%     245d  
  HONAUT      37     MFG        03-Jun-19   26,282.6    26,924.4    7      ₹4,493        +2.4%     182d  
  DMART       30     FMCG       01-Nov-19   1,966.6     1,846.7     109    ₹-13,069      -6.1%     31d   
  HINDUNILVR  34     FMCG       01-Nov-19   1,949.5     1,827.7     110    ₹-13,397      -6.2%     31d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HDFCAMC     1      FIN SVC    3.941    0.44   +146.4%   +41.1%    1,498.3     140    ₹209,762      -0.9%     
  LALPATHLAB  4      HEALTH     2.944    0.18   +93.3%    +36.6%    760.0       276    ₹209,750      +2.5%     
  GUJGASLTD   11     OIL&GAS    2.246    0.08   +74.3%    +21.1%    206.7       1016   ₹209,958      +8.8%     
  RELIANCE    12     OIL&GAS    2.215    0.16   +41.6%    +24.4%    706.5       297    ₹209,818      +4.8%     
  SIEMENS     13     ENERGY     2.192    0.19   +52.8%    +23.9%    841.1       249    ₹209,425      -3.9%     
  NAUKRI      15     IT         2.098    -0.01  +83.9%    +25.6%    492.0       426    ₹209,611      -1.4%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   21     CON DUR    03-Sep-18   993.8       1,526.7     198    ₹105,519      +53.6%      -2.5%     
  BAJFINANCE  20     FIN SVC    01-Aug-18   264.0       383.7       725    ₹86,790       +45.3%      -3.4%     
  ABBOTINDIA  2      HEALTH     01-Jul-19   8,050.8     11,580.2    24     ₹84,705       +43.8%      +2.8%     
  PIIND       3      MFG        03-Jun-19   1,116.0     1,478.3     178    ₹64,492       +32.5%      +5.3%     
  ICICIGI     23     FIN SVC    02-May-19   1,053.0     1,314.4     182    ₹47,571       +24.8%      +2.1%     
  IGL         7      OIL&GAS    01-Nov-19   175.0       185.5       1227   ₹12,858       +6.0%       +2.4%     
  MGL         25     OIL&GAS    01-Nov-19   854.5       877.1       251    ₹5,665        +2.6%       +1.9%     
  BERGEPAINT  6      CON DUR    01-Nov-19   411.5       400.4       521    ₹-5,776       -2.7%       +1.3%     
  NESTLEIND   14     FMCG       01-Nov-19   696.3       677.6       308    ₹-5,765       -2.7%       +0.9%     
  WHIRLPOOL   5      CON DUR    01-Nov-19   2,199.4     2,121.8     97     ₹-7,521       -3.5%       -0.8%     
  COLPAL      16     FMCG       01-Nov-19   1,309.1     1,248.1     164    ₹-9,997       -4.7%       -3.8%     
  MANAPPURAM  8      FIN SVC    01-Nov-19   152.5       139.6       1407   ₹-18,174      -8.5%       -1.9%     

  AFTER: Invested ₹4,091,084 | Cash ₹109,050 | Total ₹4,200,134 | Positions 18/20 | Slot ₹210,081

========================================================================
  REBALANCE #61  —  01 Jan 2020
  NAV: ₹4,325,085  |  Slot: ₹216,254  |  Cash: ₹109,050
========================================================================
  [SECTOR CAP≤4] dropped: SBILIFE

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BATAINDIA   29     CON DUR    03-Sep-18   993.8       1,637.8     198    ₹127,528      +64.8%    485d  
  BAJFINANCE  28     FIN SVC    01-Aug-18   264.0       411.0       725    ₹106,568      +55.7%    518d  
  SIEMENS     39     ENERGY     02-Dec-19   841.1       845.9       249    ₹1,208        +0.6%     30d   
  COLPAL      86     FMCG       01-Nov-19   1,309.1     1,256.4     164    ₹-8,643       -4.0%     61d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  NAM-INDIA   5      FIN SVC    2.594    0.26   +119.0%   +33.5%    292.5       739    ₹216,182      +1.4%     
  COROMANDEL  6      MFG        2.582    -0.01  +20.4%    +28.8%    492.0       439    ₹215,986      +3.3%     
  IPCALAB     7      HEALTH     2.550    0.09   +46.7%    +23.3%    556.7       388    ₹216,018      +0.8%     
  SRF         12     MFG        2.394    -0.09  +55.7%    +25.0%    673.0       321    ₹216,020      +3.6%     
  COFORGE     15     IT         2.159    0.22   +40.9%    +13.9%    292.5       739    ₹216,136      +1.6%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  2      HEALTH     01-Jul-19   8,050.8     12,056.1    24     ₹96,128       +49.8%      +1.3%     
  PIIND       13     MFG        03-Jun-19   1,116.0     1,424.0     178    ₹54,824       +27.6%      -0.7%     
  ICICIGI     16     FIN SVC    02-May-19   1,053.0     1,328.8     182    ₹50,180       +26.2%      +0.4%     
  GUJGASLTD   1      OIL&GAS    02-Dec-19   206.7       242.9       1016   ₹36,818       +17.5%      +12.3%    
  IGL         3      OIL&GAS    01-Nov-19   175.0       189.6       1227   ₹17,946       +8.4%       +0.7%     
  MGL         27     OIL&GAS    01-Nov-19   854.5       902.8       251    ₹12,126       +5.7%       +1.7%     
  WHIRLPOOL   10     CON DUR    01-Nov-19   2,199.4     2,293.0     97     ₹9,080        +4.3%       +1.8%     
  MANAPPURAM  4      FIN SVC    01-Nov-19   152.5       157.7       1407   ₹7,233        +3.4%       +3.6%     
  NAUKRI      14     IT         02-Dec-19   492.0       502.1       426    ₹4,290        +2.0%       +0.2%     
  BERGEPAINT  11     CON DUR    01-Nov-19   411.5       418.7       521    ₹3,773        +1.8%       +2.1%     
  NESTLEIND   26     FMCG       01-Nov-19   696.3       690.6       308    ₹-1,769       -0.8%       +1.9%     
  LALPATHLAB  25     HEALTH     02-Dec-19   760.0       731.3       276    ₹-7,924       -3.8%       +1.4%     
  RELIANCE    20     OIL&GAS    02-Dec-19   706.5       672.2       297    ₹-10,170      -4.8%       -2.4%     
  HDFCAMC     9      FIN SVC    02-Dec-19   1,498.3     1,399.2     140    ₹-13,880      -6.6%       -1.1%     

  AFTER: Invested ₹4,257,422 | Cash ₹66,380 | Total ₹4,323,802 | Positions 19/20 | Slot ₹216,254

========================================================================
  REBALANCE #62  —  03 Feb 2020
  NAV: ₹4,607,070  |  Slot: ₹230,353  |  Cash: ₹66,380
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ICICIGI     41     FIN SVC    02-May-19   1,053.0     1,247.4     182    ₹35,378       +18.5%    277d  
  NAM-INDIA   23     FIN SVC    01-Jan-20   292.5       285.9       739    ₹-4,899       -2.3%     33d   
  RELIANCE    75     OIL&GAS    02-Dec-19   706.5       617.0       297    ₹-26,583      -12.7%    63d   

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    5      CONSUMP    2.586    0.34   +66.3%    +25.1%    385.8       597    ₹230,323      +9.9%     
  VBL         12     FMCG       2.267    0.20   +50.1%    +26.0%    69.5        3312   ₹230,300      +2.8%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  17     HEALTH     01-Jul-19   8,050.8     11,694.8    24     ₹87,458       +45.3%      +0.5%     
  PIIND       7      MFG        03-Jun-19   1,116.0     1,520.1     178    ₹71,924       +36.2%      +3.5%     
  GUJGASLTD   1      OIL&GAS    02-Dec-19   206.7       269.8       1016   ₹64,140       +30.5%      +2.3%     
  IGL         3      OIL&GAS    01-Nov-19   175.0       227.3       1227   ₹64,125       +29.9%      +6.7%     
  COFORGE     8      IT         01-Jan-20   292.5       349.4       739    ₹42,071       +19.5%      +2.6%     
  COROMANDEL  2      MFG        01-Jan-20   492.0       585.9       439    ₹41,240       +19.1%      +6.6%     
  MGL         22     OIL&GAS    01-Nov-19   854.5       986.6       251    ₹33,159       +15.5%      +1.8%     
  BERGEPAINT  6      CON DUR    01-Nov-19   411.5       461.5       521    ₹26,055       +12.2%      +4.1%     
  NAUKRI      21     IT         02-Dec-19   492.0       550.2       426    ₹24,757       +11.8%      +5.0%     
  SRF         4      MFG        01-Jan-20   673.0       741.4       321    ₹21,972       +10.2%      +4.2%     
  NESTLEIND   14     FMCG       01-Nov-19   696.3       761.7       308    ₹20,134       +9.4%       +6.2%     
  LALPATHLAB  10     HEALTH     02-Dec-19   760.0       830.1       276    ₹19,361       +9.2%       +4.0%     
  MANAPPURAM  19     FIN SVC    01-Nov-19   152.5       163.8       1407   ₹15,806       +7.4%       +1.0%     
  WHIRLPOOL   20     CON DUR    01-Nov-19   2,199.4     2,361.3     97     ₹15,710       +7.4%       -0.1%     
  IPCALAB     9      HEALTH     01-Jan-20   556.7       583.3       388    ₹10,295       +4.8%       -1.9%     
  HDFCAMC     11     FIN SVC    02-Dec-19   1,498.3     1,354.2     140    ₹-20,181      -9.6%       -3.1%     

  AFTER: Invested ₹4,379,763 | Cash ₹226,760 | Total ₹4,606,523 | Positions 18/20 | Slot ₹230,353

========================================================================
  REBALANCE #63  —  02 Mar 2020
  NAV: ₹4,507,437  |  Slot: ₹225,372  |  Cash: ₹226,760
========================================================================
  [SECTOR CAP≤4] dropped: DIVISLAB

  [REGIME OFF] Nifty 200 5,775.2 < SMA200 6,030.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  2      HEALTH     01-Jul-19   8,050.8     14,123.8    24     ₹145,753      +75.4%      +2.8%     
  PIIND       19     MFG        03-Jun-19   1,116.0     1,496.3     178    ₹67,695       +34.1%      -0.7%     
  GUJGASLTD   3      OIL&GAS    02-Dec-19   206.7       265.9       1016   ₹60,168       +28.7%      -2.5%     
  IPCALAB     5      HEALTH     01-Jan-20   556.7       704.0       388    ₹57,137       +26.4%      +6.4%     
  COROMANDEL  7      MFG        01-Jan-20   492.0       572.6       439    ₹35,399       +16.4%      +0.1%     
  COFORGE     13     IT         01-Jan-20   292.5       333.6       739    ₹30,393       +14.1%      -2.5%     
  SRF         8      MFG        01-Jan-20   673.0       765.8       321    ₹29,802       +13.8%      -2.5%     
  IGL         25     OIL&GAS    01-Nov-19   175.0       192.8       1227   ₹21,803       +10.2%      -7.9%     
  BERGEPAINT  6      CON DUR    01-Nov-19   411.5       452.2       521    ₹21,216       +9.9%       -1.8%     
  NESTLEIND   10     FMCG       01-Nov-19   696.3       752.8       308    ₹17,385       +8.1%       -0.2%     
  NAUKRI      27     IT         02-Dec-19   492.0       526.2       426    ₹14,543       +6.9%       -2.2%     
  LALPATHLAB  28     HEALTH     02-Dec-19   760.0       784.5       276    ₹6,772        +3.2%       -0.0%     
  VBL         23     FMCG       03-Feb-20   69.5        70.6        3312   ₹3,643        +1.6%       +0.0%     
  MGL         52     OIL&GAS    01-Nov-19   854.5       847.2       251    ₹-1,828       -0.9%       -9.9%     
  WHIRLPOOL   21     CON DUR    01-Nov-19   2,199.4     2,175.0     97     ₹-2,359       -1.1%       -3.0%     
  MANAPPURAM  37     FIN SVC    01-Nov-19   152.5       143.2       1407   ₹-13,074      -6.1%       -6.3%     
  HDFCAMC     18     FIN SVC    02-Dec-19   1,498.3     1,343.6     140    ₹-21,654      -10.3%      -5.8%     
  JUBLFOOD    41     CONSUMP    03-Feb-20   385.8       329.1       597    ₹-33,852      -14.7%      -8.3%     

  AFTER: Invested ₹4,280,678 | Cash ₹226,760 | Total ₹4,507,437 | Positions 18/20 | Slot ₹225,372

========================================================================
  REBALANCE #64  —  01 Apr 2020
  NAV: ₹3,721,664  |  Slot: ₹186,083  |  Cash: ₹226,760
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, DIVISLAB

  [REGIME OFF] Nifty 200 4,275.7 < SMA200 5,900.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  1      HEALTH     01-Jul-19   8,050.8     14,325.7    24     ₹150,598      +77.9%      +6.4%     
  IPCALAB     2      HEALTH     01-Jan-20   556.7       686.1       388    ₹50,204       +23.2%      +4.0%     
  PIIND       21     MFG        03-Jun-19   1,116.0     1,175.0     178    ₹10,505       +5.3%       -3.7%     
  NESTLEIND   3      FMCG       01-Nov-19   696.3       731.5       308    ₹10,831       +5.1%       +3.6%     
  GUJGASLTD   9      OIL&GAS    02-Dec-19   206.7       216.2       1016   ₹9,689        +4.6%       -6.3%     
  COROMANDEL  11     MFG        01-Jan-20   492.0       497.7       439    ₹2,492        +1.2%       -3.2%     
  IGL         13     OIL&GAS    01-Nov-19   175.0       174.2       1227   ₹-1,040       -0.5%       +2.5%     
  BERGEPAINT  7      CON DUR    01-Nov-19   411.5       392.5       521    ₹-9,875       -4.6%       +0.5%     
  LALPATHLAB  12     HEALTH     02-Dec-19   760.0       664.5       276    ₹-26,340      -12.6%      -5.6%     
  MGL         44     OIL&GAS    01-Nov-19   854.5       696.6       251    ₹-39,646      -18.5%      -4.9%     
  NAUKRI      29     IT         02-Dec-19   492.0       394.6       426    ₹-41,494      -19.8%      -6.8%     
  WHIRLPOOL   30     CON DUR    01-Nov-19   2,199.4     1,742.9     97     ₹-44,274      -20.8%      -8.2%     
  SRF         28     MFG        01-Jan-20   673.0       520.3       321    ₹-48,992      -22.7%      -14.5%    
  COFORGE     34     IT         01-Jan-20   292.5       216.9       739    ₹-55,868      -25.8%      -10.1%    
  JUBLFOOD    27     CONSUMP    03-Feb-20   385.8       273.9       597    ₹-66,830      -29.0%      -5.2%     
  VBL         33     FMCG       03-Feb-20   69.5        46.9        3312   ₹-74,892      -32.5%      -15.7%    
  HDFCAMC     20     FIN SVC    02-Dec-19   1,498.3     962.5       140    ₹-75,014      -35.8%      -9.0%     
  MANAPPURAM  61     FIN SVC    01-Nov-19   152.5       83.7        1407   ₹-96,885      -45.1%      -16.9%    

  AFTER: Invested ₹3,494,905 | Cash ₹226,760 | Total ₹3,721,664 | Positions 18/20 | Slot ₹186,083

========================================================================
  REBALANCE #65  —  04 May 2020
  NAV: ₹4,250,965  |  Slot: ₹212,548  |  Cash: ₹226,760
========================================================================
  [SECTOR CAP≤4] dropped: ALKEM, DIVISLAB, CIPLA, BIOCON, ZYDUSLIFE

  [REGIME OFF] Nifty 200 4,810.0 < SMA200 5,776.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  1      HEALTH     01-Jul-19   8,050.8     16,694.5    24     ₹207,449      +107.4%     +8.5%     
  IPCALAB     2      HEALTH     01-Jan-20   556.7       792.6       388    ₹91,496       +42.4%      +4.5%     
  PIIND       10     MFG        03-Jun-19   1,116.0     1,489.1     178    ₹66,409       +33.4%      +5.5%     
  IGL         14     OIL&GAS    01-Nov-19   175.0       209.4       1227   ₹42,266       +19.7%      +5.4%     
  NESTLEIND   4      FMCG       01-Nov-19   696.3       815.7       308    ₹36,764       +17.1%      +2.5%     
  GUJGASLTD   17     OIL&GAS    02-Dec-19   206.7       239.5       1016   ₹33,330       +15.9%      -0.2%     
  COROMANDEL  18     MFG        01-Jan-20   492.0       529.7       439    ₹16,545       +7.7%       +2.7%     
  SRF         11     MFG        01-Jan-20   673.0       713.5       321    ₹13,014       +6.0%       +6.1%     
  NAUKRI      23     IT         02-Dec-19   492.0       490.0       426    ₹-858         -0.4%       +5.8%     
  LALPATHLAB  20     HEALTH     02-Dec-19   760.0       734.0       276    ₹-7,168       -3.4%       +1.3%     
  BERGEPAINT  19     CON DUR    01-Nov-19   411.5       390.6       521    ₹-10,869      -5.1%       -4.8%     
  MGL         50     OIL&GAS    01-Nov-19   854.5       804.9       251    ₹-12,452      -5.8%       +2.0%     
  WHIRLPOOL   31     CON DUR    01-Nov-19   2,199.4     1,930.8     97     ₹-26,048      -12.2%      +2.5%     
  JUBLFOOD    34     CONSUMP    03-Feb-20   385.8       306.5       597    ₹-47,350      -20.6%      +3.2%     
  VBL         45     FMCG       03-Feb-20   69.5        53.8        3312   ₹-51,976      -22.6%      -1.8%     
  COFORGE     61     IT         01-Jan-20   292.5       220.1       739    ₹-53,472      -24.7% ⚠    +1.7%     
  HDFCAMC     21     FIN SVC    02-Dec-19   1,498.3     1,125.0     140    ₹-52,263      -24.9%      +1.5%     
  MANAPPURAM  52     FIN SVC    01-Nov-19   152.5       108.2       1407   ₹-62,348      -29.1%      +5.5%     
  ⚠  WAZ < 0 (momentum below universe mean): COFORGE

  AFTER: Invested ₹4,024,206 | Cash ₹226,760 | Total ₹4,250,965 | Positions 18/20 | Slot ₹212,548

========================================================================
  REBALANCE #66  —  01 Jun 2020
  NAV: ₹4,347,350  |  Slot: ₹217,368  |  Cash: ₹226,760
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, DIVISLAB, SYNGENE, TORNTPHARM, AUROPHARMA, SUNPHARMA

  [REGIME OFF] Nifty 200 5,087.4 < SMA200 5,669.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  1      HEALTH     01-Jul-19   8,050.8     15,437.7    24     ₹177,286      +91.8%      -0.9%     
  PIIND       16     MFG        03-Jun-19   1,116.0     1,552.3     178    ₹77,661       +39.1%      +4.1%     
  IPCALAB     7      HEALTH     01-Jan-20   556.7       749.6       388    ₹74,812       +34.6%      -2.1%     
  COROMANDEL  6      MFG        01-Jan-20   492.0       605.6       439    ₹49,881       +23.1%      +4.7%     
  IGL         12     OIL&GAS    01-Nov-19   175.0       209.8       1227   ₹42,731       +19.9%      +2.0%     
  NESTLEIND   5      FMCG       01-Nov-19   696.3       802.9       308    ₹32,835       +15.3%      +2.0%     
  GUJGASLTD   26     OIL&GAS    02-Dec-19   206.7       233.9       1016   ₹27,710       +13.2%      +0.9%     
  SRF         35     MFG        01-Jan-20   673.0       726.6       321    ₹17,204       +8.0%       +6.3%     
  NAUKRI      —      IT         02-Dec-19   492.0       525.5       426    ₹14,234       +6.8%       +6.6%     
  BERGEPAINT  21     CON DUR    01-Nov-19   411.5       397.9       521    ₹-7,083       -3.3%       +4.6%     
  MGL         43     OIL&GAS    01-Nov-19   854.5       822.2       251    ₹-8,108       -3.8%       +4.7%     
  LALPATHLAB  23     HEALTH     02-Dec-19   760.0       728.6       276    ₹-8,654       -4.1%       -1.4%     
  COFORGE     46     IT         01-Jan-20   292.5       272.4       739    ₹-14,815      -6.9% ⚠     +4.4%     
  WHIRLPOOL   24     CON DUR    01-Nov-19   2,199.4     1,998.5     97     ₹-19,485      -9.1%       +6.4%     
  JUBLFOOD    —      CONSUMP    03-Feb-20   385.8       331.8       597    ₹-32,213      -14.0%      +4.1%     
  VBL         60     FMCG       03-Feb-20   69.5        54.4        3312   ₹-50,097      -21.8% ⚠    +1.3%     
  HDFCAMC     25     FIN SVC    02-Dec-19   1,498.3     1,168.9     140    ₹-46,122      -22.0%      +5.5%     
  MANAPPURAM  54     FIN SVC    01-Nov-19   152.5       117.8       1407   ₹-48,924      -22.8%      +9.3%     
  ⚠  WAZ < 0 (momentum below universe mean): COFORGE, VBL

  AFTER: Invested ₹4,120,591 | Cash ₹226,760 | Total ₹4,347,350 | Positions 18/20 | Slot ₹217,368

========================================================================
  REBALANCE #67  —  01 Jul 2020
  NAV: ₹4,502,266  |  Slot: ₹225,113  |  Cash: ₹226,760
========================================================================
  [SECTOR CAP≤4] dropped: CIPLA, AUROPHARMA, TORNTPHARM

  [REGIME OFF] Nifty 200 5,404.3 < SMA200 5,627.0 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  11     HEALTH     01-Jul-19   8,050.8     14,456.3    24     ₹153,732      +79.6%      -2.0%     
  GUJGASLTD   5      OIL&GAS    02-Dec-19   206.7       311.5       1016   ₹106,481      +50.7%      +11.1%    
  IPCALAB     8      HEALTH     01-Jan-20   556.7       809.5       388    ₹98,067       +45.4%      +1.6%     
  COROMANDEL  3      MFG        01-Jan-20   492.0       700.0       439    ₹91,305       +42.3%      +5.6%     
  PIIND       20     MFG        03-Jun-19   1,116.0     1,508.1     178    ₹69,795       +35.1%      -0.9%     
  NESTLEIND   23     FMCG       01-Nov-19   696.3       787.9       308    ₹28,195       +13.1%      +0.9%     
  IGL         31     OIL&GAS    01-Nov-19   175.0       195.6       1227   ₹25,305       +11.8%      -3.5%     
  NAUKRI      —      IT         02-Dec-19   492.0       540.6       426    ₹20,691       +9.9%       +1.2%     
  MGL         42     OIL&GAS    01-Nov-19   854.5       897.6       251    ₹10,805       +5.0%       +1.3%     
  SRF         36     MFG        01-Jan-20   673.0       705.2       321    ₹10,334       +4.8%       -0.3%     
  LALPATHLAB  37     HEALTH     02-Dec-19   760.0       755.0       276    ₹-1,383       -0.7%       +0.4%     
  BERGEPAINT  28     CON DUR    01-Nov-19   411.5       401.3       521    ₹-5,285       -2.5%       -1.4%     
  MANAPPURAM  30     FIN SVC    01-Nov-19   152.5       141.8       1407   ₹-15,080      -7.0%       +7.5%     
  WHIRLPOOL   43     CON DUR    01-Nov-19   2,199.4     2,015.5     97     ₹-17,831      -8.4%       +0.6%     
  COFORGE     64     IT         01-Jan-20   292.5       260.0       739    ₹-23,973      -11.1% ⚠    -0.3%     
  JUBLFOOD    —      CONSUMP    03-Feb-20   385.8       336.0       597    ₹-29,713      -12.9%      -0.6%     
  VBL         55     FMCG       03-Feb-20   69.5        60.3        3312   ₹-30,564      -13.3% ⚠    +3.1%     
  HDFCAMC     44     FIN SVC    02-Dec-19   1,498.3     1,090.4     140    ₹-57,111      -27.2% ⚠    -1.4%     
  ⚠  WAZ < 0 (momentum below universe mean): HDFCAMC, VBL, COFORGE

  AFTER: Invested ₹4,275,507 | Cash ₹226,760 | Total ₹4,502,266 | Positions 18/20 | Slot ₹225,113

========================================================================
  REBALANCE #68  —  03 Aug 2020
  NAV: ₹4,741,641  |  Slot: ₹237,082  |  Cash: ₹226,760
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, TORNTPHARM, DIVISLAB, INFY, AJANTPHARM

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  NAUKRI      —      IT         02-Dec-19   492.0       618.9       426    ₹54,061       +25.8%    245d  
  SRF         48     MFG        01-Jan-20   673.0       751.3       321    ₹25,146       +11.6%    215d  
  NESTLEIND   51     FMCG       01-Nov-19   696.3       775.0       308    ₹24,236       +11.3%    276d  
  IGL         88     OIL&GAS    01-Nov-19   175.0       175.4       1227   ₹520          +0.2%     276d  
  MGL         56     OIL&GAS    01-Nov-19   854.5       831.1       251    ₹-5,876       -2.7%     276d  
  WHIRLPOOL   46     CON DUR    01-Nov-19   2,199.4     2,080.9     97     ₹-11,491      -5.4%     276d  
  MANAPPURAM  37     FIN SVC    01-Nov-19   152.5       142.6       1407   ₹-13,945      -6.5%     276d  
  JUBLFOOD    —      CONSUMP    03-Feb-20   385.8       344.3       597    ₹-24,802      -10.8%    182d  
  VBL         65     FMCG       03-Feb-20   69.5        59.7        3312   ₹-32,574      -14.1%    182d  
  HDFCAMC     77     FIN SVC    02-Dec-19   1,498.3     1,080.2     140    ₹-58,541      -27.9%    245d  

  ENTRIES (9)
  [52w filter blocked 1: ENDURANCE(-22.9%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JUBILANT    1      CONSUMP    3.755    0.79   +100.3%   +127.5%   618.2       383    ₹236,764      +16.9%    
  SYNGENE     3      HEALTH     2.811    0.44   +48.5%    +51.9%    473.6       500    ₹236,783      +7.9%     
  MUTHOOTFIN  4      FIN SVC    2.672    1.12   +115.6%   +64.0%    1,172.9     202    ₹236,927      +4.0%     
  MPHASIS     5      IT         2.547    0.54   +30.2%    +66.8%    1,024.9     231    ₹236,762      +9.8%     
  BALKRISIND  6      MFG        2.349    0.94   +81.9%    +47.5%    1,249.0     189    ₹236,052      +3.8%     
  WIPRO       10     IT         1.986    0.61   +6.9%     +52.9%    129.8       1826   ₹236,960      +7.9%     
  HCLTECH     11     IT         1.960    0.71   +41.1%    +36.1%    559.9       423    ₹236,826      +8.0%     
  BRITANNIA   19     FMCG       1.591    0.74   +40.7%    +26.8%    3,411.0     69     ₹235,361      +0.6%     
  ATGL        21     OIL&GAS    1.506    0.73   -3.8%     +60.8%    156.2       1517   ₹237,018      +3.5%     

  HOLDS (8)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  35     HEALTH     01-Jul-19   8,050.8     14,688.6    24     ₹159,307      +82.4%      +4.1%     
  IPCALAB     7      HEALTH     01-Jan-20   556.7       927.0       388    ₹143,645      +66.5%      +7.8%     
  PIIND       15     MFG        03-Jun-19   1,116.0     1,809.4     178    ₹123,424      +62.1%      +6.6%     
  COROMANDEL  2      MFG        01-Jan-20   492.0       737.5       439    ₹107,776      +49.9%      +1.3%     
  GUJGASLTD   9      OIL&GAS    02-Dec-19   206.7       291.7       1016   ₹86,425       +41.2%      +3.6%     
  COFORGE     16     IT         01-Jan-20   292.5       362.1       739    ₹51,467       +23.8%      +12.5%    
  LALPATHLAB  14     HEALTH     02-Dec-19   760.0       892.7       276    ₹36,624       +17.5%      +0.5%     
  BERGEPAINT  20     CON DUR    01-Nov-19   411.5       426.3       521    ₹7,743        +3.6%       +1.5%     

  AFTER: Invested ₹4,519,960 | Cash ₹219,153 | Total ₹4,739,112 | Positions 17/20 | Slot ₹237,082

========================================================================
  REBALANCE #69  —  01 Sep 2020
  NAV: ₹4,746,939  |  Slot: ₹237,347  |  Cash: ₹219,153
========================================================================
  [SECTOR CAP≤4] dropped: DIVISLAB, DRREDDY, TORNTPHARM, ALKEM

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COFORGE     32     IT         01-Jan-20   292.5       355.0       739    ₹46,243       +21.4%    244d  
  ATGL        30     OIL&GAS    03-Aug-20   156.2       176.9       1517   ₹31,350       +13.2%    29d   
  BERGEPAINT  29     CON DUR    01-Nov-19   411.5       449.2       521    ₹19,672       +9.2%     305d  
  WIPRO       27     IT         03-Aug-20   129.8       125.2       1826   ₹-8,374       -3.5%     29d   

  ENTRIES (5)
  [52w filter blocked 1: ADANIENSOL(-28.1%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ENDURANCE   10     AUTO       1.895    0.49   +25.7%    +35.1%    1,041.5     227    ₹236,425      +2.5%     
  HINDZINC    11     METAL      1.879    0.53   +23.1%    +32.6%    135.1       1756   ₹237,315      +0.5%     
  HUDCO       15     FIN SVC    1.803    0.95   +21.7%    +47.4%    25.7        9246   ₹237,326      -2.2%     
  ASIANPAINT  21     CONSUMP    1.605    0.67   +26.1%    +21.3%    1,875.4     126    ₹236,297      +5.0%     
  INFY        24     IT         1.502    0.79   +17.0%    +29.2%    779.3       304    ₹236,916      -2.5%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ABBOTINDIA  21     HEALTH     01-Jul-19   8,050.8     15,273.5    24     ₹173,345      +89.7%      +0.3%     
  IPCALAB     2      HEALTH     01-Jan-20   556.7       989.1       388    ₹167,749      +77.7%      +2.4%     
  PIIND       8      MFG        03-Jun-19   1,116.0     1,828.7     178    ₹126,863      +63.9%      -3.1%     
  GUJGASLTD   14     OIL&GAS    02-Dec-19   206.7       299.4       1016   ₹94,273       +44.9%      -0.4%     
  COROMANDEL  3      MFG        01-Jan-20   492.0       699.9       439    ₹91,275       +42.3%      -4.6%     
  LALPATHLAB  19     HEALTH     02-Dec-19   760.0       859.7       276    ₹27,537       +13.1%      -2.6%     
  BALKRISIND  9      MFG        03-Aug-20   1,249.0     1,276.1     189    ₹5,124        +2.2%       -0.8%     
  BRITANNIA   23     FMCG       03-Aug-20   3,411.0     3,480.5     69     ₹4,791        +2.0%       -0.1%     
  SYNGENE     4      HEALTH     03-Aug-20   473.6       467.8       500    ₹-2,896       -1.2%       -1.3%     
  MPHASIS     17     IT         03-Aug-20   1,024.9     1,009.9     231    ₹-3,485       -1.5%       -2.5%     
  HCLTECH     24     IT         03-Aug-20   559.9       546.9       423    ₹-5,485       -2.3%       -1.2%     
  JUBILANT    5      CONSUMP    03-Aug-20   618.2       563.4       383    ₹-20,963      -8.9%       -2.9%     
  MUTHOOTFIN  7      FIN SVC    03-Aug-20   1,172.9     1,060.1     202    ₹-22,780      -9.6%       -3.4%     

  AFTER: Invested ₹4,718,679 | Cash ₹26,854 | Total ₹4,745,533 | Positions 18/20 | Slot ₹237,347

========================================================================
  REBALANCE #70  —  01 Oct 2020
  NAV: ₹5,004,885  |  Slot: ₹250,244  |  Cash: ₹26,854
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, APOLLOHOSP, CIPLA, SANOFI, TORNTPHARM

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LALPATHLAB  28     HEALTH     02-Dec-19   760.0       891.4       276    ₹36,281       +17.3%    304d  
  BRITANNIA   37     FMCG       03-Aug-20   3,411.0     3,515.2     69     ₹7,186        +3.1%     59d   
  HUDCO       65     FIN SVC    01-Sep-20   25.7        25.3        9246   ₹-3,054       -1.3%     30d   
  HINDZINC    48     METAL      01-Sep-20   135.1       123.8       1756   ₹-19,889      -8.4%     30d   
  JUBILANT    —      OTHER      03-Aug-20   618.2       527.1       383    ₹-34,888      -14.7%    59d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIENT    2      METAL      3.375    1.08   +103.2%   +87.7%    307.3       814    ₹250,114      +8.0%     
  DIVISLAB    3      HEALTH     3.215    0.55   +84.3%    +41.8%    2,973.1     84     ₹249,738      -1.9%     
  NAVINFLUOR  5      MFG        2.919    0.71   +180.4%   +24.5%    2,095.0     119    ₹249,305      +3.6%     
  WIPRO       10     IT         2.346    0.66   +32.6%    +40.6%    144.3       1734   ₹250,197      +3.1%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  IPCALAB     2      HEALTH     01-Jan-20   556.7       1,106.3     388    ₹213,232      +98.7%      +7.1%     
  ABBOTINDIA  23     HEALTH     01-Jul-19   8,050.8     15,256.9    24     ₹172,946      +89.5%      +0.2%     
  PIIND       19     MFG        03-Jun-19   1,116.0     1,913.5     178    ₹141,948      +71.5%      +0.2%     
  COROMANDEL  14     MFG        01-Jan-20   492.0       749.7       439    ₹113,110      +52.4%      +0.9%     
  GUJGASLTD   26     OIL&GAS    02-Dec-19   206.7       297.7       1016   ₹92,529       +44.1%      +1.9%     
  MPHASIS     8      IT         03-Aug-20   1,024.9     1,215.0     231    ₹43,915       +18.5%      +4.9%     
  SYNGENE     7      HEALTH     03-Aug-20   473.6       554.2       500    ₹40,322       +17.0%      +3.6%     
  HCLTECH     6      IT         03-Aug-20   559.9       646.5       423    ₹36,653       +15.5%      +3.5%     
  BALKRISIND  17     MFG        03-Aug-20   1,249.0     1,397.6     189    ₹28,089       +11.9%      +6.1%     
  INFY        16     IT         01-Sep-20   779.3       867.6       304    ₹26,824       +11.3%      +3.0%     
  ENDURANCE   25     AUTO       01-Sep-20   1,041.5     1,111.9     227    ₹15,966       +6.8%       +3.6%     
  ASIANPAINT  24     CONSUMP    01-Sep-20   1,875.4     1,933.3     126    ₹7,300        +3.1%       +3.6%     
  MUTHOOTFIN  32     FIN SVC    03-Aug-20   1,172.9     1,055.2     202    ₹-23,768      -10.0%      +3.8%     

  AFTER: Invested ₹4,835,233 | Cash ₹168,465 | Total ₹5,003,698 | Positions 17/20 | Slot ₹250,244

========================================================================
  REBALANCE #71  —  02 Nov 2020
  NAV: ₹5,048,504  |  Slot: ₹252,425  |  Cash: ₹168,465
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, APOLLOHOSP

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ABBOTINDIA  50     HEALTH     01-Jul-19   8,050.8     14,444.1    24     ₹153,439      +79.4%    490d  
  COROMANDEL  44     MFG        01-Jan-20   492.0       678.1       439    ₹81,707       +37.8%    306d  

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ATGL        6      OIL&GAS    2.122    0.83   +49.8%    +37.8%    219.0       1152   ₹252,274      +10.5%    
  LALPATHLAB  7      HEALTH     2.097    0.53   +44.8%    +23.8%    1,090.4     231    ₹251,871      +4.8%     
  SRF         15     MFG        1.934    0.89   +57.9%    +15.7%    864.0       292    ₹252,286      +1.0%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  IPCALAB     3      HEALTH     01-Jan-20   556.7       1,130.2     388    ₹222,488      +103.0%     +4.7%     
  PIIND       5      MFG        03-Jun-19   1,116.0     2,192.7     178    ₹191,646      +96.5%      +5.7%     
  GUJGASLTD   37     OIL&GAS    02-Dec-19   206.7       284.2       1016   ₹78,759       +37.5%      +0.4%     
  MPHASIS     13     IT         03-Aug-20   1,024.9     1,216.6     231    ₹44,276       +18.7%      +0.8%     
  INFY        9      IT         01-Sep-20   779.3       924.0       304    ₹43,972       +18.6%      -0.7%     
  HCLTECH     12     IT         03-Aug-20   559.9       657.1       423    ₹41,109       +17.4%      -1.9%     
  ADANIENT    1      METAL      01-Oct-20   307.3       340.1       814    ₹26,744       +10.7%      +7.6%     
  SYNGENE     11     HEALTH     03-Aug-20   473.6       523.7       500    ₹25,074       +10.6%      -3.2%     
  ASIANPAINT  8      CONSUMP    01-Sep-20   1,875.4     2,061.9     126    ₹23,503       +9.9%       +2.8%     
  WIPRO       17     IT         01-Oct-20   144.3       154.3       1734   ₹17,383       +6.9%       -0.9%     
  BALKRISIND  29     MFG        03-Aug-20   1,249.0     1,276.8     189    ₹5,259        +2.2%       -2.3%     
  NAVINFLUOR  2      MFG        01-Oct-20   2,095.0     2,138.5     119    ₹5,175        +2.1%       +3.2%     
  ENDURANCE   34     AUTO       01-Sep-20   1,041.5     1,054.1     227    ₹2,861        +1.2%       +1.6%     
  DIVISLAB    4      HEALTH     01-Oct-20   2,973.1     2,957.3     84     ₹-1,324       -0.5%       -2.2%     
  MUTHOOTFIN  30     FIN SVC    03-Aug-20   1,172.9     1,132.3     202    ₹-8,203       -3.5%       +3.7%     

  AFTER: Invested ₹4,992,119 | Cash ₹55,487 | Total ₹5,047,606 | Positions 18/20 | Slot ₹252,425

========================================================================
  REBALANCE #72  —  01 Dec 2020
  NAV: ₹5,589,604  |  Slot: ₹279,480  |  Cash: ₹55,487
========================================================================
  [SECTOR CAP≤4] dropped: APOLLOHOSP, MRF, TATACHEM

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ENDURANCE   74     AUTO       01-Sep-20   1,041.5     1,119.3     227    ₹17,647       +7.5%     91d   
  MUTHOOTFIN  51     FIN SVC    03-Aug-20   1,172.9     1,055.1     202    ₹-23,787      -10.0%    120d  

  ENTRIES (1)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIENSOL  16     ENERGY     1.772    -0.17  +35.4%    +43.8%    379.0       737    ₹279,286      +8.4%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  PIIND       13     MFG        03-Jun-19   1,116.0     2,294.6     178    ₹209,794      +105.6%     +1.8%     
  IPCALAB     11     HEALTH     01-Jan-20   556.7       1,120.5     388    ₹218,723      +101.3%     +5.0%     
  ATGL        1      OIL&GAS    02-Nov-20   219.0       360.4       1152   ₹162,854      +64.6%      +22.2%    
  GUJGASLTD   24     OIL&GAS    02-Dec-19   206.7       326.4       1016   ₹121,673      +58.0%      +5.8%     
  ADANIENT    4      METAL      01-Oct-20   307.3       420.7       814    ₹92,327       +36.9%      +10.8%    
  BALKRISIND  12     MFG        03-Aug-20   1,249.0     1,584.9     189    ₹63,490       +26.9%      +5.6%     
  NAVINFLUOR  2      MFG        01-Oct-20   2,095.0     2,638.5     119    ₹64,679       +25.9%      +6.2%     
  INFY        8      IT         01-Sep-20   779.3       980.5       304    ₹61,143       +25.8%      +2.3%     
  HCLTECH     18     IT         03-Aug-20   559.9       666.4       423    ₹45,063       +19.0%      +0.6%     
  SYNGENE     6      HEALTH     03-Aug-20   473.6       563.1       500    ₹44,753       +18.9%      +1.1%     
  DIVISLAB    5      HEALTH     01-Oct-20   2,973.1     3,512.3     84     ₹45,292       +18.1%      +5.7%     
  SRF         15     MFG        02-Nov-20   864.0       1,000.5     292    ₹39,846       +15.8%      +2.4%     
  MPHASIS     22     IT         03-Aug-20   1,024.9     1,173.6     231    ₹34,348       +14.5%      -1.1%     
  ASIANPAINT  38     CONSUMP    01-Sep-20   1,875.4     2,116.2     126    ₹30,342       +12.8%      +2.3%     
  WIPRO       9      IT         01-Oct-20   144.3       162.6       1734   ₹31,809       +12.7%      +1.5%     
  LALPATHLAB  14     HEALTH     02-Nov-20   1,090.4     1,093.5     231    ₹729          +0.3%       +2.6%     

  AFTER: Invested ₹5,346,192 | Cash ₹243,081 | Total ₹5,589,272 | Positions 17/20 | Slot ₹279,480

========================================================================
  REBALANCE #73  —  01 Jan 2021
  NAV: ₹6,011,231  |  Slot: ₹300,562  |  Cash: ₹243,081
========================================================================
  [SECTOR CAP≤4] dropped: LTTS

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  IPCALAB     40     HEALTH     01-Jan-20   556.7       1,074.3     388    ₹200,798      +93.0%    366d  
  BALKRISIND  38     MFG        03-Aug-20   1,249.0     1,572.8     189    ₹61,200       +25.9%    151d  

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TATACHEM    2      MFG        3.085    0.07   +69.1%    +59.1%    440.7       682    ₹300,555      +2.3%     
  CROMPTON    10     CON DUR    1.951    -0.05  +59.3%    +31.9%    363.7       826    ₹300,388      +7.0%     
  BERGEPAINT  13     CON DUR    1.905    -0.02  +49.4%    +29.9%    623.1       482    ₹300,338      +7.3%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  PIIND       26     MFG        03-Jun-19   1,116.0     2,227.4     178    ₹197,835      +99.6%      +0.3%     
  GUJGASLTD   20     OIL&GAS    02-Dec-19   206.7       362.0       1016   ₹157,872      +75.2%      +4.4%     
  ATGL        1      OIL&GAS    02-Nov-20   219.0       376.2       1152   ₹181,083      +71.8%      +5.0%     
  ADANIENT    3      METAL      01-Oct-20   307.3       489.7       814    ₹148,535      +59.4%      +6.8%     
  ASIANPAINT  7      CONSUMP    01-Sep-20   1,875.4     2,633.8     126    ₹95,565       +40.4%      +7.3%     
  INFY        11     IT         01-Sep-20   779.3       1,086.1     304    ₹93,258       +39.4%      +4.7%     
  HCLTECH     18     IT         03-Aug-20   559.9       759.4       423    ₹84,401       +35.6%      +5.4%     
  MPHASIS     29     IT         03-Aug-20   1,024.9     1,370.0     231    ₹79,700       +33.7%      +5.4%     
  SYNGENE     12     HEALTH     03-Aug-20   473.6       626.1       500    ₹76,263       +32.2%      +4.5%     
  SRF         9      MFG        02-Nov-20   864.0       1,114.9     292    ₹73,265       +29.0%      +4.9%     
  DIVISLAB    4      HEALTH     01-Oct-20   2,973.1     3,733.7     84     ₹63,895       +25.6%      +3.5%     
  WIPRO       17     IT         01-Oct-20   144.3       178.9       1734   ₹59,982       +24.0%      +4.7%     
  NAVINFLUOR  5      MFG        01-Oct-20   2,095.0     2,594.4     119    ₹59,429       +23.8%      +2.6%     
  ADANIENSOL  8      ENERGY     01-Dec-20   379.0       434.2       737    ₹40,756       +14.6%      +3.4%     
  LALPATHLAB  14     HEALTH     02-Nov-20   1,090.4     1,155.2     231    ₹14,985       +5.9%       +6.7%     

  AFTER: Invested ₹5,955,363 | Cash ₹54,798 | Total ₹6,010,161 | Positions 18/20 | Slot ₹300,562

========================================================================
  REBALANCE #74  —  01 Feb 2021
  NAV: ₹5,912,667  |  Slot: ₹295,633  |  Cash: ₹54,798
========================================================================
  [SECTOR CAP≤4] dropped: LTTS, MRF

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PIIND       90     MFG        03-Jun-19   1,116.0     1,966.5     178    ₹151,393      +76.2%    609d  
  GUJGASLTD   50     OIL&GAS    02-Dec-19   206.7       344.2       1016   ₹139,724      +66.5%    427d  
  ASIANPAINT  45     CONSUMP    01-Sep-20   1,875.4     2,322.8     126    ₹56,377       +23.9%    153d  
  LALPATHLAB  71     HEALTH     02-Nov-20   1,090.4     1,102.0     231    ₹2,685        +1.1%     91d   
  BERGEPAINT  44     CON DUR    01-Jan-21   623.1       585.6       482    ₹-18,090      -6.0%     31d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HINDZINC    5      METAL      2.297    0.10   +75.0%    +42.0%    188.7       1566   ₹295,451      +5.7%     
  BAJAJ-AUTO  7      AUTO       2.091    -0.00  +40.5%    +42.5%    3,548.3     83     ₹294,508      +8.6%     
  LT          10     INFRA      1.926    0.15   +12.2%    +58.9%    1,361.6     217    ₹295,462      +7.5%     
  HONAUT      13     MFG        1.749    -0.11  +48.0%    +43.6%    40,417.2    7      ₹282,920      +5.0%     
  GRASIM      14     INFRA      1.729    0.03   +40.0%    +44.6%    1,093.3     270    ₹295,194      +9.9%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    02-Nov-20   219.0       388.5       1152   ₹195,229      +77.4%      +5.3%     
  ADANIENT    2      METAL      01-Oct-20   307.3       535.7       814    ₹185,912      +74.3%      +4.7%     
  INFY        12     IT         01-Sep-20   779.3       1,086.5     304    ₹93,376       +39.4%      -2.6%     
  WIPRO       8      IT         01-Oct-20   144.3       194.7       1734   ₹87,433       +34.9%      -0.9%     
  MPHASIS     27     IT         03-Aug-20   1,024.9     1,370.0     231    ₹79,710       +33.7%      -3.1%     
  HCLTECH     32     IT         03-Aug-20   559.9       744.7       423    ₹78,176       +33.0%      -3.3%     
  ADANIENSOL  6      ENERGY     01-Dec-20   379.0       481.5       737    ₹75,579       +27.1%      +6.6%     
  SRF         25     MFG        02-Nov-20   864.0       1,081.4     292    ₹63,480       +25.2%      -1.4%     
  SYNGENE     16     HEALTH     03-Aug-20   473.6       556.1       500    ₹41,263       +17.4%      -5.5%     
  DIVISLAB    11     HEALTH     01-Oct-20   2,973.1     3,359.8     84     ₹32,487       +13.0%      -3.7%     
  CROMPTON    4      CON DUR    01-Jan-21   363.7       400.6       826    ₹30,512       +10.2%      +3.1%     
  NAVINFLUOR  26     MFG        01-Oct-20   2,095.0     2,307.2     119    ₹25,251       +10.1%      -6.4%     
  TATACHEM    3      MFG        01-Jan-21   440.7       453.6       682    ₹8,830        +2.9%       -1.5%     

  AFTER: Invested ₹5,792,202 | Cash ₹118,726 | Total ₹5,910,929 | Positions 18/20 | Slot ₹295,633

========================================================================
  REBALANCE #75  —  01 Mar 2021
  NAV: ₹6,772,536  |  Slot: ₹338,627  |  Cash: ₹118,726
========================================================================
  [SECTOR CAP≤4] dropped: LTTS

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  NAVINFLUOR  40     MFG        01-Oct-20   2,095.0     2,613.9     119    ₹61,747       +24.8%    151d  
  SRF         59     MFG        02-Nov-20   864.0       1,067.8     292    ₹59,501       +23.6%    119d  
  SYNGENE     31     HEALTH     03-Aug-20   473.6       552.6       500    ₹39,530       +16.7%    210d  
  DIVISLAB    41     HEALTH     01-Oct-20   2,973.1     3,357.8     84     ₹32,320       +12.9%    151d  
  BAJAJ-AUTO  33     AUTO       01-Feb-21   3,548.3     3,289.0     83     ₹-21,519      -7.3%     28d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  GUJGASLTD   5      OIL&GAS    2.592    0.03   +88.2%    +52.8%    510.5       663    ₹338,488      +16.0%    
  HUDCO       7      FIN SVC    2.132    0.09   +87.2%    +52.3%    41.6        8143   ₹338,625      +17.3%    
  IDFCFIRSTB  9      PVT BNK    2.054    0.04   +62.9%    +72.7%    63.2        5354   ₹338,603      +11.3%    
  APOLLOHOSP  12     HEALTH     1.703    -0.02  +74.6%    +31.2%    3,041.4     111    ₹337,594      +5.0%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIENT    1      METAL      01-Oct-20   307.3       849.1       814    ₹441,018      +176.3%     +15.5%    
  ATGL        4      OIL&GAS    02-Nov-20   219.0       524.6       1152   ₹352,044      +139.5%     +14.2%    
  ADANIENSOL  3      ENERGY     01-Dec-20   379.0       758.0       737    ₹279,397      +100.0%     +12.7%    
  TATACHEM    2      MFG        01-Jan-21   440.7       682.6       682    ₹165,009      +54.9%      +19.7%    
  MPHASIS     10     IT         03-Aug-20   1,024.9     1,467.9     231    ₹102,329      +43.2%      -0.7%     
  INFY        22     IT         01-Sep-20   779.3       1,091.7     304    ₹94,974       +40.1%      -1.1%     
  HCLTECH     24     IT         03-Aug-20   559.9       746.3       423    ₹78,855       +33.3%      -0.9%     
  WIPRO       14     IT         01-Oct-20   144.3       191.4       1734   ₹81,746       +32.7%      -2.6%     
  GRASIM      8      INFRA      01-Feb-21   1,093.3     1,229.6     270    ₹36,793       +12.5%      +5.1%     
  HONAUT      20     MFG        01-Feb-21   40,417.2    44,964.1    7      ₹31,829       +11.3%      +4.4%     
  HINDZINC    6      METAL      01-Feb-21   188.7       196.1       1566   ₹11,665       +3.9%       +1.0%     
  CROMPTON    23     CON DUR    01-Jan-21   363.7       371.8       826    ₹6,714        +2.2%       -1.9%     
  LT          27     INFRA      01-Feb-21   1,361.6     1,384.1     217    ₹4,894        +1.7%       -0.8%     

  AFTER: Invested ₹6,552,921 | Cash ₹218,008 | Total ₹6,770,929 | Positions 17/20 | Slot ₹338,627

========================================================================
  REBALANCE #76  —  01 Apr 2021
  NAV: ₹7,811,570  |  Slot: ₹390,578  |  Cash: ₹218,008
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HCLTECH     32     IT         03-Aug-20   559.9       804.2       423    ₹103,370      +43.6%    241d  
  WIPRO       33     IT         01-Oct-20   144.3       192.4       1734   ₹83,348       +33.3%    182d  
  CROMPTON    44     CON DUR    01-Jan-21   363.7       378.3       826    ₹12,078       +4.0%     90d   
  LT          38     INFRA      01-Feb-21   1,361.6     1,357.6     217    ₹-867         -0.3%     59d   
  HUDCO       —      OTHER      01-Mar-21   41.6        35.5        8143   ₹-49,868      -14.7%    31d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  DEEPAKNTR   5      MFG        2.783    0.00   +338.0%   +77.5%    1,618.8     241    ₹390,129      +7.1%     
  TATAELXSI   7      IT         2.572    0.02   +341.6%   +49.7%    2,599.1     150    ₹389,863      +2.7%     
  DIXON       8      CON DUR    2.507    0.03   +429.0%   +34.2%    3,580.2     109    ₹390,247      -5.8%     
  LAURUSLABS  9      HEALTH     1.956    0.01   +462.4%   +4.5%     359.2       1087   ₹390,466      +2.0%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    02-Nov-20   219.0       1,058.9     1152   ₹967,576      +383.5%     +31.6%    
  ADANIENT    2      METAL      01-Oct-20   307.3       1,104.0     814    ₹648,561      +259.3%     +16.5%    
  ADANIENSOL  3      ENERGY     01-Dec-20   379.0       999.2       737    ₹457,124      +163.7%     +21.3%    
  TATACHEM    4      MFG        01-Jan-21   440.7       715.4       682    ₹187,334      +62.3%      +5.5%     
  INFY        25     IT         01-Sep-20   779.3       1,193.6     304    ₹125,936      +53.2%      +2.5%     
  MPHASIS     27     IT         03-Aug-20   1,024.9     1,565.1     231    ₹124,773      +52.7%      +3.6%     
  GRASIM      6      INFRA      01-Feb-21   1,093.3     1,412.7     270    ₹86,239       +29.2%      +5.6%     
  HONAUT      24     MFG        01-Feb-21   40,417.2    45,601.3    7      ₹36,289       +12.8%      +0.5%     
  GUJGASLTD   10     OIL&GAS    01-Mar-21   510.5       524.7       663    ₹9,366        +2.8%       +5.5%     
  HINDZINC    28     METAL      01-Feb-21   188.7       183.1       1566   ₹-8,697       -2.9%       -1.4%     
  APOLLOHOSP  19     HEALTH     01-Mar-21   3,041.4     2,856.7     111    ₹-20,500      -6.1%       -1.0%     
  IDFCFIRSTB  11     PVT BNK    01-Mar-21   63.2        56.9        5354   ₹-34,154      -10.1%      -4.6%     

  AFTER: Invested ₹7,584,708 | Cash ₹225,008 | Total ₹7,809,717 | Positions 16/20 | Slot ₹390,578

========================================================================
  REBALANCE #77  —  03 May 2021
  NAV: ₹8,538,271  |  Slot: ₹426,914  |  Cash: ₹225,008
========================================================================

  EXITS (1)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HONAUT      43     MFG        01-Feb-21   40,417.2    42,567.1    7      ₹15,050       +5.3%     91d   

  ENTRIES (1)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JSWENERGY   10     ENERGY     1.781    0.27   +175.7%   +49.0%    106.9       3995   ₹426,899      +10.2%    

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    02-Nov-20   219.0       1,235.2     1152   ₹1,170,625    +464.0%     +13.0%    
  ADANIENT    2      METAL      01-Oct-20   307.3       1,251.9     814    ₹768,971      +307.4%     +10.2%    
  ADANIENSOL  3      ENERGY     01-Dec-20   379.0       1,066.0     737    ₹506,356      +181.3%     +6.3%     
  TATACHEM    7      MFG        01-Jan-21   440.7       719.1       682    ₹189,870      +63.2%      +3.2%     
  MPHASIS     20     IT         03-Aug-20   1,024.9     1,599.1     231    ₹132,633      +56.0%      +3.6%     
  INFY        33     IT         01-Sep-20   779.3       1,165.0     304    ₹117,253      +49.5%      -0.8%     
  LAURUSLABS  8      HEALTH     01-Apr-21   359.2       469.3       1087   ₹119,663      +30.6%      +9.0%     
  TATAELXSI   6      IT         01-Apr-21   2,599.1     3,314.9     150    ₹107,374      +27.5%      +13.3%    
  GRASIM      9      INFRA      01-Feb-21   1,093.3     1,373.9     270    ₹75,762       +25.7%      +3.9%     
  DIXON       5      CON DUR    01-Apr-21   3,580.2     4,241.7     109    ₹72,099       +18.5%      +8.0%     
  DEEPAKNTR   4      MFG        01-Apr-21   1,618.8     1,871.8     241    ₹60,968       +15.6%      +12.8%    
  APOLLOHOSP  15     HEALTH     01-Mar-21   3,041.4     3,162.9     111    ₹13,493       +4.0%       +2.5%     
  HINDZINC    36     METAL      01-Feb-21   188.7       194.5       1566   ₹9,158        +3.1%       -0.4%     
  GUJGASLTD   11     OIL&GAS    01-Mar-21   510.5       509.9       663    ₹-445         -0.1%       -0.5%     
  IDFCFIRSTB  34     PVT BNK    01-Mar-21   63.2        53.8        5354   ₹-50,430      -14.9%      -0.7%     

  AFTER: Invested ₹8,442,191 | Cash ₹95,573 | Total ₹8,537,764 | Positions 16/20 | Slot ₹426,914

========================================================================
  REBALANCE #78  —  01 Jun 2021
  NAV: ₹9,367,348  |  Slot: ₹468,367  |  Cash: ₹95,573
========================================================================
  [SECTOR CAP≤4] dropped: POLYCAB

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIENT    —      METAL      01-Oct-20   307.3       1,412.2     814    ₹899,406      +359.6%   243d  
  TATACHEM    —      MFG        01-Jan-21   440.7       648.6       682    ₹141,808      +47.2%    151d  
  GRASIM      —      INFRA      01-Feb-21   1,093.3     1,403.1     270    ₹83,629       +28.3%    120d  
  JSWENERGY   —      ENERGY     03-May-21   106.9       121.8       3995   ₹59,662       +14.0%    29d   
  HINDZINC    58     METAL      01-Feb-21   188.7       214.0       1566   ₹39,700       +13.4%    120d  
  APOLLOHOSP  —      HEALTH     01-Mar-21   3,041.4     3,198.0     111    ₹17,383       +5.1%     92d   
  IDFCFIRSTB  —      PVT BNK    01-Mar-21   63.2        57.4        5354   ₹-31,486      -9.3%     92d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JSWSTEEL    3      METAL      3.491    0.45   +281.0%   +70.2%    657.4       712    ₹468,089      +0.4%     
  TATASTEEL   6      METAL      2.684    0.72   +282.6%   +51.3%    92.9        5043   ₹468,288      +0.6%     
  BSE         7      FIN SVC    2.506    0.40   +155.2%   +60.8%    97.3        4813   ₹468,351      +16.9%    
  SAIL        8      METAL      2.487    0.68   +308.4%   +69.7%    104.5       4480   ₹468,296      -0.8%     
  DALBHARAT   9      MFG        1.948    0.60   +227.2%   +25.2%    1,770.4     264    ₹467,388      +3.9%     
  ZYDUSLIFE   11     HEALTH     1.906    0.25   +79.8%    +42.9%    597.4       783    ₹467,802      +2.1%     
  BALKRISIND  12     MFG        1.867    0.48   +103.2%   +37.4%    2,089.8     224    ₹468,106      +5.8%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    02-Nov-20   219.0       1,438.2     1152   ₹1,404,555    +556.8%     +10.6%    
  ADANIENSOL  2      ENERGY     01-Dec-20   379.0       1,499.4     737    ₹825,808      +295.7%     +13.2%    
  MPHASIS     29     IT         03-Aug-20   1,024.9     1,746.1     231    ₹166,598      +70.4%      +5.5%     
  INFY        27     IT         01-Sep-20   779.3       1,208.2     304    ₹130,381      +55.0%      +2.4%     
  LAURUSLABS  4      HEALTH     01-Apr-21   359.2       524.5       1087   ₹179,649      +46.0%      +7.1%     
  TATAELXSI   5      IT         01-Apr-21   2,599.1     3,384.3     150    ₹117,775      +30.2%      +1.8%     
  DIXON       18     CON DUR    01-Apr-21   3,580.2     4,098.5     109    ₹56,486       +14.5%      +3.4%     
  DEEPAKNTR   10     MFG        01-Apr-21   1,618.8     1,730.5     241    ₹26,913       +6.9%       -0.6%     
  GUJGASLTD   37     OIL&GAS    01-Mar-21   510.5       517.6       663    ₹4,667        +1.4%       +3.1%     

  AFTER: Invested ₹9,093,582 | Cash ₹269,875 | Total ₹9,363,457 | Positions 16/20 | Slot ₹468,367

========================================================================
  REBALANCE #79  —  01 Jul 2021
  NAV: ₹9,024,886  |  Slot: ₹451,244  |  Cash: ₹269,875
========================================================================
  [SECTOR CAP≤4] dropped: SRF

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MPHASIS     25     IT         03-Aug-20   1,024.9     1,944.8     231    ₹212,478      +89.7%    332d  
  GUJGASLTD   28     OIL&GAS    01-Mar-21   510.5       646.5       663    ₹90,139       +26.6%    122d  

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  POLYCAB     5      MFG        2.890    0.42   +155.2%   +46.2%    1,949.7     231    ₹450,380      +6.5%     
  COFORGE     12     IT         2.112    0.52   +198.7%   +42.2%    790.4       570    ₹450,541      +6.3%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        10     OIL&GAS    02-Nov-20   219.0       967.5       1152   ₹862,266      +341.8%     -23.5%    
  ADANIENSOL  23     ENERGY     01-Dec-20   379.0       1,006.5     737    ₹462,504      +165.6%     -21.0%    
  LAURUSLABS  1      HEALTH     01-Apr-21   359.2       656.5       1087   ₹323,114      +82.8%      +7.4%     
  INFY        20     IT         01-Sep-20   779.3       1,359.1     304    ₹176,240      +74.4%      +4.0%     
  TATAELXSI   2      IT         01-Apr-21   2,599.1     4,030.3     150    ₹214,678      +55.1%      +11.3%    
  DIXON       9      CON DUR    01-Apr-21   3,580.2     4,401.9     109    ₹89,561       +22.9%      +1.2%     
  DEEPAKNTR   16     MFG        01-Apr-21   1,618.8     1,850.4     241    ₹55,816       +14.3%      +5.7%     
  TATASTEEL   4      METAL      01-Jun-21   92.9        100.3       5043   ₹37,725       +8.1%       +3.1%     
  BALKRISIND  17     MFG        01-Jun-21   2,089.8     2,190.2     224    ₹22,498       +4.8%       +2.2%     
  SAIL        6      METAL      01-Jun-21   104.5       109.5       4480   ₹22,217       +4.7%       -0.8%     
  DALBHARAT   18     MFG        01-Jun-21   1,770.4     1,843.5     264    ₹19,297       +4.1%       +3.4%     
  ZYDUSLIFE   8      HEALTH     01-Jun-21   597.4       616.5       783    ₹14,915       +3.2%       +1.1%     
  JSWSTEEL    3      METAL      01-Jun-21   657.4       644.2       712    ₹-9,402       -2.0%       -1.4%     
  BSE         7      FIN SVC    01-Jun-21   97.3        93.2        4813   ₹-19,785      -4.2%       -0.3%     

  AFTER: Invested ₹8,778,067 | Cash ₹245,750 | Total ₹9,023,817 | Positions 16/20 | Slot ₹451,244

========================================================================
  REBALANCE #80  —  02 Aug 2021
  NAV: ₹9,453,188  |  Slot: ₹472,659  |  Cash: ₹245,750
========================================================================
  [SECTOR CAP≤4] dropped: SRF, PIDILITIND

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIENSOL  53     ENERGY     01-Dec-20   379.0       908.8       737    ₹390,499      +139.8%   244d  
  DIXON       41     CON DUR    01-Apr-21   3,580.2     4,339.1     109    ₹82,720       +21.2%    123d  
  ZYDUSLIFE   79     HEALTH     01-Jun-21   597.4       574.2       783    ₹-18,215      -3.9%     62d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  GUJGASLTD   7      OIL&GAS    2.363    0.41   +167.8%   +41.7%    722.7       653    ₹471,946      +8.3%     
  MPHASIS     9      IT         2.104    0.36   +130.0%   +47.1%    2,352.2     200    ₹470,439      +8.5%     
  ABBOTINDIA  10     HEALTH     2.103    -0.01  +37.1%    +32.3%    18,684.8    25     ₹467,121      +11.5%    

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        28     OIL&GAS    02-Nov-20   219.0       887.1       1152   ₹769,608      +305.1%     -5.7%     
  INFY        16     IT         01-Sep-20   779.3       1,421.0     304    ₹195,079      +82.3%      +3.3%     
  LAURUSLABS  5      HEALTH     01-Apr-21   359.2       644.6       1087   ₹310,266      +79.5%      +1.3%     
  TATAELXSI   4      IT         01-Apr-21   2,599.1     4,010.4     150    ₹211,693      +54.3%      +0.6%     
  BSE         1      FIN SVC    01-Jun-21   97.3        133.3       4813   ₹173,089      +37.0%      +10.0%    
  TATASTEEL   2      METAL      01-Jun-21   92.9        121.6       5043   ₹144,925      +30.9%      +8.9%     
  DEEPAKNTR   24     MFG        01-Apr-21   1,618.8     2,032.0     241    ₹99,586       +25.5%      +6.9%     
  COFORGE     3      IT         01-Jul-21   790.4       969.2       570    ₹101,903      +22.6%      +10.9%    
  DALBHARAT   6      MFG        01-Jun-21   1,770.4     2,108.0     264    ₹89,124       +19.1%      +1.3%     
  BALKRISIND  12     MFG        01-Jun-21   2,089.8     2,406.3     224    ₹70,913       +15.1%      +7.1%     
  SAIL        19     METAL      01-Jun-21   104.5       120.3       4480   ₹70,708       +15.1%      +6.6%     
  JSWSTEEL    13     METAL      01-Jun-21   657.4       713.8       712    ₹40,103       +8.6%       +5.1%     
  POLYCAB     14     MFG        01-Jul-21   1,949.7     1,815.6     231    ₹-30,985      -6.9%       -0.8%     

  AFTER: Invested ₹9,024,604 | Cash ₹426,911 | Total ₹9,451,515 | Positions 16/20 | Slot ₹472,659

========================================================================
  REBALANCE #81  —  01 Sep 2021
  NAV: ₹10,215,231  |  Slot: ₹510,762  |  Cash: ₹426,911
========================================================================
  [SECTOR CAP≤4] dropped: TECHM

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BALKRISIND  61     MFG        01-Jun-21   2,089.8     2,246.9     224    ₹35,195       +7.5%     92d   
  SAIL        47     METAL      01-Jun-21   104.5       103.5       4480   ₹-4,443       -0.9%     92d   
  JSWSTEEL    50     METAL      01-Jun-21   657.4       646.8       712    ₹-7,572       -1.6%     92d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SRF         3      MFG        3.005    0.62   +135.4%   +52.3%    1,958.2     260    ₹509,132      +9.1%     
  ADANIENSOL  4      ENERGY     2.773    0.65   +496.9%   +6.5%     1,659.6     307    ₹509,497      +33.6%    
  BAJAJFINSV  6      FIN SVC    2.476    0.93   +157.1%   +42.0%    1,673.8     305    ₹510,521      +9.1%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    02-Nov-20   219.0       1,508.6     1152   ₹1,485,679    +588.9%     +29.6%    
  INFY        13     IT         01-Sep-20   779.3       1,461.3     304    ₹207,312      +87.5%      -0.8%     
  LAURUSLABS  18     HEALTH     01-Apr-21   359.2       652.7       1087   ₹319,045      +81.7%      -1.0%     
  TATAELXSI   2      IT         01-Apr-21   2,599.1     4,604.6     150    ₹300,827      +77.2%      +5.1%     
  DEEPAKNTR   9      MFG        01-Apr-21   1,618.8     2,271.7     241    ₹157,353      +40.3%      +8.0%     
  TATASTEEL   8      METAL      01-Jun-21   92.9        121.7       5043   ₹145,425      +31.1%      +0.4%     
  BSE         22     FIN SVC    01-Jun-21   97.3        126.6       4813   ₹140,813      +30.1%      +4.7%     
  COFORGE     11     IT         01-Jul-21   790.4       976.6       570    ₹106,148      +23.6%      +4.5%     
  DALBHARAT   14     MFG        01-Jun-21   1,770.4     2,088.5     264    ₹83,974       +18.0%      +4.9%     
  MPHASIS     7      IT         02-Aug-21   2,352.2     2,538.1     200    ₹37,188       +7.9%       +2.6%     
  POLYCAB     10     MFG        01-Jul-21   1,949.7     2,069.3     231    ₹27,636       +6.1%       +10.7%    
  ABBOTINDIA  21     HEALTH     02-Aug-21   18,684.8    18,947.9    25     ₹6,577        +1.4%       +6.3%     
  GUJGASLTD   25     OIL&GAS    02-Aug-21   722.7       674.6       653    ₹-31,428      -6.7%       -1.8%     

  AFTER: Invested ₹9,889,800 | Cash ₹323,615 | Total ₹10,213,416 | Positions 16/20 | Slot ₹510,762

========================================================================
  REBALANCE #82  —  01 Oct 2021
  NAV: ₹10,318,535  |  Slot: ₹515,927  |  Cash: ₹323,615
========================================================================
  [SECTOR CAP≤4] dropped: TECHM, HCLTECH

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LAURUSLABS  74     HEALTH     01-Apr-21   359.2       609.9       1087   ₹272,473      +69.8%    183d  
  ABBOTINDIA  —      OTHER      02-Aug-21   18,684.8    20,842.3    25     ₹53,937       +11.5%    60d   
  GUJGASLTD   83     OIL&GAS    02-Aug-21   722.7       589.5       653    ₹-87,016      -18.4%    60d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IRCTC       3      PSE        3.381    0.73   +181.5%   +86.7%    725.8       710    ₹515,301      +6.9%     
  BAJAJHLDNG  7      FIN SVC    2.404    0.47   +99.7%    +34.8%    4,433.3     116    ₹514,261      +4.4%     
  PRESTIGE    8      REALTY     2.353    0.77   +97.5%    +69.3%    477.2       1081   ₹515,860      +8.8%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        2      OIL&GAS    02-Nov-20   219.0       1,421.5     1152   ₹1,385,314    +549.1%     +3.2%     
  TATAELXSI   4      IT         01-Apr-21   2,599.1     5,491.9     150    ₹433,922      +111.3%     +6.9%     
  INFY        40     IT         01-Sep-20   779.3       1,450.3     304    ₹203,976      +86.1%      -2.0%     
  DEEPAKNTR   11     MFG        01-Apr-21   1,618.8     2,348.6     241    ₹175,883      +45.1%      +0.2%     
  BSE         15     FIN SVC    01-Jun-21   97.3        129.4       4813   ₹154,271      +32.9%      +1.3%     
  COFORGE     29     IT         01-Jul-21   790.4       997.8       570    ₹118,208      +26.2%      -0.5%     
  TATASTEEL   12     METAL      01-Jun-21   92.9        111.9       5043   ₹96,131       +20.5%      -3.1%     
  MPHASIS     9      IT         02-Aug-21   2,352.2     2,770.0     200    ₹83,568       +17.8%      -1.8%     
  POLYCAB     10     MFG        01-Jul-21   1,949.7     2,283.5     231    ₹77,108       +17.1%      +0.5%     
  DALBHARAT   21     MFG        01-Jun-21   1,770.4     2,064.4     264    ₹77,619       +16.6%      -1.7%     
  SRF         5      MFG        01-Sep-21   1,958.2     2,186.9     260    ₹59,466       +11.7%      +3.6%     
  BAJAJFINSV  6      FIN SVC    01-Sep-21   1,673.8     1,712.9     305    ₹11,907       +2.3%       -0.1%     
  ADANIENSOL  1      ENERGY     01-Sep-21   1,659.6     1,577.8     307    ₹-25,097      -4.9%       -1.1%     

  AFTER: Invested ₹9,971,417 | Cash ₹345,283 | Total ₹10,316,700 | Positions 16/20 | Slot ₹515,927

========================================================================
  REBALANCE #83  —  01 Nov 2021
  NAV: ₹10,515,264  |  Slot: ₹525,763  |  Cash: ₹345,283
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TATASTEEL   18     METAL      01-Jun-21   92.9        117.7       5043   ₹125,290      +26.8%    153d  
  COFORGE     58     IT         01-Jul-21   790.4       959.6       570    ₹96,441       +21.4%    123d  
  DALBHARAT   54     MFG        01-Jun-21   1,770.4     1,967.8     264    ₹52,101       +11.1%    153d  
  BAJAJFINSV  6      FIN SVC    01-Sep-21   1,673.8     1,751.6     305    ₹23,704       +4.6%     61d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ONGC        9      OIL&GAS    2.138    1.03   +132.0%   +34.9%    113.8       4618   ₹525,754      -0.4%     
  TECHM       10     IT         2.074    0.69   +98.2%    +25.8%    1,277.2     411    ₹524,931      +2.2%     
  DIXON       11     CON DUR    2.052    0.70   +178.7%   +23.3%    5,295.8     99     ₹524,285      +4.1%     
  OBEROIRLTY  13     REALTY     1.954    0.77   +113.6%   +43.4%    943.0       557    ₹525,240      +5.8%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        2      OIL&GAS    02-Nov-20   219.0       1,432.8     1152   ₹1,398,255    +554.3%     +0.6%     
  TATAELXSI   5      IT         01-Apr-21   2,599.1     5,673.7     150    ₹461,188      +118.3%     +0.8%     
  INFY        47     IT         01-Sep-20   779.3       1,493.8     304    ₹217,206      +91.7%      -0.4%     
  BSE         31     FIN SVC    01-Jun-21   97.3        143.5       4813   ₹222,382      +47.5%      +2.0%     
  DEEPAKNTR   16     MFG        01-Apr-21   1,618.8     2,270.2     241    ₹156,984      +40.2%      -7.8%     
  MPHASIS     7      IT         02-Aug-21   2,352.2     3,039.8     200    ₹137,517      +29.2%      +2.1%     
  POLYCAB     8      MFG        01-Jul-21   1,949.7     2,269.4     231    ₹73,860       +16.4%      -0.6%     
  IRCTC       3      PSE        01-Oct-21   725.8       818.3       710    ₹65,696       +12.7%      -2.9%     
  SRF         10     MFG        01-Sep-21   1,958.2     2,092.4     260    ₹34,896       +6.9%       -3.8%     
  ADANIENSOL  1      ENERGY     01-Sep-21   1,659.6     1,760.6     307    ₹30,992       +6.1%       +0.5%     
  BAJAJHLDNG  9      FIN SVC    01-Oct-21   4,433.3     4,457.9     116    ₹2,860        +0.6%       +0.6%     
  PRESTIGE    38     REALTY     01-Oct-21   477.2       432.3       1081   ₹-48,534      -9.4%       -1.2%     

  AFTER: Invested ₹10,075,916 | Cash ₹436,854 | Total ₹10,512,770 | Positions 16/20 | Slot ₹525,763

========================================================================
  REBALANCE #84  —  01 Dec 2021
  NAV: ₹10,708,665  |  Slot: ₹535,433  |  Cash: ₹436,854
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        4      OIL&GAS    02-Nov-20   219.0       1,623.7     1152   ₹1,618,253    +641.5%     +1.9%     
  TATAELXSI   1      IT         01-Apr-21   2,599.1     5,587.5     150    ₹448,260      +115.0%     -3.0%     
  INFY        35     IT         01-Sep-20   779.3       1,506.9     304    ₹221,172      +93.4%      -0.7%     
  BSE         5      FIN SVC    01-Jun-21   97.3        175.3       4813   ₹375,381      +80.1%      +9.5%     
  DEEPAKNTR   24     MFG        01-Apr-21   1,618.8     2,120.9     241    ₹121,003      +31.0%      -4.0%     
  MPHASIS     15     IT         02-Aug-21   2,352.2     2,756.6     200    ₹80,886       +17.2%      -6.0%     
  POLYCAB     9      MFG        01-Jul-21   1,949.7     2,268.6     231    ₹73,669       +16.4%      -1.5%     
  BAJAJHLDNG  8      FIN SVC    01-Oct-21   4,433.3     4,853.5     116    ₹48,746       +9.5%       +4.8%     
  ADANIENSOL  2      ENERGY     01-Sep-21   1,659.6     1,797.2     307    ₹42,228       +8.3%       -4.2%     
  IRCTC       6      PSE        01-Oct-21   725.8       776.7       710    ₹36,152       +7.0%       -4.6%     
  TECHM       12     IT         01-Nov-21   1,277.2     1,345.4     411    ₹28,020       +5.3%       +2.8%     
  SRF         30     MFG        01-Sep-21   1,958.2     1,988.5     260    ₹7,875        +1.5%       -5.0%     
  ONGC        11     OIL&GAS    01-Nov-21   113.8       109.7       4618   ₹-19,369      -3.7%       -3.8%     
  DIXON       10     CON DUR    01-Nov-21   5,295.8     5,067.9     99     ₹-22,563      -4.3%       -2.5%     
  PRESTIGE    23     REALTY     01-Oct-21   477.2       439.3       1081   ₹-40,973      -7.9%       -2.2%     
  OBEROIRLTY  31     REALTY     01-Nov-21   943.0       818.1       557    ₹-69,545      -13.2%      -6.9%     

  AFTER: Invested ₹10,271,811 | Cash ₹436,854 | Total ₹10,708,665 | Positions 16/20 | Slot ₹535,433

========================================================================
  REBALANCE #85  —  03 Jan 2022
  NAV: ₹11,515,061  |  Slot: ₹575,753  |  Cash: ₹436,854
========================================================================
  [SECTOR CAP≤4] dropped: WIPRO, COFORGE

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PRESTIGE    49     REALTY     01-Oct-21   477.2       468.8       1081   ₹-9,117       -1.8%     94d   
  OBEROIRLTY  57     REALTY     01-Nov-21   943.0       868.5       557    ₹-41,471      -7.9%     63d   

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ESCORTS     7      MFG        2.462    0.67   +54.0%    +29.2%    1,852.9     310    ₹574,397      +2.5%     
  LT          16     INFRA      1.998    1.07   +50.8%    +13.3%    1,827.5     315    ₹575,656      +2.8%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        2      OIL&GAS    02-Nov-20   219.0       1,741.2     1152   ₹1,753,645    +695.1%     +0.4%     
  TATAELXSI   6      IT         01-Apr-21   2,599.1     5,597.2     150    ₹449,713      +115.4%     +2.5%     
  INFY        7      IT         01-Sep-20   779.3       1,668.2     304    ₹270,203      +114.1%     +3.8%     
  BSE         1      FIN SVC    01-Jun-21   97.3        202.3       4813   ₹505,167      +107.9%     +0.4%     
  DEEPAKNTR   12     MFG        01-Apr-21   1,618.8     2,494.2     241    ₹210,973      +54.1%      +6.8%     
  MPHASIS     10     IT         02-Aug-21   2,352.2     3,132.9     200    ₹156,135      +33.2%      +5.2%     
  POLYCAB     15     MFG        01-Jul-21   1,949.7     2,393.4     231    ₹102,497      +22.8%      +2.6%     
  SRF         14     MFG        01-Sep-21   1,958.2     2,378.8     260    ₹109,344      +21.5%      +5.8%     
  TECHM       3      IT         01-Nov-21   1,277.2     1,512.5     411    ₹96,704       +18.4%      +4.9%     
  BAJAJHLDNG  9      FIN SVC    01-Oct-21   4,433.3     5,006.4     116    ₹66,484       +12.9%      +3.1%     
  IRCTC       13     PSE        01-Oct-21   725.8       808.9       710    ₹59,040       +11.5%      +0.6%     
  ADANIENSOL  5      ENERGY     01-Sep-21   1,659.6     1,731.1     307    ₹21,950       +4.3%       -2.8%     
  DIXON       18     CON DUR    01-Nov-21   5,295.8     5,506.4     99     ₹20,848       +4.0%       +1.1%     
  ONGC        41     OIL&GAS    01-Nov-21   113.8       110.3       4618   ₹-16,521      -3.1%       +1.0%     

  AFTER: Invested ₹11,237,748 | Cash ₹275,948 | Total ₹11,513,696 | Positions 16/20 | Slot ₹575,753

========================================================================
  REBALANCE #86  —  01 Feb 2022
  NAV: ₹11,771,823  |  Slot: ₹588,591  |  Cash: ₹275,948
========================================================================

  EXITS (1)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DIXON       74     CON DUR    01-Nov-21   5,295.8     4,429.2     99     ₹-85,795      -16.4%    92d   

  ENTRIES (1)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  POWERGRID   9      ENERGY     2.306    0.66   +60.2%    +17.4%    130.0       4526   ₹588,520      +1.4%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        2      OIL&GAS    02-Nov-20   219.0       1,861.9     1152   ₹1,892,660    +750.2%     +3.3%     
  TATAELXSI   3      IT         01-Apr-21   2,599.1     7,089.3     150    ₹673,538      +172.8%     +10.5%    
  BSE         1      FIN SVC    01-Jun-21   97.3        211.4       4813   ₹549,067      +117.2%     +0.2%     
  INFY        25     IT         01-Sep-20   779.3       1,557.1     304    ₹236,438      +99.8%      -1.4%     
  DEEPAKNTR   22     MFG        01-Apr-21   1,618.8     2,232.1     241    ₹147,814      +37.9%      -5.7%     
  POLYCAB     7      MFG        01-Jul-21   1,949.7     2,471.0     231    ₹120,411      +26.7%      +0.5%     
  SRF         5      MFG        01-Sep-21   1,958.2     2,413.0     260    ₹118,240      +23.2%      -0.3%     
  MPHASIS     29     IT         02-Aug-21   2,352.2     2,880.5     200    ₹105,664      +22.5%      +0.4%     
  ADANIENSOL  4      ENERGY     01-Sep-21   1,659.6     1,986.9     307    ₹100,481      +19.7%      +1.7%     
  ONGC        6      OIL&GAS    01-Nov-21   113.8       131.8       4618   ₹82,798       +15.7%      +5.2%     
  IRCTC       12     PSE        01-Oct-21   725.8       819.7       710    ₹66,681       +12.9%      +0.5%     
  BAJAJHLDNG  10     FIN SVC    01-Oct-21   4,433.3     4,889.1     116    ₹52,871       +10.3%      -0.4%     
  LT          13     INFRA      03-Jan-22   1,827.5     1,891.8     315    ₹20,253       +3.5%       +2.3%     
  TECHM       34     IT         01-Nov-21   1,277.2     1,276.0     411    ₹-488         -0.1%       -5.6%     
  ESCORTS     8      MFG        03-Jan-22   1,852.9     1,800.2     310    ₹-16,324      -2.8%       -0.5%     

  AFTER: Invested ₹11,645,905 | Cash ₹125,219 | Total ₹11,771,124 | Positions 16/20 | Slot ₹588,591

========================================================================
  REBALANCE #87  —  02 Mar 2022
  NAV: ₹11,189,563  |  Slot: ₹559,478  |  Cash: ₹125,219
========================================================================

  [REGIME OFF] Nifty 200 8,782.3 < SMA200 8,961.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        6      OIL&GAS    02-Nov-20   219.0       1,665.5     1152   ₹1,666,336    +660.5%     -1.1%     
  TATAELXSI   5      IT         01-Apr-21   2,599.1     6,234.6     150    ₹545,321      +139.9%     -3.4%     
  BSE         1      FIN SVC    01-Jun-21   97.3        213.2       4813   ₹557,903      +119.1%     -3.3%     
  INFY        27     IT         01-Sep-20   779.3       1,496.2     304    ₹217,940      +92.0%      -1.5%     
  ADANIENSOL  2      ENERGY     01-Sep-21   1,659.6     2,241.1     307    ₹178,505      +35.0%      +11.1%    
  MPHASIS     11     IT         02-Aug-21   2,352.2     2,883.0     200    ₹106,158      +22.6%      +2.8%     
  SRF         4      MFG        01-Sep-21   1,958.2     2,321.2     260    ₹94,374       +18.5%      -2.6%     
  DEEPAKNTR   47     MFG        01-Apr-21   1,618.8     1,914.1     241    ₹71,160       +18.2%      -7.8%     
  POLYCAB     14     MFG        01-Jul-21   1,949.7     2,277.7     231    ₹75,771       +16.8%      -2.1%     
  ONGC        8      OIL&GAS    01-Nov-21   113.8       126.9       4618   ₹60,429       +11.5%      -0.0%     
  BAJAJHLDNG  29     FIN SVC    01-Oct-21   4,433.3     4,774.2     116    ₹39,546       +7.7%       -2.2%     
  IRCTC       10     PSE        01-Oct-21   725.8       775.6       710    ₹35,348       +6.9%       -0.8%     
  POWERGRID   16     ENERGY     01-Feb-22   130.0       132.4       4526   ₹10,529       +1.8%       +4.7%     
  ESCORTS     23     MFG        03-Jan-22   1,852.9     1,820.4     310    ₹-10,072      -1.8%       +1.1%     
  LT          36     INFRA      03-Jan-22   1,827.5     1,696.0     315    ₹-41,419      -7.2%       -3.6%     
  TECHM       50     IT         01-Nov-21   1,277.2     1,181.6     411    ₹-39,287      -7.5%       -3.1%     

  AFTER: Invested ₹11,064,343 | Cash ₹125,219 | Total ₹11,189,563 | Positions 16/20 | Slot ₹559,478

========================================================================
  REBALANCE #88  —  01 Apr 2022
  NAV: ₹12,986,635  |  Slot: ₹649,332  |  Cash: ₹125,219
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BSE         1      FIN SVC    01-Jun-21   97.3        297.8       4813   ₹964,727      +206.0%   304d  
  DEEPAKNTR   52     MFG        01-Apr-21   1,618.8     2,267.7     241    ₹156,390      +40.1%    365d  
  IRCTC       27     PSE        01-Oct-21   725.8       765.4       710    ₹28,131       +5.5%     182d  
  TECHM       67     IT         01-Nov-21   1,277.2     1,260.2     411    ₹-6,983       -1.3%     151d  
  ESCORTS     73     MFG        03-Jan-22   1,852.9     1,654.2     310    ₹-61,595      -10.7%    88d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HAL         5      DEFENCE    2.725    0.84   +59.8%    +28.3%    723.8       897    ₹649,252      +7.9%     
  PERSISTENT  6      IT         2.636    0.94   +163.5%   -1.4%     2,292.8     283    ₹648,854      +4.7%     
  NTPC        8      ENERGY     2.240    0.80   +46.8%    +15.9%    126.0       5151   ₹649,270      +6.4%     
  RELIANCE    12     OIL&GAS    1.959    1.09   +33.8%    +12.6%    1,202.9     539    ₹648,353      +5.3%     
  BEL         13     DEFENCE    1.945    0.86   +84.2%    +4.0%     68.5        9473   ₹649,329      +3.3%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        6      OIL&GAS    02-Nov-20   219.0       2,246.2     1152   ₹2,335,359    +925.7%     +16.1%    
  TATAELXSI   2      IT         01-Apr-21   2,599.1     8,464.4     150    ₹879,804      +225.7%     +12.8%    
  INFY        28     IT         01-Sep-20   779.3       1,672.6     304    ₹271,565      +114.6%     +2.7%     
  ADANIENSOL  3      ENERGY     01-Sep-21   1,659.6     2,421.4     307    ₹233,888      +45.9%      +4.1%     
  SRF         4      MFG        01-Sep-21   1,958.2     2,590.2     260    ₹164,318      +32.3%      +3.3%     
  MPHASIS     18     IT         02-Aug-21   2,352.2     3,060.9     200    ₹141,746      +30.1%      +2.7%     
  POLYCAB     25     MFG        01-Jul-21   1,949.7     2,369.9     231    ₹97,068       +21.6%      +2.3%     
  ONGC        10     OIL&GAS    01-Nov-21   113.8       130.8       4618   ₹78,416       +14.9%      -1.3%     
  BAJAJHLDNG  14     FIN SVC    01-Oct-21   4,433.3     5,052.3     116    ₹71,803       +14.0%      +6.5%     
  POWERGRID   15     ENERGY     01-Feb-22   130.0       141.2       4526   ₹50,721       +8.6%       +6.2%     
  LT          54     INFRA      03-Jan-22   1,827.5     1,701.3     315    ₹-39,742      -6.9%       +1.3%     

  AFTER: Invested ₹12,552,696 | Cash ₹430,086 | Total ₹12,982,782 | Positions 16/20 | Slot ₹649,332

========================================================================
  REBALANCE #89  —  02 May 2022
  NAV: ₹13,012,212  |  Slot: ₹650,611  |  Cash: ₹430,086
========================================================================

  [REGIME OFF] Nifty 200 9,099.0 < SMA200 9,129.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        5      OIL&GAS    02-Nov-20   219.0       2,484.6     1152   ₹2,609,939    +1034.6%    +3.7%     
  TATAELXSI   6      IT         01-Apr-21   2,599.1     7,286.3     150    ₹703,083      +180.3%     -4.3%     
  INFY        83     IT         01-Sep-20   779.3       1,354.1     304    ₹174,720      +73.7% ⚠    -7.0%     
  ADANIENSOL  2      ENERGY     01-Sep-21   1,659.6     2,804.4     307    ₹351,469      +69.0%      +6.0%     
  SRF         11     MFG        01-Sep-21   1,958.2     2,474.6     260    ₹134,252      +26.4%      -1.6%     
  POLYCAB     31     MFG        01-Jul-21   1,949.7     2,400.8     231    ₹104,194      +23.1%      -2.6%     
  BAJAJHLDNG  19     FIN SVC    01-Oct-21   4,433.3     4,983.5     116    ₹63,829       +12.4%      -1.5%     
  POWERGRID   12     ENERGY     01-Feb-22   130.0       145.3       4526   ₹69,184       +11.8%      +2.0%     
  BEL         7      DEFENCE    01-Apr-22   68.5        75.9        9473   ₹69,582       +10.7%      -0.3%     
  NTPC        4      ENERGY     01-Apr-22   126.0       138.9       5151   ₹66,085       +10.2%      +2.5%     
  ONGC        38     OIL&GAS    01-Nov-21   113.8       121.8       4618   ₹36,687       +7.0%       -7.1%     
  MPHASIS     42     IT         02-Aug-21   2,352.2     2,508.1     200    ₹31,183       +6.6%       -7.0%     
  RELIANCE    8      OIL&GAS    01-Apr-22   1,202.9     1,259.3     539    ₹30,418       +4.7%       +3.4%     
  HAL         10     DEFENCE    01-Apr-22   723.8       748.0       897    ₹21,680       +3.3%       -1.7%     
  PERSISTENT  13     IT         01-Apr-22   2,292.8     2,013.4     283    ₹-79,061      -12.2%      -3.1%     
  LT          90     INFRA      03-Jan-22   1,827.5     1,596.2     315    ₹-72,868      -12.7% ⚠    -2.6%     
  ⚠  WAZ < 0 (momentum below universe mean): INFY, LT

  AFTER: Invested ₹12,582,125 | Cash ₹430,086 | Total ₹13,012,212 | Positions 16/20 | Slot ₹650,611

========================================================================
  REBALANCE #90  —  01 Jun 2022
  NAV: ₹12,620,728  |  Slot: ₹631,036  |  Cash: ₹430,086
========================================================================

  [REGIME OFF] Nifty 200 8,710.7 < SMA200 9,143.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        5      OIL&GAS    02-Nov-20   219.0       2,322.5     1152   ₹2,423,242    +960.6%     -2.8%     
  TATAELXSI   2      IT         01-Apr-21   2,599.1     8,173.1     150    ₹836,099      +214.5%     +5.6%     
  INFY        111    IT         01-Sep-20   779.3       1,312.9     304    ₹162,221      +68.5% ⚠    -0.8%     
  POLYCAB     22     MFG        01-Jul-21   1,949.7     2,426.5     231    ₹110,139      +24.5%      +0.4%     
  HAL         1      DEFENCE    01-Apr-22   723.8       899.0       897    ₹157,191      +24.2%      +9.7%     
  SRF         18     MFG        01-Sep-21   1,958.2     2,368.5     260    ₹106,678      +21.0%      +1.5%     
  ADANIENSOL  —      ENERGY     01-Sep-21   1,659.6     1,957.9     307    ₹91,578       +18.0%      -13.5%    
  BEL         4      DEFENCE    01-Apr-22   68.5        78.4        9473   ₹93,581       +14.4%      +6.0%     
  POWERGRID   10     ENERGY     01-Feb-22   130.0       143.8       4526   ₹62,509       +10.6%      -0.3%     
  NTPC        7      ENERGY     01-Apr-22   126.0       138.3       5151   ₹62,906       +9.7%       +3.0%     
  BAJAJHLDNG  35     FIN SVC    01-Oct-21   4,433.3     4,726.9     116    ₹34,059       +6.6%       +0.4%     
  ONGC        49     OIL&GAS    01-Nov-21   113.8       116.7       4618   ₹12,945       +2.5%       -3.5%     
  RELIANCE    17     OIL&GAS    01-Apr-22   1,202.9     1,192.8     539    ₹-5,456       -0.8%       +1.5%     
  MPHASIS     81     IT         02-Aug-21   2,352.2     2,315.3     200    ₹-7,380       -1.6% ⚠     -1.7%     
  LT          —      INFRA      03-Jan-22   1,827.5     1,566.3     315    ₹-82,284      -14.3%      +1.6%     
  PERSISTENT  37     IT         01-Apr-22   2,292.8     1,815.3     283    ₹-135,136     -20.8%      -0.8%     
  ⚠  WAZ < 0 (momentum below universe mean): MPHASIS, INFY

  AFTER: Invested ₹12,190,642 | Cash ₹430,086 | Total ₹12,620,728 | Positions 16/20 | Slot ₹631,036

========================================================================
  REBALANCE #91  —  01 Jul 2022
  NAV: ₹12,082,681  |  Slot: ₹604,134  |  Cash: ₹430,086
========================================================================

  [REGIME OFF] Nifty 200 8,294.9 < SMA200 9,086.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        6      OIL&GAS    02-Nov-20   219.0       2,385.5     1152   ₹2,495,827    +989.3%     +2.0%     
  TATAELXSI   7      IT         01-Apr-21   2,599.1     7,781.4     150    ₹777,347      +199.4%     +1.4%     
  INFY        122    IT         01-Sep-20   779.3       1,313.7     304    ₹162,450      +68.6% ⚠    +1.2%     
  ADANIENSOL  —      ENERGY     01-Sep-21   1,659.6     2,400.6     307    ₹227,502      +44.7%      +9.0%     
  HAL         5      DEFENCE    01-Apr-22   723.8       824.4       897    ₹90,256       +13.9%      -3.3%     
  POLYCAB     45     MFG        01-Jul-21   1,949.7     2,132.6     231    ₹42,246       +9.4%       -2.7%     
  SRF         28     MFG        01-Sep-21   1,958.2     2,135.5     260    ₹46,110       +9.1%       -4.4%     
  BEL         12     DEFENCE    01-Apr-22   68.5        73.2        9473   ₹44,375       +6.8%       -2.0%     
  POWERGRID   24     ENERGY     01-Feb-22   130.0       129.6       4526   ₹-1,969       -0.3%       -4.0%     
  NTPC        14     ENERGY     01-Apr-22   126.0       124.0       5151   ₹-10,446      -1.6%       -2.1%     
  BAJAJHLDNG  30     FIN SVC    01-Oct-21   4,433.3     4,299.7     116    ₹-15,497      -3.0%       -2.5%     
  RELIANCE    46     OIL&GAS    01-Apr-22   1,202.9     1,090.9     539    ₹-60,335      -9.3%       -6.1%     
  ONGC        70     OIL&GAS    01-Nov-21   113.8       102.1       4618   ₹-54,325      -10.3% ⚠    -10.5%    
  MPHASIS     102    IT         02-Aug-21   2,352.2     2,051.4     200    ₹-60,158      -12.8% ⚠    -4.7%     
  LT          —      INFRA      03-Jan-22   1,827.5     1,494.2     315    ₹-104,991     -18.2%      +1.5%     
  PERSISTENT  79     IT         01-Apr-22   2,292.8     1,608.9     283    ₹-193,547     -29.8% ⚠    -3.5%     
  ⚠  WAZ < 0 (momentum below universe mean): ONGC, PERSISTENT, MPHASIS, INFY

  AFTER: Invested ₹11,652,595 | Cash ₹430,086 | Total ₹12,082,681 | Positions 16/20 | Slot ₹604,134

========================================================================
  REBALANCE #92  —  01 Aug 2022
  NAV: ₹14,201,863  |  Slot: ₹710,093  |  Cash: ₹430,086
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJ-AUTO

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIENSOL  —      ENERGY     01-Sep-21   1,659.6     3,261.8     307    ₹491,860      +96.5%    334d  
  INFY        80     IT         01-Sep-20   779.3       1,377.3     304    ₹181,792      +76.7%    699d  
  SRF         29     MFG        01-Sep-21   1,958.2     2,427.5     260    ₹122,020      +24.0%    334d  
  POLYCAB     63     MFG        01-Jul-21   1,949.7     2,294.6     231    ₹79,676       +17.7%    396d  
  POWERGRID   62     ENERGY     01-Feb-22   130.0       137.4       4526   ₹33,395       +5.7%     181d  
  RELIANCE    71     OIL&GAS    01-Apr-22   1,202.9     1,166.2     539    ₹-19,774      -3.0%     122d  
  LT          —      INFRA      03-Jan-22   1,827.5     1,746.8     315    ₹-25,420      -4.4%     210d  
  ONGC        90     OIL&GAS    01-Nov-21   113.8       107.8       4618   ₹-27,885      -5.3%     273d  
  MPHASIS     122    IT         02-Aug-21   2,352.2     2,155.2     200    ₹-39,399      -8.4%     364d  
  PERSISTENT  93     IT         01-Apr-22   2,292.8     1,780.3     283    ₹-145,039     -22.4%    122d  

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TVSMOTOR    2      AUTO       3.431    0.78   +62.0%    +47.7%    911.5       779    ₹710,051      +7.4%     
  M&M         3      AUTO       3.336    0.97   +71.6%    +39.5%    1,193.6     594    ₹708,998      +8.2%     
  ITC         5      FMCG       2.676    0.73   +54.3%    +21.3%    254.0       2795   ₹709,883      +3.9%     
  VBL         6      FMCG       2.646    0.63   +83.6%    +27.4%    183.1       3877   ₹709,956      +7.8%     
  SIEMENS     7      ENERGY     2.413    0.88   +42.9%    +25.4%    1,598.0     444    ₹709,506      +3.8%     
  CUMMINSIND  8      INFRA      2.302    0.76   +48.7%    +20.6%    1,157.0     613    ₹709,213      +6.1%     
  EICHERMOT   11     AUTO       1.955    0.96   +21.8%    +24.4%    2,962.8     239    ₹708,113      +2.7%     
  BOSCHLTD    12     AUTO       1.909    1.08   +19.1%    +26.2%    16,768.6    42     ₹704,281      +8.4%     

  HOLDS (6)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    02-Nov-20   219.0       3,213.2     1152   ₹3,449,382    +1367.3%    +13.8%    
  TATAELXSI   10     IT         01-Apr-21   2,599.1     8,324.1     150    ₹858,752      +220.3%     +5.1%     
  HAL         4      DEFENCE    01-Apr-22   723.8       963.6       897    ₹215,054      +33.1%      +8.3%     
  BEL         9      DEFENCE    01-Apr-22   68.5        90.6        9473   ₹208,594      +32.1%      +10.0%    
  BAJAJHLDNG  43     FIN SVC    01-Oct-21   4,433.3     4,946.0     116    ₹59,476       +11.6%      +7.1%     
  NTPC        38     ENERGY     01-Apr-22   126.0       138.0       5151   ₹61,543       +9.5%       +4.9%     

  AFTER: Invested ₹13,627,053 | Cash ₹568,077 | Total ₹14,195,131 | Positions 14/20 | Slot ₹710,093

========================================================================
  REBALANCE #93  —  01 Sep 2022
  NAV: ₹15,394,002  |  Slot: ₹769,700  |  Cash: ₹568,077
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
  [52w filter blocked 1: ADANIGREEN(-21.5%)]
    —

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        1      OIL&GAS    02-Nov-20   219.0       3,634.3     1152   ₹3,934,400    +1559.6%    +7.4%     
  TATAELXSI   34     IT         01-Apr-21   2,599.1     8,610.2     150    ₹901,663      +231.3%     -3.8%     
  HAL         9      DEFENCE    01-Apr-22   723.8       1,098.4     897    ₹335,973      +51.7%      +5.1%     
  BEL         4      DEFENCE    01-Apr-22   68.5        102.5       9473   ₹321,334      +49.5%      +9.4%     
  BAJAJHLDNG  28     FIN SVC    01-Oct-21   4,433.3     5,395.1     116    ₹111,576      +21.7%      +6.9%     
  NTPC        33     ENERGY     01-Apr-22   126.0       144.4       5151   ₹94,620       +14.6%      +1.8%     
  VBL         3      FMCG       01-Aug-22   183.1       206.6       3877   ₹91,000       +12.8%      +5.3%     
  EICHERMOT   14     AUTO       01-Aug-22   2,962.8     3,294.7     239    ₹79,310       +11.2%      +3.5%     
  TVSMOTOR    2      AUTO       01-Aug-22   911.5       998.5       779    ₹67,775       +9.5%       +6.7%     
  M&M         5      AUTO       01-Aug-22   1,193.6     1,265.2     594    ₹42,560       +6.0%       +4.5%     
  SIEMENS     16     ENERGY     01-Aug-22   1,598.0     1,691.8     444    ₹41,662       +5.9%       +3.0%     
  ITC         8      FMCG       01-Aug-22   254.0       262.3       2795   ₹23,317       +3.3%       +2.0%     
  CUMMINSIND  21     INFRA      01-Aug-22   1,157.0     1,166.6     613    ₹5,938        +0.8%       +1.9%     
  BOSCHLTD    19     AUTO       01-Aug-22   16,768.6    16,781.6    42     ₹547          +0.1%       +2.4%     

  AFTER: Invested ₹14,825,925 | Cash ₹568,077 | Total ₹15,394,002 | Positions 14/20 | Slot ₹769,700

========================================================================
  REBALANCE #94  —  03 Oct 2022
  NAV: ₹14,528,153  |  Slot: ₹726,408  |  Cash: ₹568,077
========================================================================

  EXITS (1)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BOSCHLTD    98     AUTO       01-Aug-22   16,768.6    14,839.5    42     ₹-81,023      -11.5%    63d   

  ENTRIES (1)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TIINDIA     1      AUTO       3.305    0.88   +97.9%    +53.1%    2,688.0     270    ₹725,757      +3.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        5      OIL&GAS    02-Nov-20   219.0       3,096.6     1152   ₹3,314,955    +1314.0%    -10.2%    
  TATAELXSI   51     IT         01-Apr-21   2,599.1     7,934.6     150    ₹800,320      +205.3%     -4.7%     
  HAL         2      DEFENCE    01-Apr-22   723.8       1,092.9     897    ₹331,054      +51.0%      -3.3%     
  BAJAJHLDNG  4      FIN SVC    01-Oct-21   4,433.3     6,221.9     116    ₹207,478      +40.3%      +1.1%     
  BEL         8      DEFENCE    01-Apr-22   68.5        94.6        9473   ₹247,122      +38.1%      -5.2%     
  VBL         3      FMCG       01-Aug-22   183.1       212.1       3877   ₹112,518      +15.8%      +0.2%     
  NTPC        24     ENERGY     01-Apr-22   126.0       144.1       5151   ₹93,000       +14.3%      -1.8%     
  EICHERMOT   18     AUTO       01-Aug-22   2,962.8     3,344.6     239    ₹91,254       +12.9%      -2.5%     
  TVSMOTOR    6      AUTO       01-Aug-22   911.5       979.2       779    ₹52,757       +7.4%       -2.8%     
  ITC         11     FMCG       01-Aug-22   254.0       267.9       2795   ₹39,015       +5.5%       -1.9%     
  M&M         13     AUTO       01-Aug-22   1,193.6     1,206.7     594    ₹7,790        +1.1%       -1.5%     
  SIEMENS     31     ENERGY     01-Aug-22   1,598.0     1,568.0     444    ₹-13,334      -1.9%       -4.7%     
  CUMMINSIND  29     INFRA      01-Aug-22   1,157.0     1,129.1     613    ₹-17,082      -2.4%       -1.9%     

  AFTER: Invested ₹14,062,575 | Cash ₹464,716 | Total ₹14,527,291 | Positions 14/20 | Slot ₹726,408

========================================================================
  REBALANCE #95  —  01 Nov 2022
  NAV: ₹15,832,830  |  Slot: ₹791,641  |  Cash: ₹464,716
========================================================================

  EXITS (1)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TATAELXSI   117    IT         01-Apr-21   2,599.1     6,639.5     150    ₹606,068      +155.5%   579d  

  ENTRIES (1)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  AMBUJACEM   3      INFRA      3.070    0.86   +46.1%    +46.7%    534.7       1480   ₹791,292      +5.5%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        4      OIL&GAS    02-Nov-20   219.0       3,676.2     1152   ₹3,982,660    +1578.7%    +9.2%     
  HAL         2      DEFENCE    01-Apr-22   723.8       1,207.2     897    ₹433,614      +66.8%      +3.3%     
  BEL         11     DEFENCE    01-Apr-22   68.5        104.3       9473   ₹338,634      +52.2%      +3.1%     
  BAJAJHLDNG  9      FIN SVC    01-Oct-21   4,433.3     6,419.1     116    ₹230,354      +44.8%      +2.0%     
  NTPC        14     ENERGY     01-Apr-22   126.0       163.4       5151   ₹192,293      +29.6%      +7.8%     
  EICHERMOT   8      AUTO       01-Aug-22   2,962.8     3,668.2     239    ₹168,586      +23.8%      +4.3%     
  TVSMOTOR    1      AUTO       01-Aug-22   911.5       1,119.4     779    ₹161,948      +22.8%      +2.6%     
  VBL         5      FMCG       01-Aug-22   183.1       219.4       3877   ₹140,800      +19.8%      +5.2%     
  ITC         7      FMCG       01-Aug-22   254.0       288.7       2795   ₹96,960       +13.7%      +2.5%     
  CUMMINSIND  15     INFRA      01-Aug-22   1,157.0     1,286.6     613    ₹79,471       +11.2%      +8.4%     
  M&M         12     AUTO       01-Aug-22   1,193.6     1,305.8     594    ₹66,618       +9.4%       +5.7%     
  SIEMENS     25     ENERGY     01-Aug-22   1,598.0     1,706.0     444    ₹47,937       +6.8%       +3.4%     
  TIINDIA     6      AUTO       03-Oct-22   2,688.0     2,785.9     270    ₹26,442       +3.6%       +2.9%     

  AFTER: Invested ₹15,163,474 | Cash ₹668,416 | Total ₹15,831,890 | Positions 14/20 | Slot ₹791,641

========================================================================
  REBALANCE #96  —  01 Dec 2022
  NAV: ₹15,709,350  |  Slot: ₹785,468  |  Cash: ₹668,416
========================================================================

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ATGL        21     OIL&GAS    02-Nov-20   219.0       3,607.5     1152   ₹3,903,511    +1547.3%  759d  
  SIEMENS     50     ENERGY     01-Aug-22   1,598.0     1,607.3     444    ₹4,141        +0.6%     122d  

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  UNIONBANK   1      PSU BNK    4.613    1.16   +96.9%    +92.4%    72.2        10883  ₹785,467      +14.1%    
  BANKINDIA   3      PSU BNK    3.185    1.19   +53.1%    +60.1%    73.9        10628  ₹785,456      +9.9%     
  INDIANB     4      PSU BNK    3.127    1.19   +94.8%    +42.1%    250.0       3142   ₹785,360      +4.6%     
  SUNPHARMA   9      HEALTH     2.352    0.63   +37.9%    +17.2%    1,009.8     777    ₹784,631      +2.3%     
  PFC         10     FIN SVC    2.296    0.88   +29.9%    +20.9%    94.9        8275   ₹785,389      +11.0%    
  AXISBANK    12     PVT BNK    2.096    1.04   +36.8%    +20.3%    901.5       871    ₹785,173      +3.3%     
  FEDERALBNK  13     PVT BNK    1.940    1.17   +52.3%    +13.4%    130.1       6035   ₹785,431      -0.2%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         2      DEFENCE    01-Apr-22   723.8       1,323.9     897    ₹538,314      +82.9%      +4.1%     
  BEL         24     DEFENCE    01-Apr-22   68.5        100.1       9473   ₹299,025      +46.1%      -2.3%     
  VBL         7      FMCG       01-Aug-22   183.1       250.6       3877   ₹261,612      +36.8%      +9.5%     
  BAJAJHLDNG  26     FIN SVC    01-Oct-21   4,433.3     6,038.0     116    ₹186,150      +36.2%      -2.6%     
  NTPC        25     ENERGY     01-Apr-22   126.0       154.8       5151   ₹147,854      +22.8%      +1.2%     
  CUMMINSIND  8      INFRA      01-Aug-22   1,157.0     1,372.2     613    ₹131,915      +18.6%      +5.9%     
  TVSMOTOR    20     AUTO       01-Aug-22   911.5       1,032.8     779    ₹94,518       +13.3%      -2.1%     
  EICHERMOT   31     AUTO       01-Aug-22   2,962.8     3,319.6     239    ₹85,276       +12.0%      -1.4%     
  ITC         11     FMCG       01-Aug-22   254.0       280.5       2795   ₹73,990       +10.4%      -0.9%     
  AMBUJACEM   5      INFRA      01-Nov-22   534.7       570.9       1480   ₹53,669       +6.8%       +3.4%     
  M&M         29     AUTO       01-Aug-22   1,193.6     1,247.3     594    ₹31,906       +4.5%       +1.6%     
  TIINDIA     14     AUTO       03-Oct-22   2,688.0     2,806.1     270    ₹31,900       +4.4%       +5.2%     

  AFTER: Invested ₹15,668,410 | Cash ₹34,413 | Total ₹15,702,823 | Positions 19/20 | Slot ₹785,468

========================================================================
  REBALANCE #97  —  02 Jan 2023
  NAV: ₹15,570,966  |  Slot: ₹778,548  |  Cash: ₹34,413
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJAJHLDNG  95     FIN SVC    01-Oct-21   4,433.3     5,361.3     116    ₹107,645      +20.9%    458d  
  BANKINDIA   1      PSU BNK    01-Dec-22   73.9        81.4        10628  ₹79,396       +10.1%    32d   
  INDIANB     4      PSU BNK    01-Dec-22   250.0       267.8       3142   ₹56,046       +7.1%     32d   
  EICHERMOT   76     AUTO       01-Aug-22   2,962.8     3,118.2     239    ₹37,130       +5.2%     154d  
  FEDERALBNK  14     PVT BNK    01-Dec-22   130.1       135.0       6035   ₹29,594       +3.8%     32d   
  UNIONBANK   2      PSU BNK    01-Dec-22   72.2        72.1        10883  ₹-964         -0.1%     32d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RECLTD      3      FIN SVC    3.233    0.92   +38.7%    +34.6%    100.3       7763   ₹778,458      +6.9%     
  COALINDIA   8      ENERGY     2.255    0.83   +68.0%    +12.7%    179.2       4344   ₹778,373      -0.1%     
  SBIN        10     PSU BNK    2.175    1.07   +34.8%    +15.4%    568.7       1368   ₹778,047      +1.3%     
  HINDZINC    11     METAL      2.153    0.67   +15.9%    +27.9%    252.5       3083   ₹778,403      +2.2%     
  JSWSTEEL    13     METAL      1.824    1.19   +21.6%    +22.7%    764.0       1019   ₹778,506      +3.5%     
  IOC         14     OIL&GAS    1.806    0.67   +11.4%    +16.5%    64.8        12018  ₹778,490      +3.0%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         10     DEFENCE    01-Apr-22   723.8       1,220.7     897    ₹445,693      +68.6%      -1.7%     
  VBL         6      FMCG       01-Aug-22   183.1       264.3       3877   ₹314,563      +44.3%      +0.3%     
  BEL         35     DEFENCE    01-Apr-22   68.5        96.4        9473   ₹263,513      +40.6%      -1.1%     
  NTPC        23     ENERGY     01-Apr-22   126.0       151.0       5151   ₹128,412      +19.8%      -0.0%     
  TVSMOTOR    18     AUTO       01-Aug-22   911.5       1,055.5     779    ₹112,180      +15.8%      +1.7%     
  CUMMINSIND  15     INFRA      01-Aug-22   1,157.0     1,322.3     613    ₹101,379      +14.3%      -1.9%     
  ITC         17     FMCG       01-Aug-22   254.0       274.9       2795   ₹58,522       +8.2%       -1.0%     
  PFC         5      FIN SVC    01-Dec-22   94.9        102.6       8275   ₹63,512       +8.1%       +7.9%     
  AXISBANK    11     PVT BNK    01-Dec-22   901.5       939.1       871    ₹32,793       +4.2%       +1.8%     
  TIINDIA     38     AUTO       03-Oct-22   2,688.0     2,773.5     270    ₹23,081       +3.2%       -1.6%     
  M&M         27     AUTO       01-Aug-22   1,193.6     1,217.8     594    ₹14,378       +2.0%       +0.3%     
  AMBUJACEM   43     INFRA      01-Nov-22   534.7       517.4       1480   ₹-25,525      -3.2%       -2.7%     
  SUNPHARMA   34     HEALTH     01-Dec-22   1,009.8     962.1       777    ₹-37,078      -4.7%       -0.6%     

  AFTER: Invested ₹15,533,892 | Cash ₹31,529 | Total ₹15,565,421 | Positions 19/20 | Slot ₹778,548

========================================================================
  REBALANCE #98  —  01 Feb 2023
  NAV: ₹14,808,825  |  Slot: ₹740,441  |  Cash: ₹31,529
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BEL         64     DEFENCE    01-Apr-22   68.5        87.4        9473   ₹178,830      +27.5%    306d  
  AXISBANK    59     PVT BNK    01-Dec-22   901.5       855.0       871    ₹-40,481      -5.2%     62d   
  SBIN        84     PSU BNK    02-Jan-23   568.7       489.9       1368   ₹-107,836     -13.9%    30d   
  AMBUJACEM   131    INFRA      01-Nov-22   534.7       328.3       1480   ₹-305,361     -38.6%    92d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ABBOTINDIA  10     HEALTH     2.090    0.35   +29.4%    +5.6%     19,849.0    37     ₹734,414      -3.6%     
  LINDEINDIA  11     MFG        1.934    0.75   +27.1%    +11.9%    3,352.6     220    ₹737,568      +0.3%     
  INDIGO      12     CONSUMP    1.875    1.15   +11.9%    +15.5%    2,081.4     355    ₹738,891      +0.2%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         14     DEFENCE    01-Apr-22   723.8       1,136.0     897    ₹369,773      +57.0%      -5.1%     
  VBL         7      FMCG       01-Aug-22   183.1       232.5       3877   ₹191,407      +27.0%      -4.7%     
  NTPC        41     ENERGY     01-Apr-22   126.0       152.8       5151   ₹137,670      +21.2%      +1.1%     
  CUMMINSIND  6      INFRA      01-Aug-22   1,157.0     1,361.3     613    ₹125,276      +17.7%      -0.0%     
  ITC         1      FMCG       01-Aug-22   254.0       298.5       2795   ₹124,432      +17.5%      +6.2%     
  TVSMOTOR    22     AUTO       01-Aug-22   911.5       1,001.7     779    ₹70,266       +9.9%       -0.2%     
  M&M         12     AUTO       01-Aug-22   1,193.6     1,303.8     594    ₹65,472       +9.2%       +2.7%     
  HINDZINC    10     METAL      02-Jan-23   252.5       268.2       3083   ₹48,517       +6.2%       +0.1%     
  IOC         19     OIL&GAS    02-Jan-23   64.8        66.3        12018  ₹17,965       +2.3%       -1.4%     
  PFC         8      FIN SVC    01-Dec-22   94.9        93.4        8275   ₹-12,192      -1.6%       -6.0%     
  COALINDIA   24     ENERGY     02-Jan-23   179.2       175.9       4344   ₹-14,200      -1.8%       -0.7%     
  RECLTD      4      FIN SVC    02-Jan-23   100.3       97.8        7763   ₹-19,073      -2.5%       -2.1%     
  SUNPHARMA   38     HEALTH     01-Dec-22   1,009.8     979.4       777    ₹-23,656      -3.0%       -1.5%     
  TIINDIA     36     AUTO       03-Oct-22   2,688.0     2,605.8     270    ₹-22,194      -3.1%       -1.5%     
  JSWSTEEL    34     METAL      02-Jan-23   764.0       719.1       1019   ₹-45,789      -5.9%       -1.0%     

  AFTER: Invested ₹14,259,175 | Cash ₹547,024 | Total ₹14,806,199 | Positions 18/20 | Slot ₹740,441

========================================================================
  REBALANCE #99  —  01 Mar 2023
  NAV: ₹15,235,219  |  Slot: ₹761,761  |  Cash: ₹547,024
========================================================================

  [REGIME OFF] Nifty 200 9,053.3 < SMA200 9,190.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         7      DEFENCE    01-Apr-22   723.8       1,289.7     897    ₹507,648      +78.2%      +5.6%     
  VBL         4      FMCG       01-Aug-22   183.1       266.4       3877   ₹322,863      +45.5%      +4.6%     
  CUMMINSIND  2      INFRA      01-Aug-22   1,157.0     1,540.7     613    ₹235,256      +33.2%      +3.2%     
  NTPC        15     ENERGY     01-Apr-22   126.0       158.8       5151   ₹168,920      +26.0%      +2.2%     
  ITC         1      FMCG       01-Aug-22   254.0       317.9       2795   ₹178,697      +25.2%      +1.3%     
  TVSMOTOR    12     AUTO       01-Aug-22   911.5       1,050.7     779    ₹108,459      +15.3%      -1.6%     
  LINDEINDIA  10     MFG        01-Feb-23   3,352.6     3,709.2     220    ₹78,448       +10.6%      +2.4%     
  PFC         6      FIN SVC    01-Dec-22   94.9        104.9       8275   ₹82,685       +10.5%      +4.0%     
  TIINDIA     18     AUTO       03-Oct-22   2,688.0     2,773.4     270    ₹23,053       +3.2%       +7.3%     
  M&M         16     AUTO       01-Aug-22   1,193.6     1,226.8     594    ₹19,705       +2.8%       -3.6%     
  COALINDIA   21     ENERGY     02-Jan-23   179.2       178.7       4344   ₹-2,233       -0.3%       +1.5%     
  HINDZINC    31     METAL      02-Jan-23   252.5       251.4       3083   ₹-3,353       -0.4%       -2.9%     
  ABBOTINDIA  38     HEALTH     01-Feb-23   19,849.0    19,619.0    37     ₹-8,509       -1.2%       -0.4%     
  IOC         48     OIL&GAS    02-Jan-23   64.8        63.9        12018  ₹-10,480      -1.3%       -2.0%     
  RECLTD      11     FIN SVC    02-Jan-23   100.3       98.5        7763   ₹-14,033      -1.8%       +0.3%     
  SUNPHARMA   70     HEALTH     01-Dec-22   1,009.8     932.7       777    ₹-59,945      -7.6% ⚠     -2.7%     
  INDIGO      80     CONSUMP    01-Feb-23   2,081.4     1,840.6     355    ₹-85,482      -11.6% ⚠    -4.5%     
  JSWSTEEL    68     METAL      02-Jan-23   764.0       666.8       1019   ₹-99,008      -12.7% ⚠    -4.3%     
  ⚠  WAZ < 0 (momentum below universe mean): JSWSTEEL, SUNPHARMA, INDIGO

  AFTER: Invested ₹14,688,195 | Cash ₹547,024 | Total ₹15,235,219 | Positions 18/20 | Slot ₹761,761

========================================================================
  REBALANCE #100  —  03 Apr 2023
  NAV: ₹15,498,081  |  Slot: ₹774,904  |  Cash: ₹547,024
========================================================================

  [REGIME OFF] Nifty 200 9,030.7 < SMA200 9,229.7 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         5      DEFENCE    01-Apr-22   723.8       1,312.6     897    ₹528,194      +81.4%      +1.9%     
  VBL         3      FMCG       01-Aug-22   183.1       280.4       3877   ₹377,082      +53.1%      +5.0%     
  CUMMINSIND  4      INFRA      01-Aug-22   1,157.0     1,543.4     613    ₹236,877      +33.4%      -1.1%     
  NTPC        11     ENERGY     01-Apr-22   126.0       164.0       5151   ₹195,283      +30.1%      +1.8%     
  ITC         1      FMCG       01-Aug-22   254.0       318.1       2795   ₹179,166      +25.2%      -0.3%     
  LINDEINDIA  —      OTHER      01-Feb-23   3,352.6     4,026.1     220    ₹148,181      +20.1%      +4.7%     
  TVSMOTOR    10     AUTO       01-Aug-22   911.5       1,072.3     779    ₹125,246      +17.6%      +2.1%     
  PFC         8      FIN SVC    01-Dec-22   94.9        108.8       8275   ₹115,213      +14.7%      +1.5%     
  ABBOTINDIA  19     HEALTH     01-Feb-23   19,849.0    21,340.6    37     ₹55,190       +7.5%       +4.7%     
  HINDZINC    31     METAL      02-Jan-23   252.5       261.3       3083   ₹27,135       +3.5%       +1.9%     
  COALINDIA   34     ENERGY     02-Jan-23   179.2       179.8       4344   ₹2,734        +0.4%       +2.0%     
  RECLTD      17     FIN SVC    02-Jan-23   100.3       100.2       7763   ₹-413         -0.1%       +0.3%     
  IOC         56     OIL&GAS    02-Jan-23   64.8        64.4        12018  ₹-4,990       -0.6%       -1.0%     
  TIINDIA     43     AUTO       03-Oct-22   2,688.0     2,553.0     270    ₹-36,439      -5.0%       -0.7%     
  M&M         28     AUTO       01-Aug-22   1,193.6     1,128.2     594    ₹-38,865      -5.5%       -1.4%     
  SUNPHARMA   57     HEALTH     01-Dec-22   1,009.8     951.9       777    ₹-45,025      -5.7%       +0.5%     
  INDIGO      84     CONSUMP    01-Feb-23   2,081.4     1,896.8     355    ₹-65,529      -8.9% ⚠     +1.3%     
  JSWSTEEL    98     METAL      02-Jan-23   764.0       672.2       1019   ₹-93,485      -12.0% ⚠    +0.9%     
  ⚠  WAZ < 0 (momentum below universe mean): INDIGO, JSWSTEEL

  AFTER: Invested ₹14,951,057 | Cash ₹547,024 | Total ₹15,498,081 | Positions 18/20 | Slot ₹774,904

========================================================================
  REBALANCE #101  —  02 May 2023
  NAV: ₹16,282,313  |  Slot: ₹814,116  |  Cash: ₹547,024
========================================================================

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LINDEINDIA  —      OTHER      01-Feb-23   3,352.6     3,983.5     220    ₹138,797      +18.8%    90d   
  HINDZINC    58     METAL      02-Jan-23   252.5       276.2       3083   ₹73,057       +9.4%     120d  
  IOC         87     OIL&GAS    02-Jan-23   64.8        68.2        12018  ₹41,420       +5.3%     120d  
  M&M         59     AUTO       01-Aug-22   1,193.6     1,193.2     594    ₹-258         -0.0%     274d  
  INDIGO      76     CONSUMP    01-Feb-23   2,081.4     2,063.9     355    ₹-6,202       -0.8%     90d   
  TIINDIA     57     AUTO       03-Oct-22   2,688.0     2,573.5     270    ₹-30,923      -4.3%     211d  
  JSWSTEEL    70     METAL      02-Jan-23   764.0       727.1       1019   ₹-37,555      -4.8%     120d  
  SUNPHARMA   92     HEALTH     01-Dec-22   1,009.8     946.0       777    ₹-49,558      -6.3%     152d  

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ZYDUSLIFE   3      HEALTH     2.663    0.59   +52.6%    +18.3%    505.1       1611   ₹813,739      +1.9%     
  SIEMENS     6      ENERGY     2.571    0.81   +55.7%    +15.9%    2,039.2     399    ₹813,659      +4.3%     
  INDIANB     8      PSU BNK    2.497    1.18   +115.1%   +13.4%    298.0       2731   ₹813,936      +7.7%     
  BAJAJ-AUTO  9      AUTO       2.489    0.65   +26.7%    +22.0%    4,163.7     195    ₹811,919      +6.2%     
  BOSCHLTD    11     AUTO       2.449    0.99   +43.2%    +17.4%    19,016.2    42     ₹798,679      +4.7%     
  APOLLOTYRE  12     AUTO       2.221    1.03   +83.6%    +7.9%     332.7       2446   ₹813,879      +4.6%     
  DRREDDY     13     HEALTH     2.211    0.45   +21.1%    +16.4%    970.5       838    ₹813,245      +3.2%     
  ALKEM       15     HEALTH     2.063    0.40   +7.7%     +18.4%    3,423.3     237    ₹811,332      +4.0%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         9      DEFENCE    01-Apr-22   723.8       1,422.0     897    ₹626,257      +96.5%      +4.6%     
  VBL         13     FMCG       01-Aug-22   183.1       280.7       3877   ₹378,339      +53.3%      -0.1%     
  ITC         1      FMCG       01-Aug-22   254.0       356.3       2795   ₹286,045      +40.3%      +5.2%     
  CUMMINSIND  5      INFRA      01-Aug-22   1,157.0     1,538.5     613    ₹233,900      +33.0%      +1.9%     
  NTPC        34     ENERGY     01-Apr-22   126.0       161.9       5151   ₹184,833      +28.5%      +1.8%     
  PFC         4      FIN SVC    01-Dec-22   94.9        121.9       8275   ₹222,960      +28.4%      +6.9%     
  TVSMOTOR    2      AUTO       01-Aug-22   911.5       1,144.6     779    ₹181,613      +25.6%      +3.5%     
  RECLTD      7      FIN SVC    02-Jan-23   100.3       116.8       7763   ₹128,154      +16.5%      +9.9%     
  ABBOTINDIA  35     HEALTH     01-Feb-23   19,849.0    21,459.3    37     ₹59,582       +8.1%       +0.6%     
  COALINDIA   26     ENERGY     02-Jan-23   179.2       192.8       4344   ₹59,326       +7.6%       +3.7%     

  AFTER: Invested ₹16,065,654 | Cash ₹208,953 | Total ₹16,274,606 | Positions 18/20 | Slot ₹814,116

========================================================================
  REBALANCE #102  —  01 Jun 2023
  NAV: ₹16,666,572  |  Slot: ₹833,329  |  Cash: ₹208,953
========================================================================

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  NTPC        95     ENERGY     01-Apr-22   126.0       160.8       5151   ₹179,133      +27.6%    426d  
  COALINDIA   55     ENERGY     02-Jan-23   179.2       188.1       4344   ₹38,925       +5.0%     150d  
  ABBOTINDIA  64     HEALTH     01-Feb-23   19,849.0    20,714.1    37     ₹32,008       +4.4%     120d  
  ZYDUSLIFE   —      HEALTH     02-May-23   505.1       501.6       1611   ₹-5,588       -0.7%     30d   
  ALKEM       —      HEALTH     02-May-23   3,423.3     3,289.0     237    ₹-31,845      -3.9%     30d   
  BOSCHLTD    60     AUTO       02-May-23   19,016.2    17,867.1    42     ₹-48,260      -6.0%     30d   
  DRREDDY     107    HEALTH     02-May-23   970.5       892.4       838    ₹-65,440      -8.0%     30d   
  INDIANB     65     PSU BNK    02-May-23   298.0       248.9       2731   ₹-134,152     -16.5%    30d   

  ENTRIES (7)
  [52w filter blocked 1: ADANIGREEN(-61.4%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  CHOLAFIN    4      FIN SVC    2.758    1.11   +63.8%    +36.8%    1,039.4     801    ₹832,528      +3.0%     
  SYNGENE     6      HEALTH     2.421    0.43   +37.7%    +28.3%    720.6       1156   ₹833,062      +3.7%     
  TORNTPHARM  8      HEALTH     2.192    0.39   +24.5%    +19.9%    1,716.8     485    ₹832,663      +4.8%     
  NESTLEIND   11     FMCG       2.105    0.49   +25.3%    +17.7%    1,062.6     784    ₹833,043      +1.6%     
  BEL         15     DEFENCE    1.812    0.93   +52.6%    +19.2%    109.9       7581   ₹833,319      +3.8%     
  MAXHEALTH   16     HEALTH     1.793    0.25   +45.8%    +23.5%    530.5       1570   ₹832,940      +2.0%     
  TRENT       17     CONSUMP    1.776    1.09   +49.1%    +19.8%    1,558.5     534    ₹832,229      +4.3%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         13     DEFENCE    01-Apr-22   723.8       1,490.9     897    ₹688,094      +106.0%     +1.9%     
  VBL         4      FMCG       01-Aug-22   183.1       335.0       3877   ₹588,983      +83.0%      +5.7%     
  ITC         1      FMCG       01-Aug-22   254.0       377.4       2795   ₹345,001      +48.6%      +3.5%     
  CUMMINSIND  16     INFRA      01-Aug-22   1,157.0     1,683.2     613    ₹322,601      +45.5%      +4.5%     
  TVSMOTOR    10     AUTO       01-Aug-22   911.5       1,257.5     779    ₹269,513      +38.0%      +1.9%     
  PFC         5      FIN SVC    01-Dec-22   94.9        128.6       8275   ₹278,721      +35.5%      +7.0%     
  RECLTD      3      FIN SVC    02-Jan-23   100.3       120.3       7763   ₹155,728      +20.0%      +5.1%     
  APOLLOTYRE  8      AUTO       02-May-23   332.7       375.8       2446   ₹105,320      +12.9%      +4.2%     
  BAJAJ-AUTO  17     AUTO       02-May-23   4,163.7     4,298.5     195    ₹26,282       +3.2%       +2.5%     
  SIEMENS     37     ENERGY     02-May-23   2,039.2     2,059.2     399    ₹7,950        +1.0%       -0.9%     

  AFTER: Invested ₹16,109,634 | Cash ₹550,015 | Total ₹16,659,650 | Positions 17/20 | Slot ₹833,329

========================================================================
  REBALANCE #103  —  03 Jul 2023
  NAV: ₹18,104,489  |  Slot: ₹905,224  |  Cash: ₹550,015
========================================================================

  EXITS (1)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PFC         2      FIN SVC    01-Dec-22   94.9        157.8       8275   ₹520,266      +66.2%    214d  

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TATACOMM    8      CONSUMP    1.982    0.94   +74.9%    +32.6%    1,538.7     588    ₹904,761      +4.7%     
  LT          10     INFRA      1.872    0.91   +66.3%    +15.0%    2,362.2     383    ₹904,722      +3.4%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         7      DEFENCE    01-Apr-22   723.8       1,824.5     897    ₹987,318      +152.1%     +3.8%     
  VBL         37     FMCG       01-Aug-22   183.1       324.7       3877   ₹548,762      +77.3%      +1.6%     
  CUMMINSIND  17     INFRA      01-Aug-22   1,157.0     1,817.1     613    ₹404,671      +57.1%      +2.6%     
  ITC         6      FMCG       01-Aug-22   254.0       397.6       2795   ₹401,500      +56.6%      +3.9%     
  TVSMOTOR    24     AUTO       01-Aug-22   911.5       1,303.2     779    ₹305,127      +43.0%      +0.2%     
  RECLTD      4      FIN SVC    02-Jan-23   100.3       142.3       7763   ₹326,154      +41.9%      +6.9%     
  APOLLOTYRE  9      AUTO       02-May-23   332.7       381.6       2446   ₹119,425      +14.7%      -1.3%     
  MAXHEALTH   27     HEALTH     01-Jun-23   530.5       603.1       1570   ₹113,934      +13.7%      +4.4%     
  TRENT       12     CONSUMP    01-Jun-23   1,558.5     1,747.2     534    ₹100,753      +12.1%      +3.8%     
  CHOLAFIN    5      FIN SVC    01-Jun-23   1,039.4     1,164.1     801    ₹99,889       +12.0%      +6.3%     
  BEL         14     DEFENCE    01-Jun-23   109.9       120.5       7581   ₹80,084       +9.6%       +2.9%     
  TORNTPHARM  19     HEALTH     01-Jun-23   1,716.8     1,837.0     485    ₹58,296       +7.0%       +2.7%     
  SIEMENS     47     ENERGY     02-May-23   2,039.2     2,171.7     399    ₹52,851       +6.5%       +0.5%     
  BAJAJ-AUTO  29     AUTO       02-May-23   4,163.7     4,398.5     195    ₹45,795       +5.6%       +1.6%     
  SYNGENE     22     HEALTH     01-Jun-23   720.6       754.7       1156   ₹39,331       +4.7%       +3.1%     
  NESTLEIND   30     FMCG       01-Jun-23   1,062.6     1,099.4     784    ₹28,877       +3.5%       +0.6%     

  AFTER: Invested ₹18,058,301 | Cash ₹44,039 | Total ₹18,102,340 | Positions 18/20 | Slot ₹905,224

========================================================================
  REBALANCE #104  —  01 Aug 2023
  NAV: ₹18,843,563  |  Slot: ₹942,178  |  Cash: ₹44,039
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VBL         49     FMCG       01-Aug-22   183.1       317.7       3877   ₹521,614      +73.5%    365d  
  BAJAJ-AUTO  46     AUTO       02-May-23   4,163.7     4,697.4     195    ₹104,078      +12.8%    91d   
  SIEMENS     53     ENERGY     02-May-23   2,039.2     2,267.0     399    ₹90,860       +11.2%    91d   
  MAXHEALTH   35     HEALTH     01-Jun-23   530.5       569.7       1570   ₹61,500       +7.4%     61d   
  NESTLEIND   85     FMCG       01-Jun-23   1,062.6     1,097.5     784    ₹27,397       +3.3%     61d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  PFC         1      FIN SVC    4.015    1.20   +150.3%   +55.3%    185.9       5069   ₹942,111      +10.0%    
  POLYCAB     3      MFG        3.148    0.82   +109.5%   +41.5%    4,561.7     206    ₹939,708      +7.3%     
  NTPC        5      ENERGY     2.500    0.70   +56.9%    +27.7%    207.6       4537   ₹941,986      +12.7%    
  COLPAL      7      FMCG       2.167    0.36   +31.3%    +28.3%    1,880.1     501    ₹941,914      +7.0%     
  HDFCAMC     9      FIN SVC    2.076    1.19   +40.8%    +46.2%    1,206.3     781    ₹942,154      +6.3%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         8      DEFENCE    01-Apr-22   723.8       1,870.9     897    ₹1,028,926    +158.5%     +0.8%     
  RECLTD      2      FIN SVC    02-Jan-23   100.3       175.8       7763   ₹586,313      +75.3%      +14.9%    
  CUMMINSIND  12     INFRA      01-Aug-22   1,157.0     1,865.6     613    ₹434,386      +61.2%      +1.2%     
  ITC         15     FMCG       01-Aug-22   254.0       399.0       2795   ₹405,219      +57.1%      -0.7%     
  TVSMOTOR    24     AUTO       01-Aug-22   911.5       1,353.9     779    ₹344,668      +48.5%      +1.7%     
  APOLLOTYRE  10     AUTO       02-May-23   332.7       415.0       2446   ₹201,300      +24.7%      +2.1%     
  BEL         23     DEFENCE    01-Jun-23   109.9       127.0       7581   ₹129,537      +15.5%      +3.1%     
  TORNTPHARM  14     HEALTH     01-Jun-23   1,716.8     1,927.0     485    ₹101,927      +12.2%      +2.0%     
  SYNGENE     25     HEALTH     01-Jun-23   720.6       800.9       1156   ₹92,793       +11.1%      +3.4%     
  TATACOMM    6      CONSUMP    03-Jul-23   1,538.7     1,705.7     588    ₹98,174       +10.9%      +6.9%     
  TRENT       33     CONSUMP    01-Jun-23   1,558.5     1,703.3     534    ₹77,342       +9.3%       -0.1%     
  LT          21     INFRA      03-Jul-23   2,362.2     2,566.8     383    ₹78,378       +8.7%       +4.5%     
  CHOLAFIN    13     FIN SVC    01-Jun-23   1,039.4     1,126.2     801    ₹69,577       +8.4%       -0.7%     

  AFTER: Invested ₹18,700,432 | Cash ₹137,540 | Total ₹18,837,972 | Positions 18/20 | Slot ₹942,178

========================================================================
  REBALANCE #105  —  01 Sep 2023
  NAV: ₹19,209,069  |  Slot: ₹960,453  |  Cash: ₹137,540
========================================================================
  [SECTOR CAP≤4] dropped: SHRIRAMFIN

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  CUMMINSIND  68     INFRA      01-Aug-22   1,157.0     1,651.4     613    ₹303,100      +42.7%    396d  
  APOLLOTYRE  58     AUTO       02-May-23   332.7       373.8       2446   ₹100,317      +12.3%    122d  
  SYNGENE     50     HEALTH     01-Jun-23   720.6       773.5       1156   ₹61,095       +7.3%     92d   
  TORNTPHARM  85     HEALTH     01-Jun-23   1,716.8     1,734.9     485    ₹8,775        +1.1%     92d   
  PFC         4      FIN SVC    01-Aug-23   185.9       184.8       5069   ₹-5,277       -0.6%     31d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IRFC        3      FIN SVC    3.604    0.87   +178.0%   +74.2%    52.8        18183  ₹960,416      +16.5%    
  BHEL        4      ENERGY     3.285    1.11   +137.8%   +64.3%    135.8       7075   ₹960,435      +24.2%    
  DRREDDY     8      HEALTH     2.102    0.39   +33.1%    +21.9%    1,102.2     871    ₹959,974      -2.4%     
  SUNTV       11     MEDIA      1.983    0.78   +26.6%    +37.7%    585.1       1641   ₹960,230      +8.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         12     DEFENCE    01-Apr-22   723.8       1,914.7     897    ₹1,068,243    +164.5%     +1.7%     
  RECLTD      1      FIN SVC    02-Jan-23   100.3       212.8       7763   ₹873,169      +112.2%     +4.9%     
  TVSMOTOR    27     AUTO       01-Aug-22   911.5       1,437.4     779    ₹409,659      +57.7%      +6.0%     
  ITC         36     FMCG       01-Aug-22   254.0       378.6       2795   ₹348,240      +49.1%      -1.6%     
  TRENT       13     CONSUMP    01-Jun-23   1,558.5     2,057.9     534    ₹266,703      +32.0%      +5.3%     
  BEL         20     DEFENCE    01-Jun-23   109.9       134.8       7581   ₹188,828      +22.7%      +4.5%     
  TATACOMM    8      CONSUMP    03-Jul-23   1,538.7     1,792.4     588    ₹149,181      +16.5%      +5.4%     
  POLYCAB     2      MFG        01-Aug-23   4,561.7     5,131.2     206    ₹117,316      +12.5%      +6.5%     
  LT          9      INFRA      03-Jul-23   2,362.2     2,630.6     383    ₹102,782      +11.4%      +1.5%     
  CHOLAFIN    49     FIN SVC    01-Jun-23   1,039.4     1,124.6     801    ₹68,300       +8.2%       +3.6%     
  NTPC        6      ENERGY     01-Aug-23   207.6       215.7       4537   ₹36,726       +3.9%       +6.2%     
  COLPAL      17     FMCG       01-Aug-23   1,880.1     1,817.3     501    ₹-31,431      -3.3%       -0.6%     
  HDFCAMC     30     FIN SVC    01-Aug-23   1,206.3     1,147.3     781    ₹-46,089      -4.9%       -2.3%     

  AFTER: Invested ₹18,313,648 | Cash ₹890,860 | Total ₹19,204,508 | Positions 17/20 | Slot ₹960,453

========================================================================
  REBALANCE #106  —  03 Oct 2023
  NAV: ₹20,386,809  |  Slot: ₹1,019,340  |  Cash: ₹890,860
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ITC         75     FMCG       01-Aug-22   254.0       377.5       2795   ₹345,121      +48.6%    428d  
  IRFC        2      FIN SVC    01-Sep-23   52.8        73.2        18183  ₹370,655      +38.6%    32d   
  BEL         46     DEFENCE    01-Jun-23   109.9       136.2       7581   ₹198,838      +23.9%    124d  
  BHEL        11     ENERGY     01-Sep-23   135.8       130.8       7075   ₹-35,271      -3.7%     32d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RVNL        3      PSE        3.481    1.14   +426.1%   +41.9%    170.4       5981   ₹1,019,179    +6.7%     
  INDIANB     5      PSU BNK    2.753    1.10   +144.4%   +48.2%    412.0       2474   ₹1,019,298    +7.3%     
  COALINDIA   7      ENERGY     2.353    0.77   +50.3%    +28.2%    242.6       4201   ₹1,019,136    +5.4%     
  OIL         9      OIL&GAS    2.070    0.42   +78.2%    +22.4%    180.7       5640   ₹1,019,279    +3.8%     
  PGHH        10     FMCG       1.897    0.31   +23.2%    +24.1%    16,905.8    60     ₹1,014,347    +2.8%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         48     DEFENCE    01-Apr-22   723.8       1,901.3     897    ₹1,056,195    +162.7%     +0.3%     
  RECLTD      1      FIN SVC    02-Jan-23   100.3       260.0       7763   ₹1,239,736    +159.3%     +11.0%    
  TVSMOTOR    25     AUTO       01-Aug-22   911.5       1,512.4     779    ₹468,105      +65.9%      +2.7%     
  TRENT       24     CONSUMP    01-Jun-23   1,558.5     2,053.9     534    ₹264,573      +31.8%      -0.3%     
  LT          7      INFRA      03-Jul-23   2,362.2     2,991.9     383    ₹241,169      +26.7%      +5.9%     
  CHOLAFIN    30     FIN SVC    01-Jun-23   1,039.4     1,249.1     801    ₹168,011      +20.2%      +6.0%     
  TATACOMM    21     CONSUMP    03-Jul-23   1,538.7     1,840.2     588    ₹177,260      +19.6%      +1.5%     
  POLYCAB     5      MFG        01-Aug-23   4,561.7     5,293.7     206    ₹150,803      +16.0%      +3.5%     
  NTPC        10     ENERGY     01-Aug-23   207.6       225.5       4537   ₹81,270       +8.6%       +2.0%     
  HDFCAMC     33     FIN SVC    01-Aug-23   1,206.3     1,255.8     781    ₹38,607       +4.1%       +1.6%     
  SUNTV       13     MEDIA      01-Sep-23   585.1       586.4       1641   ₹2,088        +0.2%       +4.2%     
  COLPAL      22     FMCG       01-Aug-23   1,880.1     1,853.9     501    ₹-13,122      -1.4%       -0.8%     
  DRREDDY     44     HEALTH     01-Sep-23   1,102.2     1,079.1     871    ₹-20,108      -2.1%       -2.2%     

  AFTER: Invested ₹20,243,792 | Cash ₹136,971 | Total ₹20,380,764 | Positions 18/20 | Slot ₹1,019,340

========================================================================
  REBALANCE #107  —  01 Nov 2023
  NAV: ₹19,913,724  |  Slot: ₹995,686  |  Cash: ₹136,971
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  RECLTD      1      FIN SVC    02-Jan-23   100.3       252.0       7763   ₹1,177,835    +151.3%   303d  
  HDFCAMC     36     FIN SVC    01-Aug-23   1,206.3     1,285.1     781    ₹61,476       +6.5%     92d   
  TATACOMM    64     CONSUMP    03-Jul-23   1,538.7     1,617.5     588    ₹46,342       +5.1%     121d  
  DRREDDY     72     HEALTH     01-Sep-23   1,102.2     1,056.3     871    ₹-39,940      -4.2%     61d   
  RVNL        6      PSE        03-Oct-23   170.4       151.1       5981   ₹-115,211     -11.3%    29d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  PERSISTENT  2      IT         3.045    1.20   +64.0%    +30.9%    3,057.8     325    ₹993,789      +5.3%     
  BAJAJ-AUTO  7      AUTO       2.384    0.61   +50.7%    +8.4%     5,098.5     195    ₹994,203      +2.3%     
  VBL         8      FMCG       2.327    0.62   +78.2%    +14.3%    364.8       2729   ₹995,629      +0.2%     
  HCLTECH     10     IT         2.180    1.02   +28.2%    +13.8%    1,140.6     872    ₹994,636      +0.8%     
  KPITTECH    15     IT         1.938    0.81   +66.3%    +12.4%    1,198.3     830    ₹994,611      +3.4%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         54     DEFENCE    01-Apr-22   723.8       1,770.8     897    ₹939,118      +144.6%     -3.6%     
  TVSMOTOR    21     AUTO       01-Aug-22   911.5       1,542.9     779    ₹491,861      +69.3%      -0.5%     
  TRENT       8      CONSUMP    01-Jun-23   1,558.5     2,192.1     534    ₹338,373      +40.7%      +5.4%     
  LT          9      INFRA      03-Jul-23   2,362.2     2,818.6     383    ₹174,818      +19.3%      -2.4%     
  CHOLAFIN    29     FIN SVC    01-Jun-23   1,039.4     1,144.7     801    ₹84,380       +10.1%      -2.8%     
  POLYCAB     23     MFG        01-Aug-23   4,561.7     4,828.1     206    ₹54,888       +5.8%       -4.1%     
  COALINDIA   5      ENERGY     03-Oct-23   242.6       254.9       4201   ₹51,498       +5.1%       +0.6%     
  NTPC        20     ENERGY     01-Aug-23   207.6       217.5       4537   ₹44,786       +4.8%       -1.7%     
  COLPAL      30     FMCG       01-Aug-23   1,880.1     1,959.4     501    ₹39,740       +4.2%       +1.6%     
  SUNTV       17     MEDIA      01-Sep-23   585.1       605.4       1641   ₹33,178       +3.5%       +1.8%     
  OIL         11     OIL&GAS    03-Oct-23   180.7       185.4       5640   ₹26,517       +2.6%       -1.1%     
  PGHH        16     FMCG       03-Oct-23   16,905.8    16,933.8    60     ₹1,680        +0.2%       +2.9%     
  INDIANB     10     PSU BNK    03-Oct-23   412.0       402.4       2474   ₹-23,790      -2.3%       +3.3%     

  AFTER: Invested ₹19,014,593 | Cash ₹893,226 | Total ₹19,907,819 | Positions 18/20 | Slot ₹995,686

========================================================================
  REBALANCE #108  —  01 Dec 2023
  NAV: ₹22,323,406  |  Slot: ₹1,116,170  |  Cash: ₹893,226
========================================================================

  EXITS (1)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  INDIANB     73     PSU BNK    03-Oct-23   412.0       374.5       2474   ₹-92,737      -9.1%     59d   

  ENTRIES (1)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TORNTPOWER  6      ENERGY     2.487    0.93   +86.0%    +43.1%    914.4       1220   ₹1,115,555    +14.4%    

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         13     DEFENCE    01-Apr-22   723.8       2,392.1     897    ₹1,496,446    +230.5%     +13.8%    
  TVSMOTOR    4      AUTO       01-Aug-22   911.5       1,887.8     779    ₹760,565      +107.1%     +9.7%     
  TRENT       5      CONSUMP    01-Jun-23   1,558.5     2,800.7     534    ₹663,323      +79.7%      +10.1%    
  LT          12     INFRA      03-Jul-23   2,362.2     3,106.2     383    ₹284,942      +31.5%      +4.3%     
  COALINDIA   3      ENERGY     03-Oct-23   242.6       301.3       4201   ₹246,663      +24.2%      +6.1%     
  NTPC        8      ENERGY     01-Aug-23   207.6       253.9       4537   ₹210,014      +22.3%      +7.3%     
  KPITTECH    21     IT         01-Nov-23   1,198.3     1,464.1     830    ₹220,579      +22.2%      +4.7%     
  VBL         22     FMCG       01-Nov-23   364.8       432.7       2729   ₹185,193      +18.6%      +5.8%     
  COLPAL      16     FMCG       01-Aug-23   1,880.1     2,158.3     501    ₹139,419      +14.8%      +5.5%     
  BAJAJ-AUTO  7      AUTO       01-Nov-23   5,098.5     5,767.9     195    ₹130,537      +13.1%      +5.8%     
  POLYCAB     23     MFG        01-Aug-23   4,561.7     5,157.1     206    ₹122,654      +13.1%      +0.5%     
  SUNTV       32     MEDIA      01-Sep-23   585.1       639.8       1641   ₹89,619       +9.3%       +2.0%     
  CHOLAFIN    61     FIN SVC    01-Jun-23   1,039.4     1,124.0     801    ₹67,781       +8.1%       -0.4%     
  OIL         28     OIL&GAS    03-Oct-23   180.7       192.7       5640   ₹67,309       +6.6%       +1.5%     
  HCLTECH     29     IT         01-Nov-23   1,140.6     1,211.1     872    ₹61,427       +6.2%       +2.5%     
  PERSISTENT  19     IT         01-Nov-23   3,057.8     3,167.3     325    ₹35,569       +3.6%       +1.8%     
  PGHH        49     FMCG       03-Oct-23   16,905.8    16,628.0    60     ₹-16,668      -1.6%       -1.2%     

  AFTER: Invested ₹21,619,174 | Cash ₹702,908 | Total ₹22,322,082 | Positions 18/20 | Slot ₹1,116,170

========================================================================
  REBALANCE #109  —  01 Jan 2024
  NAV: ₹24,313,989  |  Slot: ₹1,215,699  |  Cash: ₹702,908
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  CHOLAFIN    61     FIN SVC    01-Jun-23   1,039.4     1,220.8     801    ₹145,307      +17.5%    214d  
  SUNTV       38     MEDIA      01-Sep-23   585.1       674.5       1641   ₹146,572      +15.3%    122d  
  PGHH        104    FMCG       03-Oct-23   16,905.8    16,593.3    60     ₹-18,748      -1.8%     90d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HEROMOTOCO  7      AUTO       2.396    0.82   +56.7%    +37.0%    3,771.3     322    ₹1,214,355    +5.8%     
  FORTIS      11     HEALTH     2.033    0.35   +57.5%    +28.8%    438.2       2774   ₹1,215,510    +10.9%    
  TORNTPHARM  13     HEALTH     1.968    0.35   +46.7%    +24.3%    2,230.9     544    ₹1,213,598    +5.2%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         4      DEFENCE    01-Apr-22   723.8       2,745.8     897    ₹1,813,736    +279.4%     +5.3%     
  TVSMOTOR    7      AUTO       01-Aug-22   911.5       1,995.8     779    ₹844,691      +119.0%     +3.4%     
  TRENT       3      CONSUMP    01-Jun-23   1,558.5     2,994.6     534    ₹766,872      +92.1%      +2.9%     
  LT          15     INFRA      03-Jul-23   2,362.2     3,432.1     383    ₹409,776      +45.3%      +3.3%     
  NTPC        5      ENERGY     01-Aug-23   207.6       292.4       4537   ₹384,773      +40.8%      +4.8%     
  COALINDIA   8      ENERGY     03-Oct-23   242.6       331.9       4201   ₹375,014      +36.8%      +6.5%     
  VBL         31     FMCG       01-Nov-23   364.8       493.7       2729   ₹351,574      +35.3%      +6.0%     
  OIL         16     OIL&GAS    03-Oct-23   180.7       235.5       5640   ₹308,715      +30.3%      +8.2%     
  BAJAJ-AUTO  6      AUTO       01-Nov-23   5,098.5     6,392.8     195    ₹252,395      +25.4%      +5.2%     
  COLPAL      13     FMCG       01-Aug-23   1,880.1     2,354.2     501    ₹237,554      +25.2%      +4.1%     
  KPITTECH    30     IT         01-Nov-23   1,198.3     1,471.4     830    ₹226,683      +22.8%      +0.1%     
  POLYCAB     39     MFG        01-Aug-23   4,561.7     5,383.8     206    ₹169,363      +18.0%      +0.0%     
  HCLTECH     28     IT         01-Nov-23   1,140.6     1,344.3     872    ₹177,604      +17.9%      +4.1%     
  PERSISTENT  20     IT         01-Nov-23   3,057.8     3,602.8     325    ₹177,109      +17.8%      +4.3%     
  TORNTPOWER  32     ENERGY     01-Dec-23   914.4       904.8       1220   ₹-11,713      -1.0%       +3.4%     

  AFTER: Invested ₹24,174,307 | Cash ₹135,356 | Total ₹24,309,663 | Positions 18/20 | Slot ₹1,215,699

========================================================================
  REBALANCE #110  —  01 Feb 2024
  NAV: ₹25,397,900  |  Slot: ₹1,269,895  |  Cash: ₹135,356
========================================================================
  [SECTOR CAP≤4] dropped: TATAPOWER

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LT          51     INFRA      03-Jul-23   2,362.2     3,308.0     383    ₹362,255      +40.0%    213d  
  HCLTECH     41     IT         01-Nov-23   1,140.6     1,440.2     872    ₹261,206      +26.3%    92d   
  KPITTECH    47     IT         01-Nov-23   1,198.3     1,505.2     830    ₹254,665      +25.6%    92d   
  POLYCAB     106    MFG        01-Aug-23   4,561.7     4,200.9     206    ₹-74,331      -7.9%     184d  

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IRFC        1      FIN SVC    4.124    1.11   +423.4%   +136.1%   164.1       7737   ₹1,269,858    +18.3%    
  NHPC        5      ENERGY     2.833    0.83   +124.2%   +79.2%    85.8        14802  ₹1,269,866    +17.7%    
  RVNL        6      PSE        2.644    1.17   +297.9%   +90.3%    293.9       4320   ₹1,269,787    +16.8%    

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         6      DEFENCE    01-Apr-22   723.8       2,912.1     897    ₹1,962,927    +302.3%     +2.0%     
  TVSMOTOR    16     AUTO       01-Aug-22   911.5       1,972.3     779    ₹826,364      +116.4%     +0.1%     
  TRENT       4      CONSUMP    01-Jun-23   1,558.5     3,095.0     534    ₹820,511      +98.6%      -0.6%     
  OIL         21     OIL&GAS    03-Oct-23   180.7       271.3       5640   ₹510,675      +50.1%      +9.7%     
  NTPC        11     ENERGY     01-Aug-23   207.6       304.0       4537   ₹437,244      +46.4%      +3.3%     
  COALINDIA   15     ENERGY     03-Oct-23   242.6       353.5       4201   ₹465,937      +45.7%      +4.9%     
  BAJAJ-AUTO  5      AUTO       01-Nov-23   5,098.5     7,303.5     195    ₹429,987      +43.2%      +5.9%     
  VBL         19     FMCG       01-Nov-23   364.8       510.3       2729   ₹397,113      +39.9%      +2.4%     
  PERSISTENT  20     IT         01-Nov-23   3,057.8     4,099.4     325    ₹338,505      +34.1%      +5.1%     
  COLPAL      24     FMCG       01-Aug-23   1,880.1     2,370.6     501    ₹245,761      +26.1%      +0.8%     
  HEROMOTOCO  13     AUTO       01-Jan-24   3,771.3     4,200.0     322    ₹138,034      +11.4%      +5.2%     
  TORNTPOWER  17     ENERGY     01-Dec-23   914.4       1,009.7     1220   ₹116,246      +10.4%      +5.3%     
  TORNTPHARM  25     HEALTH     01-Jan-24   2,230.9     2,440.9     544    ₹114,262      +9.4%       +3.4%     
  FORTIS      39     HEALTH     01-Jan-24   438.2       425.6       2774   ₹-34,978      -2.9%       +1.2%     

  AFTER: Invested ₹24,434,586 | Cash ₹958,791 | Total ₹25,393,377 | Positions 17/20 | Slot ₹1,269,895

========================================================================
  REBALANCE #111  —  01 Mar 2024
  NAV: ₹26,974,641  |  Slot: ₹1,348,732  |  Cash: ₹958,791
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TORNTPOWER  46     ENERGY     01-Dec-23   914.4       1,082.4     1220   ₹204,963      +18.4%    91d   
  HEROMOTOCO  38     AUTO       01-Jan-24   3,771.3     4,217.5     322    ₹143,685      +11.8%    60d   
  FORTIS      79     HEALTH     01-Jan-24   438.2       398.3       2774   ₹-110,601     -9.1%     60d   
  IRFC        1      FIN SVC    01-Feb-24   164.1       142.4       7737   ₹-168,168     -13.2%    29d   
  RVNL        25     PSE        01-Feb-24   293.9       243.2       4320   ₹-219,135     -17.3%    29d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADANIGREEN  3      ENERGY     2.870    1.03   +247.1%   +91.4%    1,969.6     684    ₹1,347,172    +5.0%     
  BEL         5      DEFENCE    2.591    1.11   +118.4%   +41.5%    201.9       6681   ₹1,348,630    +5.8%     
  SUNPHARMA   8      HEALTH     2.364    0.37   +62.0%    +27.9%    1,530.0     881    ₹1,347,932    +2.6%     
  BOSCHLTD    11     AUTO       2.276    0.65   +61.9%    +35.0%    28,414.1    47     ₹1,335,462    +6.2%     
  DIXON       12     CON DUR    2.206    0.66   +157.4%   +27.1%    6,999.3     192    ₹1,343,872    +6.3%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         13     DEFENCE    01-Apr-22   723.8       3,086.9     897    ₹2,119,684    +326.5%     +4.5%     
  TRENT       5      CONSUMP    01-Jun-23   1,558.5     3,891.5     534    ₹1,245,814    +149.7%     +3.3%     
  TVSMOTOR    28     AUTO       01-Aug-22   911.5       2,216.8     779    ₹1,016,833    +143.2%     +6.5%     
  OIL         2      OIL&GAS    03-Oct-23   180.7       366.3       5640   ₹1,046,430    +102.7%     +9.2%     
  COALINDIA   18     ENERGY     03-Oct-23   242.6       392.4       4201   ₹629,495      +61.8%      +2.1%     
  NTPC        11     ENERGY     01-Aug-23   207.6       327.5       4537   ₹543,702      +57.7%      +3.3%     
  VBL         29     FMCG       01-Nov-23   364.8       563.3       2729   ₹541,700      +54.4%      +0.4%     
  BAJAJ-AUTO  8      AUTO       01-Nov-23   5,098.5     7,670.2     195    ₹501,478      +50.4%      -0.2%     
  PERSISTENT  33     IT         01-Nov-23   3,057.8     4,249.2     325    ₹387,207      +39.0%      +1.0%     
  COLPAL      36     FMCG       01-Aug-23   1,880.1     2,400.2     501    ₹260,587      +27.7%      +0.1%     
  TORNTPHARM  24     HEALTH     01-Jan-24   2,230.9     2,614.3     544    ₹208,561      +17.2%      +2.6%     
  NHPC        16     ENERGY     01-Feb-24   85.8        85.6        14802  ₹-2,382       -0.2%       +0.4%     

  AFTER: Invested ₹26,803,107 | Cash ₹163,551 | Total ₹26,966,658 | Positions 17/20 | Slot ₹1,348,732

========================================================================
  REBALANCE #112  —  01 Apr 2024
  NAV: ₹27,508,392  |  Slot: ₹1,375,420  |  Cash: ₹163,551
========================================================================
  [SECTOR CAP≤4] dropped: HEROMOTOCO

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COALINDIA   25     ENERGY     03-Oct-23   242.6       388.7       4201   ₹613,612      +60.2%    181d  
  NTPC        34     ENERGY     01-Aug-23   207.6       326.4       4537   ₹538,728      +57.2%    244d  
  BEL         24     DEFENCE    01-Mar-24   201.9       208.0       6681   ₹41,019       +3.0%     31d   
  NHPC        28     ENERGY     01-Feb-24   85.8        86.2        14802  ₹6,787        +0.5%     60d   
  ADANIGREEN  57     ENERGY     01-Mar-24   1,969.6     1,888.3     684    ₹-55,575      -4.1%     31d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  CUMMINSIND  5      INFRA      2.732    0.77   +84.2%    +52.5%    2,927.3     469    ₹1,372,919    +6.4%     
  TORNTPOWER  6      ENERGY     2.727    0.67   +172.7%   +59.9%    1,379.4     997    ₹1,375,215    +13.6%    
  KALYANKJIL  9      CON DUR    2.226    0.53   +260.5%   +20.9%    423.7       3246   ₹1,375,355    +8.3%     
  UNIONBANK   11     PSU BNK    2.116    1.16   +155.2%   +31.7%    143.5       9585   ₹1,375,319    +4.9%     
  SIEMENS     12     ENERGY     2.115    0.74   +66.6%    +37.9%    3,195.5     430    ₹1,374,050    +11.4%    

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         14     DEFENCE    01-Apr-22   723.8       3,330.6     897    ₹2,338,307    +360.2%     +6.8%     
  TRENT       7      CONSUMP    01-Jun-23   1,558.5     3,877.1     534    ₹1,238,117    +148.8%     -0.9%     
  TVSMOTOR    37     AUTO       01-Aug-22   911.5       2,122.8     779    ₹943,647      +132.9%     +1.1%     
  OIL         10     OIL&GAS    03-Oct-23   180.7       373.1       5640   ₹1,085,109    +106.5%     +2.3%     
  BAJAJ-AUTO  1      AUTO       01-Nov-23   5,098.5     8,626.2     195    ₹687,898      +69.2%      +4.3%     
  VBL         39     FMCG       01-Nov-23   364.8       555.1       2729   ₹519,310      +52.2%      -0.7%     
  COLPAL      36     FMCG       01-Aug-23   1,880.1     2,572.1     501    ₹346,719      +36.8%      +2.5%     
  PERSISTENT  55     IT         01-Nov-23   3,057.8     3,949.9     325    ₹289,932      +29.2%      -2.3%     
  TORNTPHARM  30     HEALTH     01-Jan-24   2,230.9     2,620.8     544    ₹212,132      +17.5%      +2.8%     
  DIXON       19     CON DUR    01-Mar-24   6,999.3     7,586.0     192    ₹112,633      +8.4%       +7.3%     
  BOSCHLTD    5      AUTO       01-Mar-24   28,414.1    29,732.3    47     ₹61,956       +4.6%       +2.7%     
  SUNPHARMA   3      HEALTH     01-Mar-24   1,530.0     1,598.7     881    ₹60,513       +4.5%       +3.3%     

  AFTER: Invested ₹27,146,337 | Cash ₹353,894 | Total ₹27,500,231 | Positions 17/20 | Slot ₹1,375,420

========================================================================
  REBALANCE #113  —  02 May 2024
  NAV: ₹28,805,557  |  Slot: ₹1,440,278  |  Cash: ₹353,894
========================================================================
  [SECTOR CAP≤4] dropped: M&M

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PERSISTENT  107    IT         01-Nov-23   3,057.8     3,367.8     325    ₹100,762      +10.1%    183d  
  UNIONBANK   46     PSU BNK    01-Apr-24   143.5       141.6       9585   ₹-18,425      -1.3%     31d   

  ENTRIES (1)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SBIN        11     PSU BNK    2.231    1.16   +55.8%    +35.5%    786.1       1832   ₹1,440,173    +5.9%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         3      DEFENCE    01-Apr-22   723.8       3,862.8     897    ₹2,815,714    +433.7%     +5.5%     
  TRENT       1      CONSUMP    01-Jun-23   1,558.5     4,635.0     534    ₹1,642,886    +197.4%     +11.0%    
  TVSMOTOR    40     AUTO       01-Aug-22   911.5       2,057.2     779    ₹892,507      +125.7%     +2.0%     
  OIL         6      OIL&GAS    03-Oct-23   180.7       398.0       5640   ₹1,225,579    +120.2%     +3.1%     
  BAJAJ-AUTO  15     AUTO       01-Nov-23   5,098.5     8,691.5     195    ₹700,641      +70.5%      +2.6%     
  VBL         23     FMCG       01-Nov-23   364.8       603.2       2729   ₹650,382      +65.3%      +5.0%     
  COLPAL      30     FMCG       01-Aug-23   1,880.1     2,662.6     501    ₹392,074      +41.6%      +2.7%     
  DIXON       5      CON DUR    01-Mar-24   6,999.3     8,403.6     192    ₹269,611      +20.1%      +6.4%     
  TORNTPHARM  50     HEALTH     01-Jan-24   2,230.9     2,615.6     544    ₹209,281      +17.2%      +1.7%     
  CUMMINSIND  2      INFRA      01-Apr-24   2,927.3     3,220.5     469    ₹137,502      +10.0%      +5.7%     
  SIEMENS     10     ENERGY     01-Apr-24   3,195.5     3,434.5     430    ₹102,773      +7.5%       +4.9%     
  BOSCHLTD    8      AUTO       01-Mar-24   28,414.1    30,113.4    47     ₹79,868       +6.0%       +3.5%     
  TORNTPOWER  7      ENERGY     01-Apr-24   1,379.4     1,460.4     997    ₹80,773       +5.9%       +1.9%     
  SUNPHARMA   36     HEALTH     01-Mar-24   1,530.0     1,490.5     881    ₹-34,795      -2.6%       -1.1%     
  KALYANKJIL  16     CON DUR    01-Apr-24   423.7       409.7       3246   ₹-45,533      -3.3%       -0.3%     

  AFTER: Invested ₹27,440,392 | Cash ₹1,363,455 | Total ₹28,803,847 | Positions 16/20 | Slot ₹1,440,278

========================================================================
  REBALANCE #114  —  03 Jun 2024
  NAV: ₹31,032,890  |  Slot: ₹1,551,645  |  Cash: ₹1,363,455
========================================================================
  [SECTOR CAP≤4] dropped: MOTHERSON

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HAL         1      DEFENCE    01-Apr-22   723.8       5,160.9     897    ₹3,980,111    +613.0%   794d  
  OIL         —      OIL&GAS    03-Oct-23   180.7       422.6       5640   ₹1,364,091    +133.8%   244d  
  VBL         85     FMCG       01-Nov-23   364.8       582.2       2729   ₹593,139      +59.6%    215d  
  TORNTPHARM  77     HEALTH     01-Jan-24   2,230.9     2,623.4     544    ₹213,517      +17.6%    154d  
  SBIN        37     PSU BNK    02-May-24   786.1       872.1       1832   ₹157,520      +10.9%    32d   
  TORNTPOWER  —      ENERGY     01-Apr-24   1,379.4     1,469.3     997    ₹89,667       +6.5%     63d   
  SUNPHARMA   98     HEALTH     01-Mar-24   1,530.0     1,425.8     881    ₹-91,807      -6.8%     94d   

  ENTRIES (10)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BSE         3      FIN SVC    2.845    0.70   +415.1%   +18.8%    894.9       1733   ₹1,550,797    -0.2%     
  MAZDOCK     4      DEFENCE    2.815    1.11   +327.5%   +55.7%    1,602.9     968    ₹1,551,604    +14.1%    
  BDL         6      DEFENCE    2.702    1.16   +198.4%   +70.1%    1,583.6     979    ₹1,550,315    +22.1%    
  PRESTIGE    8      REALTY     2.584    0.94   +250.6%   +41.9%    1,731.5     896    ₹1,551,464    +12.4%    
  CGPOWER     9      ENERGY     2.375    0.47   +92.6%    +60.2%    683.4       2270   ₹1,551,270    +10.2%    
  ASHOKLEY    10     AUTO       2.269    0.93   +62.4%    +39.8%    112.4       13798  ₹1,551,537    +11.7%    
  ESCORTS     11     MFG        2.253    0.74   +89.4%    +34.6%    3,820.6     406    ₹1,551,149    +5.0%     
  BHARATFORG  12     DEFENCE    2.192    1.11   +111.0%   +36.8%    1,589.1     976    ₹1,550,953    +8.8%     
  HAVELLS     13     CON DUR    2.041    0.57   +51.7%    +32.4%    1,852.4     837    ₹1,550,479    +4.5%     
  HINDALCO    14     METAL      2.032    1.06   +70.7%    +37.8%    686.9       2258   ₹1,551,123    +4.2%     

  HOLDS (9)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       11     CONSUMP    01-Jun-23   1,558.5     4,652.6     534    ₹1,652,278    +198.5%     +2.2%     
  TVSMOTOR    58     AUTO       01-Aug-22   911.5       2,232.9     779    ₹1,029,381    +145.0%     +3.9%     
  BAJAJ-AUTO  31     AUTO       01-Nov-23   5,098.5     8,906.0     195    ₹742,460      +74.7%      +3.9%     
  DIXON       6      CON DUR    01-Mar-24   6,999.3     9,878.1     192    ₹552,718      +41.1%      +10.6%    
  COLPAL      55     FMCG       01-Aug-23   1,880.1     2,578.9     501    ₹350,090      +37.2%      -0.3%     
  SIEMENS     3      ENERGY     01-Apr-24   3,195.5     4,254.8     430    ₹455,535      +33.2%      +6.5%     
  CUMMINSIND  13     INFRA      01-Apr-24   2,927.3     3,618.2     469    ₹324,031      +23.6%      +2.9%     
  BOSCHLTD    60     AUTO       01-Mar-24   28,414.1    29,444.7    47     ₹48,438       +3.6%       -2.0%     
  KALYANKJIL  32     CON DUR    01-Apr-24   423.7       388.9       3246   ₹-112,864     -8.2%       -2.3%     

  AFTER: Invested ₹30,832,811 | Cash ₹181,662 | Total ₹31,014,473 | Positions 19/20 | Slot ₹1,551,645

========================================================================
  REBALANCE #115  —  01 Jul 2024
  NAV: ₹33,768,732  |  Slot: ₹1,688,437  |  Cash: ₹181,662
========================================================================
  [SECTOR CAP≤4] dropped: M&M, HEROMOTOCO

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DIXON       4      CON DUR    01-Mar-24   6,999.3     12,436.5    192    ₹1,043,930    +77.7%    122d  
  SIEMENS     7      ENERGY     01-Apr-24   3,195.5     4,604.4     430    ₹605,833      +44.1%    91d   
  MAZDOCK     2      DEFENCE    03-Jun-24   1,602.9     2,162.0     968    ₹541,223      +34.9%    28d   
  PRESTIGE    10     REALTY     03-Jun-24   1,731.5     1,836.5     896    ₹94,061       +6.1%     28d   
  BHARATFORG  17     DEFENCE    03-Jun-24   1,589.1     1,646.1     976    ₹55,644       +3.6%     28d   
  ASHOKLEY    22     AUTO       03-Jun-24   112.4       113.5       13798  ₹14,958       +1.0%     28d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MOTHERSON   1      AUTO       3.699    0.88   +144.0%   +67.8%    129.4       13049  ₹1,688,362    +11.3%    
  VOLTAS      10     CON DUR    1.949    0.94   +83.4%    +31.6%    1,431.7     1179   ₹1,688,016    -1.0%     
  SUPREMEIND  11     MFG        1.935    0.73   +101.6%   +51.6%    5,861.9     288    ₹1,688,225    +2.3%     
  SUNTV       12     MEDIA      1.919    0.50   +80.5%    +33.6%    745.5       2264   ₹1,687,708    +4.7%     
  OFSS        14     IT         1.785    0.95   +172.1%   +20.0%    9,455.9     178    ₹1,683,150    +10.4%    
  BALKRISIND  17     MFG        1.699    0.85   +36.9%    +40.8%    3,152.9     535    ₹1,686,823    +1.0%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       3      CONSUMP    01-Jun-23   1,558.5     5,504.3     534    ₹2,107,085    +253.2%     +6.6%     
  TVSMOTOR    43     AUTO       01-Aug-22   911.5       2,336.7     779    ₹1,110,246    +156.4%     -0.5%     
  BAJAJ-AUTO  29     AUTO       01-Nov-23   5,098.5     9,167.8     195    ₹793,510      +79.8%      +0.2%     
  COLPAL      56     FMCG       01-Aug-23   1,880.1     2,746.1     501    ₹433,895      +46.1%      +0.6%     
  CUMMINSIND  16     INFRA      01-Apr-24   2,927.3     3,884.9     469    ₹449,076      +32.7%      +4.0%     
  BOSCHLTD    33     AUTO       01-Mar-24   28,414.1    33,627.9    47     ₹245,049      +18.3%      +4.7%     
  KALYANKJIL  11     CON DUR    01-Apr-24   423.7       494.9       3246   ₹231,217      +16.8%      +13.2%    
  ESCORTS     9      MFG        03-Jun-24   3,820.6     4,067.5     406    ₹100,237      +6.5%       +0.8%     
  CGPOWER     35     ENERGY     03-Jun-24   683.4       718.6       2270   ₹79,909       +5.2%       +6.5%     
  BDL         8      DEFENCE    03-Jun-24   1,583.6     1,602.2     979    ₹18,256       +1.2%       +6.9%     
  HINDALCO    49     METAL      03-Jun-24   686.9       680.9       2258   ₹-13,596      -0.9%       +1.5%     
  HAVELLS     51     CON DUR    03-Jun-24   1,852.4     1,797.8     837    ₹-45,707      -2.9%       -1.1%     
  BSE         14     FIN SVC    03-Jun-24   894.9       855.3       1733   ₹-68,500      -4.4%       -1.7%     

  AFTER: Invested ₹32,430,226 | Cash ₹1,326,487 | Total ₹33,756,713 | Positions 19/20 | Slot ₹1,688,437

========================================================================
  REBALANCE #116  —  01 Aug 2024
  NAV: ₹34,741,681  |  Slot: ₹1,737,084  |  Cash: ₹1,326,487
========================================================================
  [SECTOR CAP≤4] dropped: M&M

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VOLTAS      70     CON DUR    01-Jul-24   1,431.7     1,523.3     1179   ₹107,996      +6.4%     31d   
  CGPOWER     50     ENERGY     03-Jun-24   683.4       725.6       2270   ₹95,732       +6.2%     59d   
  HAVELLS     87     CON DUR    03-Jun-24   1,852.4     1,811.8     837    ₹-34,032      -2.2%     59d   
  HINDALCO    105    METAL      03-Jun-24   686.9       664.8       2258   ₹-50,036      -3.2%     59d   
  SUPREMEIND  118    MFG        01-Jul-24   5,861.9     5,152.7     288    ₹-204,239     -12.1%    31d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ZYDUSLIFE   5      HEALTH     2.445    0.63   +103.4%   +30.5%    1,227.3     1415   ₹1,736,687    +5.2%     
  PERSISTENT  6      IT         2.316    0.84   +91.3%    +42.7%    4,750.7     365    ₹1,734,011    +2.8%     
  LUPIN       8      HEALTH     2.075    0.39   +107.4%   +19.2%    1,941.3     894    ₹1,735,501    +7.9%     
  INFY        14     IT         1.872    0.77   +32.1%    +33.0%    1,741.4     997    ₹1,736,185    +4.6%     
  TORNTPHARM  15     HEALTH     1.781    0.34   +67.8%    +21.5%    3,146.3     552    ₹1,736,750    +5.1%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       2      CONSUMP    01-Jun-23   1,558.5     5,759.0     534    ₹2,243,055    +269.5%     +5.3%     
  TVSMOTOR    11     AUTO       01-Aug-22   911.5       2,564.6     779    ₹1,287,745    +181.4%     +5.0%     
  BAJAJ-AUTO  25     AUTO       01-Nov-23   5,098.5     9,358.3     195    ₹830,662      +83.6%      +2.2%     
  COLPAL      15     FMCG       01-Aug-23   1,880.1     3,238.2     501    ₹680,430      +72.2%      +7.2%     
  KALYANKJIL  7      CON DUR    01-Apr-24   423.7       561.8       3246   ₹448,226      +32.6%      +5.2%     
  CUMMINSIND  43     INFRA      01-Apr-24   2,927.3     3,738.1     469    ₹380,235      +27.7%      +0.9%     
  BOSCHLTD    19     AUTO       01-Mar-24   28,414.1    33,910.4    47     ₹258,329      +19.3%      -0.3%     
  SUNTV       16     MEDIA      01-Jul-24   745.5       852.7       2264   ₹242,805      +14.4%      +8.2%     
  ESCORTS     26     MFG        03-Jun-24   3,820.6     4,093.5     406    ₹110,813      +7.1%       +1.4%     
  OFSS        8      IT         01-Jul-24   9,455.9     10,120.6    178    ₹118,325      +7.0%       +1.7%     
  BALKRISIND  31     MFG        01-Jul-24   3,152.9     3,310.3     535    ₹84,162       +5.0%       +4.1%     
  MOTHERSON   4      AUTO       01-Jul-24   129.4       129.1       13049  ₹-3,862       -0.2%       +1.1%     
  BSE         48     FIN SVC    03-Jun-24   894.9       878.3       1733   ₹-28,680      -1.8%       +8.3%     
  BDL         18     DEFENCE    03-Jun-24   1,583.6     1,438.5     979    ₹-142,019     -9.2%       -3.8%     

  AFTER: Invested ₹34,149,797 | Cash ₹581,579 | Total ₹34,731,375 | Positions 19/20 | Slot ₹1,737,084

========================================================================
  REBALANCE #117  —  02 Sep 2024
  NAV: ₹35,967,674  |  Slot: ₹1,798,384  |  Cash: ₹581,579
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  CUMMINSIND  36     INFRA      01-Apr-24   2,927.3     3,727.7     469    ₹375,363      +27.3%    154d  
  SUNTV       61     MEDIA      01-Jul-24   745.5       781.2       2264   ₹81,025       +4.8%     63d   
  ESCORTS     125    MFG        03-Jun-24   3,820.6     3,732.5     406    ₹-35,740      -2.3%     91d   
  BALKRISIND  141    MFG        01-Jul-24   3,152.9     2,868.7     535    ₹-152,047     -9.0%     63d   
  ZYDUSLIFE   59     HEALTH     01-Aug-24   1,227.3     1,099.0     1415   ₹-181,657     -10.5%    32d   
  BDL         100    DEFENCE    03-Jun-24   1,583.6     1,293.3     979    ₹-284,184     -18.3%    91d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ICICIGI     7      FIN SVC    2.321    0.63   +65.6%    +38.8%    2,156.1     834    ₹1,798,166    +5.9%     
  HCLTECH     10     IT         2.192    0.72   +59.2%    +37.5%    1,684.5     1067   ₹1,797,313    +7.5%     
  SUNPHARMA   12     HEALTH     2.157    0.47   +61.1%    +24.8%    1,787.5     1006   ₹1,798,263    +3.2%     
  VOLTAS      13     CON DUR    2.011    0.89   +119.8%   +30.6%    1,754.3     1025   ₹1,798,173    +7.0%     
  SBILIFE     15     FIN SVC    1.936    0.80   +48.0%    +36.2%    1,882.5     955    ₹1,797,831    +6.4%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       1      CONSUMP    01-Jun-23   1,558.5     7,133.5     534    ₹2,977,078    +357.7%     +7.8%     
  TVSMOTOR    8      AUTO       01-Aug-22   911.5       2,769.5     779    ₹1,447,386    +203.8%     +4.5%     
  BAJAJ-AUTO  3      AUTO       01-Nov-23   5,098.5     10,700.5    195    ₹1,092,394    +109.9%     +8.8%     
  COLPAL      7      FMCG       01-Aug-23   1,880.1     3,483.3     501    ₹803,204      +85.3%      +3.4%     
  KALYANKJIL  2      CON DUR    01-Apr-24   423.7       638.0       3246   ₹695,538      +50.6%      +9.6%     
  LUPIN       4      HEALTH     01-Aug-24   1,941.3     2,218.9     894    ₹248,240      +14.3%      +6.4%     
  BOSCHLTD    38     AUTO       01-Mar-24   28,414.1    31,894.7    47     ₹163,589      +12.2%      -0.4%     
  PERSISTENT  6      IT         01-Aug-24   4,750.7     5,157.2     365    ₹148,350      +8.6%       +6.1%     
  OFSS        10     IT         01-Jul-24   9,455.9     10,145.5    178    ₹122,758      +7.3%       +0.9%     
  TORNTPHARM  16     HEALTH     01-Aug-24   3,146.3     3,366.9     552    ₹121,789      +7.0%       +2.7%     
  INFY        14     IT         01-Aug-24   1,741.4     1,846.6     997    ₹104,868      +6.0%       +5.0%     
  BSE         39     FIN SVC    03-Jun-24   894.9       918.8       1733   ₹41,487       +2.7%       +3.5%     
  MOTHERSON   20     AUTO       01-Jul-24   129.4       127.6       13049  ₹-22,918      -1.4%       +0.5%     

  AFTER: Invested ₹34,987,479 | Cash ₹969,520 | Total ₹35,956,999 | Positions 18/20 | Slot ₹1,798,384

========================================================================
  REBALANCE #118  —  01 Oct 2024
  NAV: ₹38,103,352  |  Slot: ₹1,905,168  |  Cash: ₹969,520
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       1      CONSUMP    01-Jun-23   1,558.5     7,597.1     534    ₹3,224,612    +387.5%     +2.9%     
  TVSMOTOR    11     AUTO       01-Aug-22   911.5       2,817.1     779    ₹1,484,455    +209.1%     +0.7%     
  BAJAJ-AUTO  3      AUTO       01-Nov-23   5,098.5     11,692.4    195    ₹1,285,814    +129.3%     +2.8%     
  COLPAL      4      FMCG       01-Aug-23   1,880.1     3,666.2     501    ₹894,846      +95.0%      +3.9%     
  KALYANKJIL  2      CON DUR    01-Apr-24   423.7       748.0       3246   ₹1,052,527    +76.5%      +6.8%     
  BSE         7      FIN SVC    03-Jun-24   894.9       1,282.3     1733   ₹671,352      +43.3%      +11.1%    
  BOSCHLTD    15     AUTO       01-Mar-24   28,414.1    37,334.5    47     ₹419,258      +31.4%      +6.5%     
  PERSISTENT  16     IT         01-Aug-24   4,750.7     5,435.4     365    ₹249,919      +14.4%      +3.4%     
  LUPIN       6      HEALTH     01-Aug-24   1,941.3     2,180.8     894    ₹214,166      +12.3%      -0.1%     
  OFSS        20     IT         01-Jul-24   9,455.9     10,613.9    178    ₹206,121      +12.2%      +0.6%     
  MOTHERSON   34     AUTO       01-Jul-24   129.4       139.2       13049  ₹128,180      +7.6%       +4.1%     
  SUNPHARMA   5      HEALTH     02-Sep-24   1,787.5     1,889.9     1006   ₹102,987      +5.7%       +2.9%     
  TORNTPHARM  10     HEALTH     01-Aug-24   3,146.3     3,308.2     552    ₹89,357       +5.1%       -1.3%     
  VOLTAS      9      CON DUR    02-Sep-24   1,754.3     1,838.5     1025   ₹86,268       +4.8%       +0.4%     
  INFY        37     IT         01-Aug-24   1,741.4     1,790.1     997    ₹48,498       +2.8%       +0.1%     
  HCLTECH     18     IT         02-Sep-24   1,684.5     1,693.6     1067   ₹9,799        +0.5%       +2.4%     
  ICICIGI     21     FIN SVC    02-Sep-24   2,156.1     2,124.8     834    ₹-26,121      -1.5%       -1.3%     
  SBILIFE     26     FIN SVC    02-Sep-24   1,882.5     1,828.2     955    ₹-51,924      -2.9%       -1.1%     

  AFTER: Invested ₹37,133,832 | Cash ₹969,520 | Total ₹38,103,352 | Positions 18/20 | Slot ₹1,905,168

========================================================================
  REBALANCE #119  —  01 Nov 2024
  NAV: ₹35,483,852  |  Slot: ₹1,774,193  |  Cash: ₹969,520
========================================================================
  [SECTOR CAP≤4] dropped: ALKEM

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COLPAL      82     FMCG       01-Aug-23   1,880.1     2,942.4     501    ₹532,245      +56.5%    458d  
  INFY        75     IT         01-Aug-24   1,741.4     1,674.0     997    ₹-67,195      -3.9%     92d   
  ICICIGI     68     FIN SVC    02-Sep-24   2,156.1     1,895.8     834    ₹-217,091     -12.1%    60d   
  SBILIFE     109    FIN SVC    02-Sep-24   1,882.5     1,623.5     955    ₹-247,389     -13.8%    60d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MANKIND     4      HEALTH     2.541    0.42   +53.2%    +34.0%    2,679.6     662    ₹1,773,924    +3.3%     
  COFORGE     10     IT         2.104    0.81   +56.2%    +24.5%    1,493.2     1188   ₹1,773,970    +2.0%     
  INDHOTEL    14     CONSUMP    1.859    1.18   +74.2%    +9.4%     682.5       2599   ₹1,773,933    +0.8%     
  ICICIBANK   16     PVT BNK    1.791    0.99   +40.1%    +8.9%     1,281.9     1384   ₹1,774,162    +1.4%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       1      CONSUMP    01-Jun-23   1,558.5     7,134.3     534    ₹2,977,504    +357.8%     -4.5%     
  TVSMOTOR    47     AUTO       01-Aug-22   911.5       2,490.6     779    ₹1,230,150    +173.2%     -3.7%     
  BAJAJ-AUTO  17     AUTO       01-Nov-23   5,098.5     9,498.2     195    ₹857,940      +86.3%      -6.8%     
  BSE         2      FIN SVC    03-Jun-24   894.9       1,483.2     1733   ₹1,019,598    +65.7%      +5.7%     
  KALYANKJIL  4      CON DUR    01-Apr-24   423.7       668.8       3246   ₹795,709      +57.9%      -3.1%     
  BOSCHLTD    12     AUTO       01-Mar-24   28,414.1    34,711.0    47     ₹295,957      +22.2%      -3.2%     
  LUPIN       7      HEALTH     01-Aug-24   1,941.3     2,184.1     894    ₹217,099      +12.5%      +0.8%     
  PERSISTENT  9      IT         01-Aug-24   4,750.7     5,337.6     365    ₹214,218      +12.4%      -1.7%     
  OFSS        8      IT         01-Jul-24   9,455.9     10,038.5    178    ₹103,704      +6.2%       -2.9%     
  SUNPHARMA   6      HEALTH     02-Sep-24   1,787.5     1,829.3     1006   ₹42,036       +2.3%       -0.9%     
  TORNTPHARM  22     HEALTH     01-Aug-24   3,146.3     3,146.6     552    ₹163          +0.0%       -3.9%     
  HCLTECH     14     IT         02-Sep-24   1,684.5     1,649.3     1067   ₹-37,541      -2.1%       -3.3%     
  MOTHERSON   31     AUTO       01-Jul-24   129.4       120.4       13049  ₹-117,646     -7.0%       -6.5%     
  VOLTAS      10     CON DUR    02-Sep-24   1,754.3     1,628.6     1025   ₹-128,844     -7.2%       -7.0%     

  AFTER: Invested ₹35,335,655 | Cash ₹139,771 | Total ₹35,475,426 | Positions 18/20 | Slot ₹1,774,193

========================================================================
  REBALANCE #120  —  02 Dec 2024
  NAV: ₹36,287,519  |  Slot: ₹1,814,376  |  Cash: ₹139,771
========================================================================
  [SECTOR CAP≤4] dropped: WIPRO

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TVSMOTOR    79     AUTO       01-Aug-22   911.5       2,474.5     779    ₹1,217,549    +171.5%   854d  
  BAJAJ-AUTO  74     AUTO       01-Nov-23   5,098.5     8,781.1     195    ₹718,109      +72.2%    397d  

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  CGPOWER     10     ENERGY     2.015    0.82   +93.6%    +8.5%     752.0       2412   ₹1,813,921    +2.9%     
  OBEROIRLTY  13     REALTY     1.931    1.19   +48.0%    +16.9%    2,054.7     883    ₹1,814,289    +4.6%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       8      CONSUMP    01-Jun-23   1,558.5     6,791.3     534    ₹2,794,345    +335.8%     +0.3%     
  KALYANKJIL  6      CON DUR    01-Apr-24   423.7       720.0       3246   ₹961,743      +69.9%      +3.3%     
  BSE         2      FIN SVC    03-Jun-24   894.9       1,516.5     1733   ₹1,077,298    +69.5%      +0.2%     
  PERSISTENT  9      IT         01-Aug-24   4,750.7     5,875.8     365    ₹410,668      +23.7%      +3.1%     
  BOSCHLTD    10     AUTO       01-Mar-24   28,414.1    34,460.4    47     ₹284,178      +21.3%      +0.1%     
  OFSS        3      IT         01-Jul-24   9,455.9     11,378.1    178    ₹342,145      +20.3%      +5.8%     
  INDHOTEL    5      CONSUMP    01-Nov-24   682.5       795.2       2599   ₹292,688      +16.5%      +6.1%     
  COFORGE     4      IT         01-Nov-24   1,493.2     1,722.2     1188   ₹271,990      +15.3%      +5.8%     
  LUPIN       35     HEALTH     01-Aug-24   1,941.3     2,056.8     894    ₹103,241      +5.9%       -0.3%     
  HCLTECH     15     IT         02-Sep-24   1,684.5     1,756.4     1067   ₹76,713       +4.3%       +1.0%     
  TORNTPHARM  31     HEALTH     01-Aug-24   3,146.3     3,278.0     552    ₹72,694       +4.2%       +3.5%     
  ICICIBANK   17     PVT BNK    01-Nov-24   1,281.9     1,294.7     1384   ₹17,648       +1.0%       +1.8%     
  SUNPHARMA   19     HEALTH     02-Sep-24   1,787.5     1,780.3     1006   ₹-7,328       -0.4%       +0.8%     
  MANKIND     38     HEALTH     01-Nov-24   2,679.6     2,615.4     662    ₹-42,517      -2.4%       +0.8%     
  VOLTAS      12     CON DUR    02-Sep-24   1,754.3     1,706.2     1025   ₹-49,333      -2.7%       +1.4%     
  MOTHERSON   50     AUTO       01-Jul-24   129.4       109.5       13049  ₹-259,694     -15.4%      -3.0%     

  AFTER: Invested ₹36,136,045 | Cash ₹147,165 | Total ₹36,283,210 | Positions 18/20 | Slot ₹1,814,376

========================================================================
  REBALANCE #121  —  01 Jan 2025
  NAV: ₹38,434,333  |  Slot: ₹1,921,717  |  Cash: ₹147,165
========================================================================

  [REGIME OFF] Nifty 200 13,462.5 < SMA200 13,468.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       10     CONSUMP    01-Jun-23   1,558.5     7,053.5     534    ₹2,934,366    +352.6%     +1.1%     
  BSE         5      FIN SVC    03-Jun-24   894.9       1,803.0     1733   ₹1,573,767    +101.5%     +1.7%     
  KALYANKJIL  7      CON DUR    01-Apr-24   423.7       773.6       3246   ₹1,135,706    +82.6%      +4.9%     
  PERSISTENT  8      IT         01-Aug-24   4,750.7     6,375.4     365    ₹593,019      +34.2%      +1.4%     
  COFORGE     4      IT         01-Nov-24   1,493.2     1,903.7     1188   ₹487,573      +27.5%      +3.7%     
  INDHOTEL    2      CONSUMP    01-Nov-24   682.5       867.2       2599   ₹479,860      +27.1%      +2.7%     
  OFSS        3      IT         01-Jul-24   9,455.9     11,715.6    178    ₹402,228      +23.9%      +1.8%     
  LUPIN       6      HEALTH     01-Aug-24   1,941.3     2,350.3     894    ₹365,651      +21.1%      +8.1%     
  BOSCHLTD    34     AUTO       01-Mar-24   28,414.1    33,576.8    47     ₹242,649      +18.2%      -2.3%     
  OBEROIRLTY  9      REALTY     02-Dec-24   2,054.7     2,258.5     883    ₹179,969      +9.9%       +2.3%     
  MANKIND     16     HEALTH     01-Nov-24   2,679.6     2,879.9     662    ₹132,580      +7.5%       +3.3%     
  TORNTPHARM  19     HEALTH     01-Aug-24   3,146.3     3,355.9     552    ₹115,720      +6.7%       +1.4%     
  HCLTECH     18     IT         02-Sep-24   1,684.5     1,794.3     1067   ₹117,217      +6.5%       +0.0%     
  SUNPHARMA   17     HEALTH     02-Sep-24   1,787.5     1,860.4     1006   ₹73,279       +4.1%       +3.1%     
  VOLTAS      14     CON DUR    02-Sep-24   1,754.3     1,811.1     1025   ₹58,223       +3.2%       +4.8%     
  ICICIBANK   31     PVT BNK    01-Nov-24   1,281.9     1,273.8     1384   ₹-11,193      -0.6%       -1.4%     
  CGPOWER     29     ENERGY     02-Dec-24   752.0       737.9       2412   ₹-34,107      -1.9%       -0.8%     
  MOTHERSON   76     AUTO       01-Jul-24   129.4       102.6       13049  ₹-348,905     -20.7% ⚠    -3.2%     
  ⚠  WAZ < 0 (momentum below universe mean): MOTHERSON

  AFTER: Invested ₹38,287,167 | Cash ₹147,165 | Total ₹38,434,333 | Positions 18/20 | Slot ₹1,921,717

========================================================================
  REBALANCE #122  —  01 Feb 2025
  NAV: ₹33,160,134  |  Slot: ₹1,658,007  |  Cash: ₹147,165
========================================================================
  [SECTOR CAP≤4] dropped: BAJFINANCE

  [REGIME OFF] Nifty 200 13,064.5 < SMA200 13,551.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       20     CONSUMP    01-Jun-23   1,558.5     6,176.8     534    ₹2,466,182    +296.3%     +2.9%     
  BSE         3      FIN SVC    03-Jun-24   894.9       1,795.7     1733   ₹1,561,128    +100.7%     -1.5%     
  PERSISTENT  15     IT         01-Aug-24   4,750.7     5,896.4     365    ₹418,191      +24.1%      -2.5%     
  KALYANKJIL  87     CON DUR    01-Apr-24   423.7       504.0       3246   ₹260,549      +18.9% ⚠    -5.2%     
  INDHOTEL    5      CONSUMP    01-Nov-24   682.5       795.6       2599   ₹293,721      +16.6%      +1.3%     
  COFORGE     22     IT         01-Nov-24   1,493.2     1,600.2     1188   ₹127,124      +7.2%       -7.2%     
  LUPIN       30     HEALTH     01-Aug-24   1,941.3     2,043.5     894    ₹91,424       +5.3%       -3.1%     
  TORNTPHARM  21     HEALTH     01-Aug-24   3,146.3     3,173.5     552    ₹15,026       +0.9%       -1.6%     
  BOSCHLTD    105    AUTO       01-Mar-24   28,414.1    28,305.1    47     ₹-5,123       -0.4% ⚠     -6.3%     
  ICICIBANK   33     PVT BNK    01-Nov-24   1,281.9     1,245.9     1384   ₹-49,786      -2.8%       +0.8%     
  SUNPHARMA   39     HEALTH     02-Sep-24   1,787.5     1,714.9     1006   ₹-73,032      -4.1%       -1.9%     
  HCLTECH     50     IT         02-Sep-24   1,684.5     1,605.9     1067   ₹-83,818      -4.7%       -5.1%     
  MANKIND     61     HEALTH     01-Nov-24   2,679.6     2,480.6     662    ₹-131,752     -7.4% ⚠     -3.8%     
  OBEROIRLTY  36     REALTY     02-Dec-24   2,054.7     1,832.9     883    ₹-195,813     -10.8%      -2.8%     
  OFSS        76     IT         01-Jul-24   9,455.9     8,249.1     178    ₹-214,805     -12.8% ⚠    -11.8%    
  CGPOWER     74     ENERGY     02-Dec-24   752.0       609.4       2412   ₹-343,953     -19.0% ⚠    -4.9%     
  VOLTAS      88     CON DUR    02-Sep-24   1,754.3     1,312.5     1025   ₹-452,884     -25.2% ⚠    -11.6%    
  MOTHERSON   113    AUTO       01-Jul-24   129.4       94.2        13049  ₹-458,975     -27.2%      -2.0%     
  ⚠  WAZ < 0 (momentum below universe mean): MANKIND, CGPOWER, OFSS, KALYANKJIL, VOLTAS, BOSCHLTD

  AFTER: Invested ₹33,012,969 | Cash ₹147,165 | Total ₹33,160,134 | Positions 18/20 | Slot ₹1,658,007

========================================================================
  REBALANCE #123  —  03 Mar 2025
  NAV: ₹29,644,347  |  Slot: ₹1,482,217  |  Cash: ₹147,165
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJHLDNG

  [REGIME OFF] Nifty 200 12,134.7 < SMA200 13,570.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       57     CONSUMP    01-Jun-23   1,558.5     4,936.8     534    ₹1,803,996    +216.8% ⚠   -4.9%     
  BSE         7      FIN SVC    03-Jun-24   894.9       1,448.6     1733   ₹959,594      +61.9%      -17.7%    
  PERSISTENT  28     IT         01-Aug-24   4,750.7     5,259.5     365    ₹185,689      +10.7%      -7.3%     
  INDHOTEL    24     CONSUMP    01-Nov-24   682.5       721.4       2599   ₹101,003      +5.7%       -2.7%     
  KALYANKJIL  93     CON DUR    01-Apr-24   423.7       439.2       3246   ₹50,174       +3.6% ⚠     -10.8%    
  LUPIN       20     HEALTH     01-Aug-24   1,941.3     1,940.9     894    ₹-355         -0.0%       -2.2%     
  COFORGE     45     IT         01-Nov-24   1,493.2     1,457.7     1188   ₹-42,192      -2.4%       -6.4%     
  ICICIBANK   29     PVT BNK    01-Nov-24   1,281.9     1,197.0     1384   ₹-117,563     -6.6%       -2.5%     
  TORNTPHARM  42     HEALTH     01-Aug-24   3,146.3     2,927.7     552    ₹-120,657     -6.9%       -3.7%     
  BOSCHLTD    —      AUTO       01-Mar-24   28,414.1    26,334.1    47     ₹-97,758      -7.3%       -3.4%     
  HCLTECH     94     IT         02-Sep-24   1,684.5     1,490.6     1067   ₹-206,817     -11.5% ⚠    -6.4%     
  SUNPHARMA   67     HEALTH     02-Sep-24   1,787.5     1,569.7     1006   ₹-219,142     -12.2% ⚠    -5.4%     
  MANKIND     36     HEALTH     01-Nov-24   2,679.6     2,327.8     662    ₹-232,932     -13.1%      -2.8%     
  CGPOWER     —      ENERGY     02-Dec-24   752.0       584.2       2412   ₹-404,721     -22.3%      -1.4%     
  VOLTAS      41     CON DUR    02-Sep-24   1,754.3     1,354.3     1025   ₹-410,004     -22.8%      +2.9%     
  OFSS        118    IT         01-Jul-24   9,455.9     7,269.2     178    ₹-389,233     -23.1% ⚠    -10.1%    
  OBEROIRLTY  71     REALTY     02-Dec-24   2,054.7     1,497.1     883    ₹-492,347     -27.1%      -7.9%     
  MOTHERSON   98     AUTO       01-Jul-24   129.4       78.9        13049  ₹-659,118     -39.0%      -7.5%     
  ⚠  WAZ < 0 (momentum below universe mean): TRENT, SUNPHARMA, KALYANKJIL, HCLTECH, OFSS

  AFTER: Invested ₹29,497,181 | Cash ₹147,165 | Total ₹29,644,347 | Positions 18/20 | Slot ₹1,482,217

========================================================================
  REBALANCE #124  —  01 Apr 2025
  NAV: ₹31,618,216  |  Slot: ₹1,580,911  |  Cash: ₹147,165
========================================================================
  [SECTOR CAP≤4] dropped: CHOLAFIN, HDFCLIFE, BAJAJHLDNG

  [REGIME OFF] Nifty 200 12,809.3 < SMA200 13,548.6 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       54     CONSUMP    01-Jun-23   1,558.5     5,565.3     534    ₹2,139,646    +257.1%     +6.8%     
  BSE         6      FIN SVC    03-Jun-24   894.9       1,816.3     1733   ₹1,596,802    +103.0%     +15.8%    
  INDHOTEL    29     CONSUMP    01-Nov-24   682.5       799.8       2599   ₹304,814      +17.2%      +2.7%     
  PERSISTENT  60     IT         01-Aug-24   4,750.7     5,179.0     365    ₹156,309      +9.0% ⚠     -3.6%     
  KALYANKJIL  114    CON DUR    01-Apr-24   423.7       456.7       3246   ₹107,137      +7.8% ⚠     -1.1%     
  COFORGE     50     IT         01-Nov-24   1,493.2     1,541.7     1188   ₹57,555       +3.2%       +0.1%     
  ICICIBANK   16     PVT BNK    01-Nov-24   1,281.9     1,308.4     1384   ₹36,601       +2.1%       +1.7%     
  TORNTPHARM  31     HEALTH     01-Aug-24   3,146.3     3,150.0     552    ₹2,028        +0.1%       +0.8%     
  LUPIN       72     HEALTH     01-Aug-24   1,941.3     1,943.3     894    ₹1,821        +0.1% ⚠     -3.4%     
  BOSCHLTD    —      AUTO       01-Mar-24   28,414.1    27,510.7    47     ₹-42,459      -3.2%       +1.1%     
  SUNPHARMA   86     HEALTH     02-Sep-24   1,787.5     1,681.9     1006   ₹-106,319     -5.9% ⚠     -0.7%     
  MANKIND     81     HEALTH     01-Nov-24   2,679.6     2,457.1     662    ₹-147,303     -8.3% ⚠     +4.7%     
  HCLTECH     125    IT         02-Sep-24   1,684.5     1,450.8     1067   ₹-249,351     -13.9% ⚠    -3.9%     
  CGPOWER     —      ENERGY     02-Dec-24   752.0       613.6       2412   ₹-333,808     -18.4%      -1.2%     
  VOLTAS      82     CON DUR    02-Sep-24   1,754.3     1,340.3     1025   ₹-424,331     -23.6% ⚠    -4.3%     
  OBEROIRLTY  121    REALTY     02-Dec-24   2,054.7     1,564.3     883    ₹-433,022     -23.9%      -2.1%     
  OFSS        131    IT         01-Jul-24   9,455.9     7,036.0     178    ₹-430,750     -25.6% ⚠    -3.4%     
  MOTHERSON   77     AUTO       01-Jul-24   129.4       86.9        13049  ₹-553,884     -32.8%      +1.8%     
  ⚠  WAZ < 0 (momentum below universe mean): PERSISTENT, LUPIN, VOLTAS, MANKIND, SUNPHARMA, KALYANKJIL, HCLTECH, OFSS

  AFTER: Invested ₹31,471,051 | Cash ₹147,165 | Total ₹31,618,216 | Positions 18/20 | Slot ₹1,580,911

========================================================================
  REBALANCE #125  —  02 May 2025
  NAV: ₹32,563,526  |  Slot: ₹1,628,176  |  Cash: ₹147,165
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJFINSV

  [REGIME OFF] Nifty 200 13,423.5 < SMA200 13,503.7 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       68     CONSUMP    01-Jun-23   1,558.5     5,143.4     534    ₹1,914,360    +230.0%     -0.7%     
  BSE         16     FIN SVC    03-Jun-24   894.9       2,099.0     1733   ₹2,086,850    +134.6%     +5.7%     
  KALYANKJIL  35     CON DUR    01-Apr-24   423.7       505.8       3246   ₹266,374      +19.4%      +0.1%     
  INDHOTEL    33     CONSUMP    01-Nov-24   682.5       795.5       2599   ₹293,592      +16.6%      -0.2%     
  PERSISTENT  42     IT         01-Aug-24   4,750.7     5,372.7     365    ₹227,021      +13.1%      +5.4%     
  ICICIBANK   4      PVT BNK    01-Nov-24   1,281.9     1,419.1     1384   ₹189,941      +10.7%      +3.6%     
  LUPIN       39     HEALTH     01-Aug-24   1,941.3     2,046.3     894    ₹93,867       +5.4%       +1.1%     
  BOSCHLTD    —      AUTO       01-Mar-24   28,414.1    29,050.7    47     ₹29,921       +2.2%       +4.8%     
  TORNTPHARM  53     HEALTH     01-Aug-24   3,146.3     3,198.1     552    ₹28,624       +1.6%       -0.6%     
  SUNPHARMA   26     HEALTH     02-Sep-24   1,787.5     1,806.9     1006   ₹19,455       +1.1%       +3.4%     
  COFORGE     70     IT         01-Nov-24   1,493.2     1,461.4     1188   ₹-37,849      -2.1% ⚠     +2.6%     
  HCLTECH     89     IT         02-Sep-24   1,684.5     1,507.8     1067   ₹-188,467     -10.5% ⚠    +3.6%     
  MANKIND     93     HEALTH     01-Nov-24   2,679.6     2,379.6     662    ₹-198,654     -11.2% ⚠    -3.8%     
  OFSS        73     IT         01-Jul-24   9,455.9     8,047.6     178    ₹-250,675     -14.9% ⚠    +5.0%     
  CGPOWER     —      ENERGY     02-Dec-24   752.0       619.2       2412   ₹-320,331     -17.7%      -0.1%     
  OBEROIRLTY  82     REALTY     02-Dec-24   2,054.7     1,593.9     883    ₹-406,914     -22.4%      -1.2%     
  VOLTAS      129    CON DUR    02-Sep-24   1,754.3     1,193.2     1025   ₹-575,174     -32.0% ⚠    -6.8%     
  MOTHERSON   87     AUTO       01-Jul-24   129.4       87.6        13049  ₹-545,145     -32.3%      +1.3%     
  ⚠  WAZ < 0 (momentum below universe mean): COFORGE, OFSS, HCLTECH, MANKIND, VOLTAS

  AFTER: Invested ₹32,416,361 | Cash ₹147,165 | Total ₹32,563,526 | Positions 18/20 | Slot ₹1,628,176

========================================================================
  REBALANCE #126  —  02 Jun 2025
  NAV: ₹34,678,160  |  Slot: ₹1,733,908  |  Cash: ₹147,165
========================================================================

  EXITS (12)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TRENT       65     CONSUMP    01-Jun-23   1,558.5     5,610.0     534    ₹2,163,493    +260.0%   732d  
  BSE         2      FIN SVC    03-Jun-24   894.9       2,693.3     1733   ₹3,116,692    +201.0%   364d  
  COFORGE     —      IT         01-Nov-24   1,493.2     1,705.3     1188   ₹251,937      +14.2%    213d  
  INDHOTEL    42     CONSUMP    01-Nov-24   682.5       777.8       2599   ₹247,669      +14.0%    213d  
  BOSCHLTD    —      AUTO       01-Mar-24   28,414.1    30,811.8    47     ₹112,693      +8.4%     458d  
  SUNPHARMA   70     HEALTH     02-Sep-24   1,787.5     1,658.3     1006   ₹-129,979     -7.2%     273d  
  CGPOWER     —      ENERGY     02-Dec-24   752.0       677.6       2412   ₹-179,539     -9.9%     182d  
  MANKIND     75     HEALTH     01-Nov-24   2,679.6     2,413.3     662    ₹-176,287     -9.9%     213d  
  OBEROIRLTY  80     REALTY     02-Dec-24   2,054.7     1,759.4     883    ₹-260,703     -14.4%    182d  
  OFSS        82     IT         01-Jul-24   9,455.9     8,038.0     178    ₹-252,381     -15.0%    336d  
  MOTHERSON   53     AUTO       01-Jul-24   129.4       100.1       13049  ₹-382,397     -22.6%    336d  
  VOLTAS      125    CON DUR    02-Sep-24   1,754.3     1,235.2     1025   ₹-532,090     -29.6%    273d  

  ENTRIES (13)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SOLARINDS   1      DEFENCE    4.247    1.12   +68.3%    +83.8%    16,284.3    106    ₹1,726,131    +10.5%    
  BHARTIHEXA  2      IT         2.926    1.13   +80.6%    +46.6%    1,840.5     942    ₹1,733,714    +7.6%     
  HDFCLIFE    3      FIN SVC    2.572    0.72   +36.3%    +24.3%    761.9       2275   ₹1,733,257    +1.4%     
  DIVISLAB    5      HEALTH     2.298    0.58   +54.6%    +14.7%    6,509.4     266    ₹1,731,490    +2.0%     
  BHARTIARTL  6      CONSUMP    2.159    0.94   +34.7%    +15.8%    1,838.7     942    ₹1,732,082    +0.7%     
  SBILIFE     7      FIN SVC    2.100    0.79   +28.1%    +21.5%    1,800.0     963    ₹1,733,398    +2.0%     
  HDFCBANK    8      PVT BNK    2.081    0.88   +26.5%    +15.2%    937.7       1849   ₹1,733,729    +0.5%     
  UNITDSPR    9      FMCG       2.021    0.57   +34.7%    +15.7%    1,533.0     1131   ₹1,733,847    +0.6%     
  PAGEIND     10     MFG        1.635    0.56   +28.4%    +12.0%    45,270.3    38     ₹1,720,271    -1.0%     
  BAJFINANCE  11     FIN SVC    1.633    1.06   +33.7%    +9.8%     906.3       1913   ₹1,733,711    +0.4%     
  MAXHEALTH   12     HEALTH     1.623    0.78   +43.4%    +16.6%    1,150.3     1507   ₹1,733,461    +0.4%     
  TVSMOTOR    13     AUTO       1.575    1.07   +23.4%    +17.5%    2,753.8     629    ₹1,732,124    +0.1%     
  FEDERALBNK  14     PVT BNK    1.560    0.78   +26.8%    +13.8%    205.0       8457   ₹1,733,761    +3.3%     

  HOLDS (6)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  KALYANKJIL  40     CON DUR    01-Apr-24   423.7       555.0       3246   ₹426,259      +31.0%      +1.4%     
  PERSISTENT  56     IT         01-Aug-24   4,750.7     5,484.5     365    ₹267,826      +15.4%      -0.9%     
  ICICIBANK   7      PVT BNK    01-Nov-24   1,281.9     1,439.4     1384   ₹217,959      +12.3%      +0.9%     
  LUPIN       69     HEALTH     01-Aug-24   1,941.3     1,949.0     894    ₹6,886        +0.4% ⚠     -1.8%     
  TORNTPHARM  59     HEALTH     01-Aug-24   3,146.3     3,094.5     552    ₹-28,609      -1.6% ⚠     -2.5%     
  HCLTECH     67     IT         02-Sep-24   1,684.5     1,564.5     1067   ₹-127,998     -7.1% ⚠     +0.2%     
  ⚠  WAZ < 0 (momentum below universe mean): TORNTPHARM, HCLTECH, LUPIN

  AFTER: Invested ₹33,426,391 | Cash ₹1,225,039 | Total ₹34,651,430 | Positions 19/20 | Slot ₹1,733,908

========================================================================
  REBALANCE #127  —  01 Jul 2025
  NAV: ₹36,165,702  |  Slot: ₹1,808,285  |  Cash: ₹1,225,039
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  KALYANKJIL  41     CON DUR    01-Apr-24   423.7       568.7       3246   ₹470,599      +34.2%    456d  
  PERSISTENT  29     IT         01-Aug-24   4,750.7     6,000.7     365    ₹456,258      +26.3%    334d  
  TVSMOTOR    19     AUTO       02-Jun-25   2,753.8     2,883.5     629    ₹81,611       +4.7%     29d   
  LUPIN       72     HEALTH     01-Aug-24   1,941.3     1,948.8     894    ₹6,708        +0.4%     334d  
  HCLTECH     46     IT         02-Sep-24   1,684.5     1,647.7     1067   ₹-39,186      -2.2%     302d  
  UNITDSPR    87     FMCG       02-Jun-25   1,533.0     1,385.8     1131   ₹-166,535     -9.6%     29d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MUTHOOTFIN  7      FIN SVC    2.136    0.71   +50.6%    +14.4%    2,619.7     690    ₹1,807,627    +4.5%     
  APOLLOHOSP  8      HEALTH     2.046    0.71   +22.5%    +15.5%    7,476.5     241    ₹1,801,834    +5.7%     
  BRITANNIA   10     FMCG       1.854    0.46   +7.3%     +18.6%    5,669.2     318    ₹1,802,807    +1.6%     
  TECHM       11     IT         1.826    1.04   +21.5%    +17.4%    1,581.3     1143   ₹1,807,405    +0.8%     
  INDUSTOWER  12     INFRA      1.756    1.00   +18.0%    +24.2%    420.4       4301   ₹1,807,925    +4.4%     
  ETERNAL     13     CONSUMP    1.735    1.16   +32.3%    +26.5%    261.0       6928   ₹1,808,208    +2.9%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MAXHEALTH   22     HEALTH     02-Jun-25   1,150.3     1,278.2     1507   ₹192,824      +11.1%      +5.0%     
  ICICIBANK   31     PVT BNK    01-Nov-24   1,281.9     1,421.0     1384   ₹192,551      +10.9%      -0.1%     
  BHARTIARTL  4      CONSUMP    02-Jun-25   1,838.7     2,002.7     942    ₹154,422      +8.9%       +4.7%     
  TORNTPHARM  37     HEALTH     01-Aug-24   3,146.3     3,386.7     552    ₹132,709      +7.6%       +5.1%     
  BHARTIHEXA  6      IT         02-Jun-25   1,840.5     1,973.7     942    ₹125,536      +7.2%       +8.0%     
  FEDERALBNK  33     PVT BNK    02-Jun-25   205.0       217.4       8457   ₹104,824      +6.0%       +4.9%     
  HDFCLIFE    5      FIN SVC    02-Jun-25   761.9       807.0       2275   ₹102,722      +5.9%       +3.9%     
  SOLARINDS   2      DEFENCE    02-Jun-25   16,284.3    17,185.7    106    ₹95,555       +5.5%       +2.2%     
  HDFCBANK    15     PVT BNK    02-Jun-25   937.7       987.2       1849   ₹91,649       +5.3%       +3.1%     
  PAGEIND     27     MFG        02-Jun-25   45,270.3    47,518.3    38     ₹85,423       +5.0%       +2.5%     
  DIVISLAB    7      HEALTH     02-Jun-25   6,509.4     6,825.4     266    ₹84,072       +4.9%       +3.5%     
  SBILIFE     9      FIN SVC    02-Jun-25   1,800.0     1,859.9     963    ₹57,700       +3.3%       +2.6%     
  BAJFINANCE  34     FIN SVC    02-Jun-25   906.3       930.9       1913   ₹47,153       +2.7%       +1.0%     

  AFTER: Invested ₹34,858,863 | Cash ₹1,293,972 | Total ₹36,152,835 | Positions 19/20 | Slot ₹1,808,285

========================================================================
  REBALANCE #128  —  01 Aug 2025
  NAV: ₹34,720,230  |  Slot: ₹1,736,011  |  Cash: ₹1,293,972
========================================================================
  [SECTOR CAP≤4] dropped: MANKIND

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TECHM       83     IT         01-Jul-25   1,581.3     1,386.3     1143   ₹-222,909     -12.3%    31d   
  INDUSTOWER  123    INFRA      01-Jul-25   420.4       345.1       4301   ₹-323,650     -17.9%    31d   

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  INDIANB     11     PSU BNK    1.808    1.03   +6.0%     +13.5%    608.2       2854   ₹1,735,925    -1.4%     
  POLYCAB     16     MFG        1.608    1.10   +0.8%     +14.0%    6,666.4     260    ₹1,733,256    -1.3%     

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ETERNAL     5      CONSUMP    01-Jul-25   261.0       304.8       6928   ₹303,100      +16.8%      +5.8%     
  TORNTPHARM  10     HEALTH     01-Aug-24   3,146.3     3,645.9     552    ₹275,789      +15.9%      +3.8%     
  ICICIBANK   13     PVT BNK    01-Nov-24   1,281.9     1,460.3     1384   ₹246,938      +13.9%      +0.7%     
  MAXHEALTH   14     HEALTH     02-Jun-25   1,150.3     1,246.0     1507   ₹144,261      +8.3%       -0.4%     
  HDFCBANK    9      PVT BNK    02-Jun-25   937.7       989.7       1849   ₹96,282       +5.6%       +0.7%     
  BHARTIARTL  16     CONSUMP    02-Jun-25   1,838.7     1,884.4     942    ₹43,023       +2.5%       -2.0%     
  PAGEIND     36     MFG        02-Jun-25   45,270.3    46,152.7    38     ₹33,532       +1.9%       -1.3%     
  BRITANNIA   28     FMCG       01-Jul-25   5,669.2     5,723.0     318    ₹17,107       +0.9%       +1.3%     
  BHARTIHEXA  11     IT         02-Jun-25   1,840.5     1,844.6     942    ₹3,900        +0.2%       +2.0%     
  SBILIFE     42     FIN SVC    02-Jun-25   1,800.0     1,793.7     963    ₹-6,059       -0.3%       -1.3%     
  MUTHOOTFIN  3      FIN SVC    01-Jul-25   2,619.7     2,571.0     690    ₹-33,663      -1.9%       -1.5%     
  APOLLOHOSP  26     HEALTH     01-Jul-25   7,476.5     7,332.4     241    ₹-34,734      -1.9%       -0.1%     
  DIVISLAB    15     HEALTH     02-Jun-25   6,509.4     6,361.5     266    ₹-39,331      -2.3%       -4.2%     
  HDFCLIFE    43     FIN SVC    02-Jun-25   761.9       739.1       2275   ₹-51,882      -3.0%       -2.5%     
  BAJFINANCE  23     FIN SVC    02-Jun-25   906.3       870.3       1913   ₹-68,923      -4.0%       -4.2%     
  FEDERALBNK  53     PVT BNK    02-Jun-25   205.0       194.9       8457   ₹-85,658      -4.9%       -5.7%     
  SOLARINDS   22     DEFENCE    02-Jun-25   16,284.3    13,807.0    106    ₹-262,589     -15.2%      -8.1%     

  AFTER: Invested ₹33,826,668 | Cash ₹889,443 | Total ₹34,716,110 | Positions 19/20 | Slot ₹1,736,011

========================================================================
  REBALANCE #129  —  01 Sep 2025
  NAV: ₹34,906,863  |  Slot: ₹1,745,343  |  Cash: ₹889,443
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ETERNAL     —      CONSUMP    01-Jul-25   261.0       321.1       6928   ₹416,373      +23.0%    62d   
  SBILIFE     53     FIN SVC    02-Jun-25   1,800.0     1,807.3     963    ₹7,020        +0.4%     91d   
  PAGEIND     48     MFG        02-Jun-25   45,270.3    44,341.6    38     ₹-35,292      -2.1%     91d   
  FEDERALBNK  —      PVT BNK    02-Jun-25   205.0       193.6       8457   ₹-96,655      -5.6%     91d   
  DIVISLAB    37     HEALTH     02-Jun-25   6,509.4     6,093.0     266    ₹-110,752     -6.4%     91d   
  SOLARINDS   46     DEFENCE    02-Jun-25   16,284.3    14,046.0    106    ₹-237,255     -13.7%    91d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BOSCHLTD    1      AUTO       3.558    0.94   +28.0%    +32.4%    40,785.0    42     ₹1,712,970    +4.0%     
  EICHERMOT   2      AUTO       2.937    1.11   +30.7%    +18.7%    6,280.0     277    ₹1,739,560    +6.6%     
  HEROMOTOCO  4      AUTO       2.402    1.04   +1.9%     +25.8%    5,143.7     339    ₹1,743,728    +7.6%     
  NYKAA       7      CONSUMP    2.107    0.55   +3.6%     +20.1%    233.7       7467   ₹1,745,262    +5.2%     
  COFORGE     9      IT         2.004    1.12   +46.5%    +3.0%     1,756.5     993    ₹1,744,203    +2.5%     
  HINDUNILVR  13     FMCG       1.826    0.48   -2.1%     +12.9%    2,602.4     670    ₹1,743,584    +2.4%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TORNTPHARM  10     HEALTH     01-Aug-24   3,146.3     3,552.8     552    ₹224,412      +12.9%      -0.6%     
  ICICIBANK   29     PVT BNK    01-Nov-24   1,281.9     1,411.0     1384   ₹178,662      +10.1%      -1.1%     
  INDIANB     19     PSU BNK    01-Aug-25   608.2       654.6       2854   ₹132,178      +7.6%       +1.6%     
  POLYCAB     17     MFG        01-Aug-25   6,666.4     7,110.8     260    ₹115,542      +6.7%       +2.0%     
  BHARTIARTL  13     CONSUMP    02-Jun-25   1,838.7     1,900.6     942    ₹58,283       +3.4%       -0.2%     
  BRITANNIA   30     FMCG       01-Jul-25   5,669.2     5,846.5     318    ₹56,380       +3.1%       +4.0%     
  MAXHEALTH   18     HEALTH     02-Jun-25   1,150.3     1,181.2     1507   ₹46,608       +2.7%       -3.1%     
  MUTHOOTFIN  3      FIN SVC    01-Jul-25   2,619.7     2,687.3     690    ₹46,595       +2.6%       +1.8%     
  APOLLOHOSP  11     HEALTH     01-Jul-25   7,476.5     7,661.3     241    ₹44,546       +2.5%       +0.5%     
  HDFCLIFE    36     FIN SVC    02-Jun-25   761.9       779.0       2275   ₹39,021       +2.3%       +0.7%     
  HDFCBANK    25     PVT BNK    02-Jun-25   937.7       935.1       1849   ₹-4,668       -0.3%       -3.1%     
  BAJFINANCE  24     FIN SVC    02-Jun-25   906.3       883.9       1913   ₹-42,779      -2.5%       +0.1%     
  BHARTIHEXA  23     IT         02-Jun-25   1,840.5     1,770.0     942    ₹-66,374      -3.8%       -1.1%     

  AFTER: Invested ₹34,050,029 | Cash ₹844,450 | Total ₹34,894,479 | Positions 19/20 | Slot ₹1,745,343

========================================================================
  REBALANCE #130  —  01 Oct 2025
  NAV: ₹35,277,230  |  Slot: ₹1,763,862  |  Cash: ₹844,450
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ICICIBANK   55     PVT BNK    01-Nov-24   1,281.9     1,372.0     1384   ₹124,686      +7.0%     334d  
  BHARTIARTL  66     CONSUMP    02-Jun-25   1,838.7     1,867.6     942    ₹27,197       +1.6%     121d  
  MAXHEALTH   69     HEALTH     02-Jun-25   1,150.3     1,113.2     1507   ₹-55,868      -3.2%     121d  
  COFORGE     67     IT         01-Sep-25   1,756.5     1,593.5     993    ₹-161,907     -9.3%     30d   
  BHARTIHEXA  63     IT         02-Jun-25   1,840.5     1,660.8     942    ₹-169,240     -9.8%     121d  

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  FORTIS      2      HEALTH     3.545    0.73   +62.4%    +25.0%    989.7       1782   ₹1,763,556    +3.4%     
  GODFRYPHLP  9      FMCG       1.937    1.15   +43.6%    +15.1%    3,357.6     525    ₹1,762,720    -1.7%     
  COROMANDEL  10     MFG        1.875    0.92   +38.7%    -0.6%     2,242.2     786    ₹1,762,339    -0.5%     
  SBIN        11     PSU BNK    1.863    0.97   +9.9%     +6.3%     848.8       2078   ₹1,763,807    +1.9%     
  BANKINDIA   12     PSU BNK    1.676    1.12   +16.9%    +4.5%     120.7       14615  ₹1,763,850    +4.9%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  3      FIN SVC    01-Jul-25   2,619.7     3,118.1     690    ₹343,886      +19.0%      +5.7%     
  INDIANB     5      PSU BNK    01-Aug-25   608.2       721.7       2854   ₹323,739      +18.6%      +4.8%     
  TORNTPHARM  25     HEALTH     01-Aug-24   3,146.3     3,535.8     552    ₹215,001      +12.4%      -0.6%     
  EICHERMOT   1      AUTO       01-Sep-25   6,280.0     7,021.5     277    ₹205,396      +11.8%      +3.0%     
  POLYCAB     27     MFG        01-Aug-25   6,666.4     7,316.3     260    ₹168,978      +9.7%       +0.2%     
  BAJFINANCE  10     FIN SVC    02-Jun-25   906.3       981.7       1913   ₹144,215      +8.3%       +0.7%     
  BRITANNIA   39     FMCG       01-Jul-25   5,669.2     5,966.5     318    ₹94,540       +5.2%       -0.3%     
  HEROMOTOCO  7      AUTO       01-Sep-25   5,143.7     5,328.6     339    ₹62,680       +3.6%       +2.3%     
  NYKAA       9      CONSUMP    01-Sep-25   233.7       241.3       7467   ₹56,227       +3.2%       +2.1%     
  HDFCBANK    37     PVT BNK    02-Jun-25   937.7       949.5       1849   ₹21,980       +1.3%       +0.4%     
  HDFCLIFE    52     FIN SVC    02-Jun-25   761.9       761.3       2275   ₹-1,330       -0.1%       -0.7%     
  APOLLOHOSP  42     HEALTH     01-Jul-25   7,476.5     7,431.1     241    ₹-10,931      -0.6%       -2.7%     
  HINDUNILVR  34     FMCG       01-Sep-25   2,602.4     2,491.1     670    ₹-74,561      -4.3%       -0.9%     
  BOSCHLTD    8      AUTO       01-Sep-25   40,785.0    38,320.0    42     ₹-103,530     -6.0%       -2.3%     

  AFTER: Invested ₹34,766,563 | Cash ₹500,198 | Total ₹35,266,762 | Positions 19/20 | Slot ₹1,763,862

========================================================================
  REBALANCE #131  —  03 Nov 2025
  NAV: ₹36,210,362  |  Slot: ₹1,810,518  |  Cash: ₹500,198
========================================================================
  [SECTOR CAP≤4] dropped: BANKBARODA, PNB

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TORNTPHARM  74     HEALTH     01-Aug-24   3,146.3     3,596.4     552    ₹248,487      +14.3%    459d  
  BRITANNIA   69     FMCG       01-Jul-25   5,669.2     5,820.5     318    ₹48,112       +2.7%     125d  
  HDFCLIFE    85     FIN SVC    02-Jun-25   761.9       733.4       2275   ₹-64,804      -3.7%     154d  
  COROMANDEL  93     MFG        01-Oct-25   2,242.2     2,135.8     786    ₹-83,615      -4.7%     33d   
  HINDUNILVR  102    FMCG       01-Sep-25   2,602.4     2,416.2     670    ₹-124,706     -7.2%     63d   
  BOSCHLTD    112    AUTO       01-Sep-25   40,785.0    37,025.0    42     ₹-157,920     -9.2%     63d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TVSMOTOR    4      AUTO       2.646    1.17   +43.2%    +25.3%    3,498.3     517    ₹1,808,607    -0.7%     
  CANBK       5      PSU BNK    2.616    1.09   +43.6%    +30.2%    135.1       13397  ₹1,810,497    +8.6%     
  BEL         12     DEFENCE    2.082    1.15   +57.6%    +10.5%    420.5       4305   ₹1,810,295    +2.3%     
  CUMMINSIND  13     INFRA      2.065    1.14   +30.1%    +23.2%    4,359.9     415    ₹1,809,348    +5.4%     
  HINDPETRO   16     OIL&GAS    1.796    1.08   +29.6%    +18.7%    479.1       3778   ₹1,810,069    +6.6%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  INDIANB     2      PSU BNK    01-Aug-25   608.2       862.0       2854   ₹724,187      +41.7%      +8.6%     
  MUTHOOTFIN  4      FIN SVC    01-Jul-25   2,619.7     3,163.8     690    ₹375,429      +20.8%      +0.3%     
  BAJFINANCE  7      FIN SVC    02-Jun-25   906.3       1,036.7     1913   ₹249,547      +14.4%      -0.5%     
  BANKINDIA   9      PSU BNK    01-Oct-25   120.7       137.6       14615  ₹247,672      +14.0%      +7.1%     
  POLYCAB     39     MFG        01-Aug-25   6,666.4     7,590.5     260    ₹240,270      +13.9%      +0.4%     
  EICHERMOT   3      AUTO       01-Sep-25   6,280.0     7,023.5     277    ₹205,950      +11.8%      +1.3%     
  SBIN        11     PSU BNK    01-Oct-25   848.8       932.9       2078   ₹174,727      +9.9%       +4.8%     
  NYKAA       12     CONSUMP    01-Sep-25   233.7       250.1       7467   ₹122,085      +7.0%       -1.2%     
  HEROMOTOCO  10     AUTO       01-Sep-25   5,143.7     5,433.1     339    ₹98,093       +5.6%       +0.0%     
  APOLLOHOSP  47     HEALTH     01-Jul-25   7,476.5     7,814.1     241    ₹81,370       +4.5%       +0.3%     
  FORTIS      8      HEALTH     01-Oct-25   989.7       1,030.7     1782   ₹73,151       +4.1%       -1.2%     
  HDFCBANK    56     PVT BNK    02-Jun-25   937.7       976.5       1849   ₹71,818       +4.1% ⚠     +0.2%     
  GODFRYPHLP  50     FMCG       01-Oct-25   3,357.6     3,091.0     525    ₹-139,963     -7.9%       -4.2%     
  ⚠  WAZ < 0 (momentum below universe mean): HDFCBANK

  AFTER: Invested ₹34,401,718 | Cash ₹1,797,899 | Total ₹36,199,617 | Positions 18/20 | Slot ₹1,810,518

========================================================================
  REBALANCE #132  —  01 Dec 2025
  NAV: ₹36,747,942  |  Slot: ₹1,837,397  |  Cash: ₹1,797,899
========================================================================
  [SECTOR CAP≤4] dropped: PNB

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  POLYCAB     77     MFG        01-Aug-25   6,666.4     7,366.0     260    ₹181,917      +10.5%    122d  
  APOLLOHOSP  98     HEALTH     01-Jul-25   7,476.5     7,277.8     241    ₹-47,876      -2.7%     153d  
  GODFRYPHLP  89     FMCG       01-Oct-25   3,357.6     2,837.9     525    ₹-272,823     -15.5%    61d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  M&MFIN      4      FIN SVC    2.857    1.20   +39.5%    +44.9%    367.9       4994   ₹1,837,293    +9.0%     
  MARUTI      7      AUTO       2.296    0.79   +48.7%    +8.8%     16,097.0    114    ₹1,835,058    +1.2%     
  RELIANCE    13     OIL&GAS    1.890    1.16   +21.4%    +15.4%    1,558.9     1178   ₹1,836,375    +2.7%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  2      FIN SVC    01-Jul-25   2,619.7     3,779.1     690    ₹799,984      +44.3%      +6.6%     
  INDIANB     4      PSU BNK    01-Aug-25   608.2       868.8       2854   ₹743,748      +42.8%      +2.6%     
  HEROMOTOCO  11     AUTO       01-Sep-25   5,143.7     6,175.1     339    ₹349,643      +20.1%      +7.1%     
  BANKINDIA   8      PSU BNK    01-Oct-25   120.7       142.6       14615  ₹319,992      +18.1%      +1.8%     
  EICHERMOT   7      AUTO       01-Sep-25   6,280.0     7,125.5     277    ₹234,204      +13.5%      +1.6%     
  NYKAA       16     CONSUMP    01-Sep-25   233.7       264.9       7467   ₹232,746      +13.3%      +0.6%     
  SBIN        9      PSU BNK    01-Oct-25   848.8       955.9       2078   ₹222,492      +12.6%      +1.1%     
  BAJFINANCE  10     FIN SVC    02-Jun-25   906.3       1,014.9     1913   ₹207,718      +12.0%      -0.3%     
  CANBK       3      PSU BNK    03-Nov-25   135.1       145.7       13397  ₹141,364      +7.8%       +3.7%     
  HDFCBANK    42     PVT BNK    02-Jun-25   937.7       985.8       1849   ₹89,007       +5.1%       +0.5%     
  TVSMOTOR    17     AUTO       03-Nov-25   3,498.3     3,649.0     517    ₹77,950       +4.3%       +4.5%     
  CUMMINSIND  25     INFRA      03-Nov-25   4,359.9     4,523.6     415    ₹67,958       +3.8%       +4.5%     
  BEL         30     DEFENCE    03-Nov-25   420.5       415.5       4305   ₹-21,648      -1.2%       +0.3%     
  HINDPETRO   34     OIL&GAS    03-Nov-25   479.1       452.0       3778   ₹-102,602     -5.7%       -3.0%     
  FORTIS      58     HEALTH     01-Oct-25   989.7       904.8       1782   ₹-151,114     -8.6%       -4.9%     

  AFTER: Invested ₹35,299,740 | Cash ₹1,441,662 | Total ₹36,741,401 | Positions 18/20 | Slot ₹1,837,397

========================================================================
  REBALANCE #133  —  01 Jan 2026
  NAV: ₹36,866,587  |  Slot: ₹1,843,329  |  Cash: ₹1,441,662
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  M&MFIN      3      FIN SVC    01-Dec-25   367.9       404.1       4994   ₹181,032      +9.9%     31d   
  TVSMOTOR    16     AUTO       03-Nov-25   3,498.3     3,781.2     517    ₹146,265      +8.1%     59d   
  BEL         60     DEFENCE    03-Nov-25   420.5       396.0       4305   ₹-105,454     -5.8%     59d   
  FORTIS      72     HEALTH     01-Oct-25   989.7       900.5       1782   ₹-158,776     -9.0%     92d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SBILIFE     3      FIN SVC    2.651    0.85   +45.5%    +14.0%    2,037.6     904    ₹1,841,961    +1.2%     
  TITAN       5      CON DUR    2.362    0.74   +22.7%    +20.3%    4,049.3     455    ₹1,842,432    +2.9%     
  INDUSTOWER  7      INFRA      2.246    0.97   +32.2%    +27.1%    435.8       4229   ₹1,842,998    +5.1%     
  BHARTIARTL  10     CONSUMP    2.180    0.88   +33.0%    +12.4%    2,110.4     873    ₹1,842,379    +0.4%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  5      FIN SVC    01-Jul-25   2,619.7     3,806.8     690    ₹819,073      +45.3%      +1.8%     
  INDIANB     22     PSU BNK    01-Aug-25   608.2       815.2       2854   ₹590,751      +34.0%      +3.5%     
  BANKINDIA   19     PSU BNK    01-Oct-25   120.7       142.3       14615  ₹316,454      +17.9%      +3.3%     
  EICHERMOT   21     AUTO       01-Sep-25   6,280.0     7,348.0     277    ₹295,836      +17.0%      +1.8%     
  SBIN        17     PSU BNK    01-Oct-25   848.8       967.3       2078   ₹246,272      +14.0%      +1.5%     
  NYKAA       9      CONSUMP    01-Sep-25   233.7       265.8       7467   ₹239,093      +13.7%      +3.2%     
  HEROMOTOCO  27     AUTO       01-Sep-25   5,143.7     5,729.8     339    ₹198,680      +11.4%      +0.4%     
  CANBK       7      PSU BNK    03-Nov-25   135.1       149.3       13397  ₹189,869      +10.5%      +2.9%     
  BAJFINANCE  54     FIN SVC    02-Jun-25   906.3       967.2       1913   ₹116,455      +6.7%       -3.0%     
  HINDPETRO   38     OIL&GAS    03-Nov-25   479.1       498.6       3778   ₹73,642       +4.1%       +5.7%     
  HDFCBANK    51     PVT BNK    02-Jun-25   937.7       975.0       1849   ₹69,090       +4.0%       -0.2%     
  MARUTI      14     AUTO       01-Dec-25   16,097.0    16,708.0    114    ₹69,654       +3.8%       +1.6%     
  CUMMINSIND  25     INFRA      03-Nov-25   4,359.9     4,450.4     415    ₹37,553       +2.1%       +0.7%     
  RELIANCE    11     OIL&GAS    01-Dec-25   1,558.9     1,568.3     1178   ₹11,139       +0.6%       +1.5%     

  AFTER: Invested ₹35,511,877 | Cash ₹1,345,959 | Total ₹36,857,836 | Positions 18/20 | Slot ₹1,843,329

========================================================================
  REBALANCE #134  —  02 Feb 2026
  NAV: ₹35,048,969  |  Slot: ₹1,752,448  |  Cash: ₹1,345,959
========================================================================
  [SECTOR CAP≤4] dropped: UNIONBANK, HINDZINC

  [REGIME OFF] Nifty 200 13,949.8 < SMA200 14,034.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  INDIANB     21     PSU BNK    01-Aug-25   608.2       817.4       2854   ₹596,899      +34.4%      -3.0%     
  MUTHOOTFIN  11     FIN SVC    01-Jul-25   2,619.7     3,505.5     690    ₹611,141      +33.8%      -8.4%     
  BANKINDIA   20     PSU BNK    01-Oct-25   120.7       146.9       14615  ₹382,689      +21.7%      -3.1%     
  SBIN        6      PSU BNK    01-Oct-25   848.8       1,010.5     2078   ₹335,983      +19.0%      -0.3%     
  EICHERMOT   28     AUTO       01-Sep-25   6,280.0     6,985.5     277    ₹195,424      +11.2%      -2.8%     
  HEROMOTOCO  26     AUTO       01-Sep-25   5,143.7     5,515.5     339    ₹126,025      +7.2%       -0.1%     
  CANBK       9      PSU BNK    03-Nov-25   135.1       141.8       13397  ₹89,098       +4.9%       -3.6%     
  NYKAA       36     CONSUMP    01-Sep-25   233.7       237.6       7467   ₹29,047       +1.7%       -3.3%     
  INDUSTOWER  17     INFRA      01-Jan-26   435.8       431.8       4229   ₹-16,916      -0.9%       +1.0%     
  BAJFINANCE  82     FIN SVC    02-Jun-25   906.3       898.2       1913   ₹-15,496      -0.9% ⚠     -4.4%     
  SBILIFE     16     FIN SVC    01-Jan-26   2,037.6     1,998.2     904    ₹-35,568      -1.9%       -1.7%     
  TITAN       30     CON DUR    01-Jan-26   4,049.3     3,953.2     455    ₹-43,726      -2.4%       -2.2%     
  HDFCBANK    66     PVT BNK    02-Jun-25   937.7       913.0       1849   ₹-45,593      -2.6% ⚠     -1.2%     
  HINDPETRO   44     OIL&GAS    03-Nov-25   479.1       453.2       3778   ₹-97,691      -5.4%       +2.1%     
  CUMMINSIND  33     INFRA      03-Nov-25   4,359.9     4,073.8     415    ₹-118,731     -6.6%       -0.1%     
  BHARTIARTL  54     CONSUMP    01-Jan-26   2,110.4     1,965.4     873    ₹-126,585     -6.9%       -2.2%     
  MARUTI      81     AUTO       01-Dec-25   16,097.0    14,384.0    114    ₹-195,282     -10.6% ⚠    -7.8%     
  RELIANCE    72     OIL&GAS    01-Dec-25   1,558.9     1,384.0     1178   ₹-206,022     -11.2% ⚠    -3.1%     
  ⚠  WAZ < 0 (momentum below universe mean): HDFCBANK, RELIANCE, MARUTI, BAJFINANCE

  AFTER: Invested ₹33,703,011 | Cash ₹1,345,959 | Total ₹35,048,969 | Positions 18/20 | Slot ₹1,752,448

========================================================================
  REBALANCE #135  —  02 Mar 2026
  NAV: ₹36,912,217  |  Slot: ₹1,845,611  |  Cash: ₹1,345,959
========================================================================
  [SECTOR CAP≤4] dropped: UNIONBANK

  [REGIME OFF] Nifty 200 13,931.3 < SMA200 14,130.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  INDIANB     10     PSU BNK    01-Aug-25   608.2       956.3       2854   ₹993,295      +57.2%      +4.4%     
  BANKINDIA   14     PSU BNK    01-Oct-25   120.7       166.8       14615  ₹674,659      +38.2%      +1.7%     
  SBIN        1      PSU BNK    01-Oct-25   848.8       1,168.8     2078   ₹665,025      +37.7%      +1.6%     
  MUTHOOTFIN  50     FIN SVC    01-Jul-25   2,619.7     3,442.8     690    ₹567,898      +31.4%      -2.9%     
  EICHERMOT   12     AUTO       01-Sep-25   6,280.0     7,826.0     277    ₹428,242      +24.6%      +0.5%     
  NYKAA       37     CONSUMP    01-Sep-25   233.7       259.1       7467   ₹189,438      +10.9%      -2.0%     
  CUMMINSIND  17     INFRA      03-Nov-25   4,359.9     4,816.8     415    ₹189,624      +10.5%      +4.1%     
  CANBK       18     PSU BNK    03-Nov-25   135.1       148.7       13397  ₹181,050      +10.0%      +0.6%     
  HEROMOTOCO  46     AUTO       01-Sep-25   5,143.7     5,591.5     339    ₹151,790      +8.7%       -0.2%     
  BAJFINANCE  64     FIN SVC    02-Jun-25   906.3       972.3       1913   ₹126,247      +7.3% ⚠     -1.9%     
  TITAN       24     CON DUR    01-Jan-26   4,049.3     4,270.3     455    ₹100,555      +5.5%       +0.9%     
  INDUSTOWER  —      INFRA      01-Jan-26   435.8       448.5       4229   ₹53,920       +2.9%       -2.1%     
  SBILIFE     31     FIN SVC    01-Jan-26   2,037.6     2,029.4     904    ₹-7,402       -0.4%       -0.9%     
  HDFCBANK    111    PVT BNK    02-Jun-25   937.7       865.1       1849   ₹-134,174     -7.7% ⚠     -3.9%     
  MARUTI      87     AUTO       01-Dec-25   16,097.0    14,388.0    114    ₹-194,826     -10.6% ⚠    -4.5%     
  BHARTIARTL  89     CONSUMP    01-Jan-26   2,110.4     1,873.2     873    ₹-207,076     -11.2% ⚠    -4.8%     
  HINDPETRO   56     OIL&GAS    03-Nov-25   479.1       424.5       3778   ₹-206,308     -11.4%      -4.3%     
  RELIANCE    105    OIL&GAS    01-Dec-25   1,558.9     1,351.8     1178   ₹-244,014     -13.3% ⚠    -4.3%     
  ⚠  WAZ < 0 (momentum below universe mean): BAJFINANCE, MARUTI, BHARTIARTL, RELIANCE, HDFCBANK

  AFTER: Invested ₹35,566,258 | Cash ₹1,345,959 | Total ₹36,912,217 | Positions 18/20 | Slot ₹1,845,611

========================================================================
  REBALANCE #136  —  01 Apr 2026
  NAV: ₹33,138,057  |  Slot: ₹1,656,903  |  Cash: ₹1,345,959
========================================================================

  [REGIME OFF] Nifty 200 12,720.3 < SMA200 14,067.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  INDIANB     5      PSU BNK    01-Aug-25   608.2       869.5       2854   ₹745,565      +42.9%      -0.2%     
  MUTHOOTFIN  45     FIN SVC    01-Jul-25   2,619.7     3,228.7     690    ₹420,176      +23.2%      -1.7%     
  SBIN        11     PSU BNK    01-Oct-25   848.8       999.8       2078   ₹313,734      +17.8%      -4.7%     
  BANKINDIA   27     PSU BNK    01-Oct-25   120.7       137.2       14615  ₹241,586      +13.7%      -6.2%     
  EICHERMOT   37     AUTO       01-Sep-25   6,280.0     6,825.5     277    ₹151,104      +8.7%       -3.2%     
  CUMMINSIND  7      INFRA      03-Nov-25   4,359.9     4,609.1     415    ₹103,428      +5.7%       -0.3%     
  NYKAA       33     CONSUMP    01-Sep-25   233.7       240.0       7467   ₹46,594       +2.7%       -2.1%     
  TITAN       17     CON DUR    01-Jan-26   4,049.3     4,065.5     455    ₹7,371        +0.4%       -0.2%     
  HEROMOTOCO  26     AUTO       01-Sep-25   5,143.7     5,122.0     339    ₹-7,370       -0.4%       -3.5%     
  INDUSTOWER  —      INFRA      01-Jan-26   435.8       423.2       4229   ₹-53,074      -2.9%       -2.4%     
  CANBK       44     PSU BNK    03-Nov-25   135.1       123.2       13397  ₹-159,521     -8.8%       -6.8%     
  BAJFINANCE  99     FIN SVC    02-Jun-25   906.3       812.3       1913   ₹-179,769     -10.4% ⚠    -6.7%     
  SBILIFE     57     FIN SVC    01-Jan-26   2,037.6     1,790.5     904    ₹-223,349     -12.1%      -5.5%     
  RELIANCE    70     OIL&GAS    01-Dec-25   1,558.9     1,362.9     1178   ₹-230,881     -12.6% ⚠    -1.6%     
  BHARTIARTL  84     CONSUMP    01-Jan-26   2,110.4     1,781.9     873    ₹-286,780     -15.6% ⚠    -3.3%     
  HDFCBANK    138    PVT BNK    02-Jun-25   937.7       730.2       1849   ₹-383,639     -22.1% ⚠    -8.1%     
  MARUTI      109    AUTO       01-Dec-25   16,097.0    12,509.0    114    ₹-409,032     -22.3% ⚠    -4.6%     
  HINDPETRO   110    OIL&GAS    03-Nov-25   479.1       335.5       3778   ₹-542,361     -30.0% ⚠    -7.5%     
  ⚠  WAZ < 0 (momentum below universe mean): RELIANCE, BHARTIARTL, BAJFINANCE, MARUTI, HINDPETRO, HDFCBANK

  AFTER: Invested ₹31,792,098 | Cash ₹1,345,959 | Total ₹33,138,057 | Positions 18/20 | Slot ₹1,656,903

========================================================================
  REBALANCE #137  —  01 May 2026
  NAV: ₹34,672,418  |  Slot: ₹1,733,621  |  Cash: ₹1,345,959
========================================================================
  [SECTOR CAP≤4] dropped: ADANIENSOL, BHEL, ADANIGREEN

  [REGIME OFF] Nifty 200 13,705.5 < SMA200 14,023.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  INDIANB     42     PSU BNK    01-Aug-25   608.2       834.1       2854   ₹644,544      +37.1%      -5.6%     
  MUTHOOTFIN  46     FIN SVC    01-Jul-25   2,619.7     3,424.2     690    ₹555,071      +30.7%      -1.0%     
  SBIN        35     PSU BNK    01-Oct-25   848.8       1,049.5     2078   ₹417,121      +23.6%      -1.1%     
  CUMMINSIND  3      INFRA      03-Nov-25   4,359.9     5,266.4     415    ₹376,208      +20.8%      +3.6%     
  NYKAA       28     CONSUMP    01-Sep-25   233.7       264.8       7467   ₹231,701      +13.3%      +1.4%     
  EICHERMOT   51     AUTO       01-Sep-25   6,280.0     7,109.0     277    ₹229,633      +13.2%      -0.1%     
  BANKINDIA   82     PSU BNK    01-Oct-25   120.7       135.4       14615  ₹215,687      +12.2% ⚠    -4.3%     
  TITAN       23     CON DUR    01-Jan-26   4,049.3     4,385.2     455    ₹152,835      +8.3%       +0.1%     
  BAJFINANCE  70     FIN SVC    02-Jun-25   906.3       931.3       1913   ₹47,818       +2.8% ⚠     +2.5%     
  HEROMOTOCO  52     AUTO       01-Sep-25   5,143.7     5,099.0     339    ₹-15,167      -0.9%       -1.2%     
  CANBK       63     PSU BNK    03-Nov-25   135.1       130.4       13397  ₹-64,198      -3.5%       -2.7%     
  INDUSTOWER  —      INFRA      01-Jan-26   435.8       410.0       4229   ₹-109,320     -5.9%       -1.5%     
  RELIANCE    59     OIL&GAS    01-Dec-25   1,558.9     1,424.2     1178   ₹-158,650     -8.6%       +3.9%     
  BHARTIARTL  85     CONSUMP    01-Jan-26   2,110.4     1,886.8     873    ₹-195,203     -10.6% ⚠    +1.9%     
  SBILIFE     104    FIN SVC    01-Jan-26   2,037.6     1,819.0     904    ₹-197,585     -10.7% ⚠    -2.3%     
  MARUTI      95     AUTO       01-Dec-25   16,097.0    13,314.0    114    ₹-317,262     -17.3% ⚠    +0.7%     
  HDFCBANK    138    PVT BNK    02-Jun-25   937.7       759.1       1849   ₹-330,072     -19.0% ⚠    -2.3%     
  HINDPETRO   101    OIL&GAS    03-Nov-25   479.1       374.5       3778   ₹-395,019     -21.8% ⚠    +0.9%     
  ⚠  WAZ < 0 (momentum below universe mean): BAJFINANCE, BANKINDIA, BHARTIARTL, MARUTI, HINDPETRO, SBILIFE, HDFCBANK

  AFTER: Invested ₹33,326,460 | Cash ₹1,345,959 | Total ₹34,672,418 | Positions 18/20 | Slot ₹1,733,621

========================================================================
  REBALANCE #138  —  01 Jun 2026
  NAV: ₹33,847,047  |  Slot: ₹1,692,352  |  Cash: ₹1,345,959
========================================================================
  [SECTOR CAP≤4] dropped: ADANIGREEN, PREMIERENE, CGPOWER

  [REGIME OFF] Nifty 200 13,528.3 < SMA200 13,992.6 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  INDIANB     72     PSU BNK    01-Aug-25   608.2       792.9       2854   ₹526,897      +30.4% ⚠    -3.3%     
  CUMMINSIND  6      INFRA      03-Nov-25   4,359.9     5,680.5     415    ₹548,059      +30.3%      +3.4%     
  MUTHOOTFIN  39     FIN SVC    01-Jul-25   2,619.7     3,246.4     690    ₹432,389      +23.9%      -3.3%     
  NYKAA       45     CONSUMP    01-Sep-25   233.7       266.7       7467   ₹246,187      +14.1%      -0.3%     
  BANKINDIA   89     PSU BNK    01-Oct-25   120.7       136.7       14615  ₹234,606      +13.3% ⚠    -1.3%     
  EICHERMOT   62     AUTO       01-Sep-25   6,280.0     7,100.5     277    ₹227,278      +13.1%      -0.9%     
  SBIN        99     PSU BNK    01-Oct-25   848.8       954.1       2078   ₹218,813      +12.4% ⚠    -2.5%     
  TITAN       79     CON DUR    01-Jan-26   4,049.3     4,024.6     455    ₹-11,238      -0.6% ⚠     -3.3%     
  INDUSTOWER  —      INFRA      01-Jan-26   435.8       431.5       4229   ₹-18,396      -1.0%       +0.9%     
  BAJFINANCE  110    FIN SVC    02-Jun-25   906.3       883.6       1913   ₹-43,350      -2.5% ⚠     -3.4%     
  HEROMOTOCO  98     AUTO       01-Sep-25   5,143.7     4,819.9     339    ₹-109,782     -6.3% ⚠     -4.1%     
  CANBK       96     PSU BNK    03-Nov-25   135.1       123.9       13397  ₹-150,961     -8.3% ⚠     -2.8%     
  SBILIFE     117    FIN SVC    01-Jan-26   2,037.6     1,812.5     904    ₹-203,461     -11.0% ⚠    -2.5%     
  BHARTIARTL  —      CONSUMP    01-Jan-26   2,110.4     1,810.6     873    ₹-261,725     -14.2%      -2.2%     
  RELIANCE    106    OIL&GAS    01-Dec-25   1,558.9     1,313.9     1178   ₹-288,572     -15.7% ⚠    -2.8%     
  HINDPETRO   101    OIL&GAS    03-Nov-25   479.1       388.6       3778   ₹-341,938     -18.9% ⚠    +0.5%     
  MARUTI      109    AUTO       01-Dec-25   16,097.0    12,946.0    114    ₹-359,214     -19.6% ⚠    -1.8%     
  HDFCBANK    149    PVT BNK    02-Jun-25   937.7       730.6       1849   ₹-382,820     -22.1% ⚠    -3.0%     
  ⚠  WAZ < 0 (momentum below universe mean): INDIANB, TITAN, BANKINDIA, CANBK, HEROMOTOCO, SBIN, HINDPETRO, RELIANCE, MARUTI, BAJFINANCE, SBILIFE, HDFCBANK

  AFTER: Invested ₹32,501,088 | Cash ₹1,345,959 | Total ₹33,847,047 | Positions 18/20 | Slot ₹1,692,352

========================================================================
  REBALANCE #139  —  01 Jul 2026
  NAV: ₹34,884,898  |  Slot: ₹1,744,245  |  Cash: ₹1,345,959
========================================================================
  [SECTOR CAP≤4] dropped: GVT&D, POWERINDIA

  [REGIME OFF] Nifty 200 13,879.0 < SMA200 13,987.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  INDIANB     82     PSU BNK    01-Aug-25   608.2       821.0       2854   ₹607,066      +35.0% ⚠    -1.8%     
  NYKAA       14     CONSUMP    01-Sep-25   233.7       308.6       7467   ₹559,428      +32.1%      +5.9%     
  CUMMINSIND  9      INFRA      03-Nov-25   4,359.9     5,663.0     415    ₹540,797      +29.9%      +0.1%     
  SBIN        43     PSU BNK    01-Oct-25   848.8       1,047.4     2078   ₹412,690      +23.4%      +2.4%     
  BANKINDIA   70     PSU BNK    01-Oct-25   120.7       141.6       14615  ₹306,219      +17.4% ⚠    -1.1%     
  EICHERMOT   59     AUTO       01-Sep-25   6,280.0     7,139.0     277    ₹237,943      +13.7%      -3.2%     
  BAJFINANCE  48     FIN SVC    02-Jun-25   906.3       1,014.9     1913   ₹207,792      +12.0%      +6.4%     
  MUTHOOTFIN  112    FIN SVC    01-Jul-25   2,619.7     2,914.9     690    ₹203,654      +11.3% ⚠    -5.8%     
  TITAN       55     CON DUR    01-Jan-26   4,049.3     4,398.6     455    ₹158,932      +8.6%       +2.5%     
  HEROMOTOCO  100    AUTO       01-Sep-25   5,143.7     4,834.9     339    ₹-104,697     -6.0% ⚠     -1.6%     
  CANBK       81     PSU BNK    03-Nov-25   135.1       126.3       13397  ₹-118,724     -6.6% ⚠     -2.4%     
  MARUTI      49     AUTO       01-Dec-25   16,097.0    14,395.0    114    ₹-194,028     -10.6%      +6.2%     
  INDUSTOWER  —      INFRA      01-Jan-26   435.8       388.9       4229   ₹-198,551     -10.8%      -4.3%     
  BHARTIARTL  —      CONSUMP    01-Jan-26   2,110.4     1,871.0     873    ₹-208,996     -11.3%      +0.8%     
  SBILIFE     119    FIN SVC    01-Jan-26   2,037.6     1,790.9     904    ₹-222,987     -12.1% ⚠    +0.7%     
  HDFCBANK    120    PVT BNK    02-Jun-25   937.7       796.2       1849   ₹-261,648     -15.1% ⚠    +2.4%     
  RELIANCE    132    OIL&GAS    01-Dec-25   1,558.9     1,308.0     1178   ₹-295,551     -16.1% ⚠    -0.3%     
  HINDPETRO   83     OIL&GAS    03-Nov-25   479.1       392.1       3778   ₹-328,715     -18.2% ⚠    -0.6%     
  ⚠  WAZ < 0 (momentum below universe mean): BANKINDIA, CANBK, INDIANB, HINDPETRO, HEROMOTOCO, MUTHOOTFIN, SBILIFE, HDFCBANK, RELIANCE

  AFTER: Invested ₹33,538,939 | Cash ₹1,345,959 | Total ₹34,884,898 | Positions 18/20 | Slot ₹1,744,245

========================================================================
  FINAL SUMMARY
========================================================================
  Period        : 2015-01-01 → 2026-07-01  (11.5 years)
  Starting Cap  : ₹2,000,000
  Final Value   : ₹34,884,898
  Total Return  : +1644.2%  (on total invested)
  CAGR          : +28.2%

  Closed Trades : 421  |  Open: 18
  Win Rate      : 58.0%  (244W / 177L)
  Profit Factor : 4.19
  Avg hold      : 166 days
  Total charges : ₹874,724
  Closed net P&L: ₹31,345,220
  Open unreal   : ₹1,300,623

  YEAR-BY-YEAR:
  2015  + 10.3%  ██████████
  2016  + 13.7%  █████████████
  2017  + 49.3%  ████████████████████████████████████████
  2018  -  4.1%  ░░░░
  2019  + 16.9%  ████████████████
  2020  + 33.0%  █████████████████████████████████
  2021  + 91.6%  ████████████████████████████████████████
  2022  + 46.7%  ████████████████████████████████████████
  2023  + 42.1%  ████████████████████████████████████████
  2024  + 62.6%  ████████████████████████████████████████
  2025  +  1.3%  █
  2026  -  5.1%  ░░░░░

  Rebalance NAV exported → mom20_rebal.csv (139 rows)
