=== LM250 — Nifty LargeMidcap 250, Monthly, β≤1.2, Sector≤4 | Regime ON [SMA200] ===
    top_n=20 buffer_in=15 buffer_out=40 beta_cap=1.2 sector_cap=4
Loading PIT universe...
  417 unique PIT tickers across all periods
  Sector map loaded: 417 tickers
Loading cached data from /Users/jay/Desktop/relative_strength/data/cache/lm250_daily.pkl...
Fetching Nifty 50 (^NSEI) for beta...
  2392 bars
Fetching Nifty 200 (^CRSLDX) for regime...
  2387 bars
  Trading days: 2092 (2017-12-01 → 2026-05-22)
  Rebalance dates: 102

==================================================================================================================================
  LM250 PIT BACKTEST  |  NAV/20 slot  |  Monthly Rebalance  |  Nifty LargeMidcap 250  |  β≤1.2  |  Sector≤4  |  Regime ON [SMA200]
==================================================================================================================================

========================================================================
  REBALANCE #01  —  01 Dec 2017
  NAV: ₹2,000,000  |  Slot: ₹100,000  |  Cash: ₹2,000,000
========================================================================
  [SECTOR CAP≤4] dropped: BRITANNIA

  EXITS (0)
    —

  ENTRIES (20)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  VAKRANGEE   1      IT                    3.936    0.03   +163.6%   +41.4%    320.5       312    ₹99,991     
  BBTC        2      FMCG                  3.453    0.99   +197.5%   +48.2%    1,478.3     67     ₹99,049     
  HATSUN      3      FMCG                  3.323    -0.09  +158.5%   +40.1%    602.7       165    ₹99,441     
  SOLARINDS   4      CHEMICALS             3.135    0.54   +72.9%    +30.5%    1,121.0     89     ₹99,768     
  LTTS        5      IT                    2.885    0.03   +30.6%    +42.8%    994.3       100    ₹99,434     
  GILLETTE    6      FMCG                  2.882    0.38   +58.2%    +22.0%    5,834.9     17     ₹99,193     
  ENDURANCE   7      AUTO_ANCILLARY        2.821    0.51   +122.5%   +31.0%    1,251.9     79     ₹98,901     
  FRETAIL     8      CONSUMER_DISCRETIONARY  2.703    1.11   +348.9%   +1.6%     544.8       183    ₹99,689     
  KRBL        9      FMCG                  2.638    0.94   +133.4%   +32.3%    570.9       175    ₹99,908     
  TVSMOTOR    10     AUTO                  2.530    0.96   +103.2%   +19.5%    692.6       144    ₹99,730     
  IGL         11     OIL_GAS               2.501    0.77   +93.0%    +24.9%    139.5       716    ₹99,901     
  GODREJPROP  12     REAL_ESTATE           2.295    1.19   +139.9%   +20.7%    721.2       138    ₹99,519     
  BALKRISIND  13     AUTO_ANCILLARY        2.292    0.53   +130.1%   +28.2%    978.3       102    ₹99,789     
  MARUTI      14     AUTO                  2.171    1.09   +70.9%    +10.2%    7,994.1     12     ₹95,929     
  HDFC        16     FINANCIAL_SERVICES    2.074    0.73   +57.8%    +4.6%     431.2       231    ₹99,612     
  HDFCBANK    17     BANKING               2.074    0.73   +57.8%    +4.6%     431.2       231    ₹99,612     
  WHIRLPOOL   18     CONSUMER_DURABLES     2.007    1.13   +63.2%    +29.4%    1,499.3     66     ₹98,951     
  FINCABLES   19     CAPITAL_GOODS         1.991    0.76   +65.0%    +23.7%    615.2       162    ₹99,655     
  RELIANCE    20     OIL_GAS               1.945    0.99   +83.9%    +13.0%    402.0       248    ₹99,701     
  NBCC        21     INFRASTRUCTURE        1.920    1.09   +73.6%    +25.5%    79.7        1255   ₹99,977     

  HOLDS (0)
    —

  AFTER: Invested ₹1,987,751 | Cash ₹9,889 | Total ₹1,997,640 | Positions 20/20 | Slot ₹100,000

========================================================================
  REBALANCE #02  —  01 Jan 2018
  NAV: ₹2,062,732  |  Slot: ₹103,137  |  Cash: ₹9,889
========================================================================
  [SECTOR CAP≤4] dropped: TATACONSUM

  EXITS (6)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  IGL         40     OIL_GAS               01-Dec-17   139.5       149.3       716    ₹7,001        +7.0%     31d   
  HDFC        53     FINANCIAL_SERVICES    01-Dec-17   431.2       432.7       231    ₹340          +0.3%     31d   
  HDFCBANK    54     BANKING               01-Dec-17   431.2       432.7       231    ₹340          +0.3%     31d   
  RELIANCE    52     OIL_GAS               01-Dec-17   402.0       401.9       248    ₹-38          -0.0%     31d   
  GODREJPROP  46     REAL_ESTATE           01-Dec-17   721.2       703.8       138    ₹-2,387       -2.4%     31d   
  NBCC        84     INFRASTRUCTURE        01-Dec-17   79.7        75.5        1255   ₹-5,181       -5.2%     31d   

  ENTRIES (6)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  SAIL        7      METALS                2.708    0.46   +91.1%    +72.8%    77.5        1331   ₹103,096    
  TITAN       8      CONSUMER_DISCRETIONARY  2.671    0.12   +170.8%   +45.8%    828.8       124    ₹102,767    
  JINDALSTEL  9      CHEMICALS             2.650    0.29   +210.8%   +53.1%    203.3       507    ₹103,087    
  PAGEIND     13     TEXTILES              2.372    -0.09  +96.1%    +36.9%    23,035.8    4      ₹92,143     
  DLF         15     REAL_ESTATE           2.346    -0.25  +139.6%   +56.2%    240.2       429    ₹103,047    
  3MINDIA     17     CAPITAL_GOODS         2.163    0.21   +84.6%    +34.7%    17,896.8    5      ₹89,484     

  HOLDS (14)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  VAKRANGEE   1      IT                    01-Dec-17   320.5       377.9       312    ₹17,928       +17.9%    
  BALKRISIND  10     AUTO_ANCILLARY        01-Dec-17   978.3       1,106.8     102    ₹13,103       +13.1%    
  MARUTI      6      AUTO                  01-Dec-17   7,994.1     8,962.5     12     ₹11,621       +12.1%    
  BBTC        5      FMCG                  01-Dec-17   1,478.3     1,618.0     67     ₹9,356        +9.4%     
  WHIRLPOOL   28     CONSUMER_DURABLES     01-Dec-17   1,499.3     1,588.1     66     ₹5,866        +5.9%     
  TVSMOTOR    12     AUTO                  01-Dec-17   692.6       732.5       144    ₹5,753        +5.8%     
  ENDURANCE   3      AUTO_ANCILLARY        01-Dec-17   1,251.9     1,317.4     79     ₹5,171        +5.2%     
  FINCABLES   30     CAPITAL_GOODS         01-Dec-17   615.2       646.2       162    ₹5,022        +5.0%     
  SOLARINDS   4      CHEMICALS             01-Dec-17   1,121.0     1,154.8     89     ₹3,006        +3.0%     
  GILLETTE    11     FMCG                  01-Dec-17   5,834.9     6,012.4     17     ₹3,017        +3.0%     
  KRBL        37     FMCG                  01-Dec-17   570.9       557.0       175    ₹-2,438       -2.4%     
  LTTS        16     IT                    01-Dec-17   994.3       966.8       100    ₹-2,756       -2.8%     
  FRETAIL     14     CONSUMER_DISCRETIONARY  01-Dec-17   544.8       526.3       183    ₹-3,376       -3.4%     
  HATSUN      32     FMCG                  01-Dec-17   602.7       564.8       165    ₹-6,253       -6.3%     

  AFTER: Invested ₹2,048,074 | Cash ₹13,954 | Total ₹2,062,028 | Positions 20/20 | Slot ₹103,137

========================================================================
  REBALANCE #03  —  01 Feb 2018
  NAV: ₹1,984,768  |  Slot: ₹99,238  |  Cash: ₹13,954
========================================================================

  EXITS (12)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  FRETAIL     43     CONSUMER_DISCRETIONARY  01-Dec-17   544.8       542.5       183    ₹-403         -0.4%     62d   
  KRBL        157    FMCG                  01-Dec-17   570.9       560.9       175    ₹-1,749       -1.8%     62d   
  WHIRLPOOL   58     CONSUMER_DURABLES     01-Dec-17   1,499.3     1,467.9     66     ₹-2,070       -2.1%     62d   
  SAIL        84     METALS                01-Jan-18   77.5        74.7        1331   ₹-3,726       -3.6%     31d   
  DLF         52     REAL_ESTATE           01-Jan-18   240.2       229.3       429    ₹-4,697       -4.6%     31d   
  SOLARINDS   71     CHEMICALS             01-Dec-17   1,121.0     1,054.9     89     ₹-5,883       -5.9%     62d   
  BBTC        123    FMCG                  01-Dec-17   1,478.3     1,383.9     67     ₹-6,329       -6.4%     62d   
  ENDURANCE   75     AUTO_ANCILLARY        01-Dec-17   1,251.9     1,160.3     79     ₹-7,241       -7.3%     62d   
  TVSMOTOR    119    AUTO                  01-Dec-17   692.6       641.5       144    ₹-7,356       -7.4%     62d   
  HATSUN      50     FMCG                  01-Dec-17   602.7       541.1       165    ₹-10,157      -10.2%    62d   
  PAGEIND     98     TEXTILES              01-Jan-18   23,035.8    19,455.8    4      ₹-14,320      -15.5%    31d   
  VAKRANGEE   81     IT                    01-Dec-17   320.5       262.4       312    ₹-18,124      -18.1%    62d   

  ENTRIES (11)
  [52w filter blocked 1: KIOCL(-35.5%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  BIOCON      4      PHARMA                2.542    0.41   +82.6%    +62.8%    305.3       325    ₹99,230     
  PFIZER      5      PHARMA                2.508    0.13   +30.1%    +31.1%    1,997.6     49     ₹97,881     
  VBL         6      FMCG                  2.255    0.23   +70.7%    +35.2%    39.1        2535   ₹99,231     
  ADANIENT    7      ENERGY                2.247    -0.17  +132.3%   +60.5%    114.8       864    ₹99,201     
  THERMAX     8      CAPITAL_GOODS         2.218    0.16   +61.8%    +31.5%    1,231.6     80     ₹98,525     
  PCJEWELLER  11     CONSUMER_DISCRETIONARY  2.140    0.86   +149.8%   +39.3%    48.1        2061   ₹99,200     
  MPHASIS     12     IT                    2.098    -0.04  +61.5%    +26.3%    722.3       137    ₹98,954     
  LT          13     CAPITAL_GOODS         2.095    0.15   +55.4%    +20.2%    1,278.9     77     ₹98,473     
  TATACONSUM  14     FMCG                  2.086    -0.15  +129.8%   +27.4%    267.7       370    ₹99,037     
  JUBLFOOD    15     CONSUMER_DISCRETIONARY  1.999    0.18   +133.4%   +26.6%    200.8       494    ₹99,192     
  TECHM       16     IT                    1.981    -0.20  +33.9%    +30.6%    462.4       214    ₹98,964     

  HOLDS (8)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  JINDALSTEL  3      CHEMICALS             01-Jan-18   203.3       262.2       507    ₹29,871       +29.0%    
  LTTS        2      IT                    01-Dec-17   994.3       1,181.4     100    ₹18,711       +18.8%    
  MARUTI      19     AUTO                  01-Dec-17   7,994.1     8,730.3     12     ₹8,835        +9.2%     
  FINCABLES   34     CAPITAL_GOODS         01-Dec-17   615.2       665.7       162    ₹8,188        +8.2%     
  BALKRISIND  10     AUTO_ANCILLARY        01-Dec-17   978.3       1,047.3     102    ₹7,031        +7.0%     
  3MINDIA     9      CAPITAL_GOODS         01-Jan-18   17,896.8    17,913.1    5      ₹81           +0.1%     
  GILLETTE    31     FMCG                  01-Dec-17   5,834.9     5,816.3     17     ₹-317         -0.3%     
  TITAN       24     CONSUMER_DISCRETIONARY  01-Jan-18   828.8       807.9       124    ₹-2,586       -2.5%     

  AFTER: Invested ₹1,947,040 | Cash ₹36,436 | Total ₹1,983,476 | Positions 19/20 | Slot ₹99,238

========================================================================
  REBALANCE #04  —  01 Mar 2018
  NAV: ₹1,909,371  |  Slot: ₹95,469  |  Cash: ₹36,436
========================================================================

  EXITS (7)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  FINCABLES   41     CAPITAL_GOODS         01-Dec-17   615.2       649.7       162    ₹5,603        +5.6%     90d   
  MARUTI      47     AUTO                  01-Dec-17   7,994.1     8,239.6     12     ₹2,947        +3.1%     90d   
  BALKRISIND  55     AUTO_ANCILLARY        01-Dec-17   978.3       982.9       102    ₹462          +0.5%     90d   
  TATACONSUM  56     FMCG                  01-Feb-18   267.7       255.1       370    ₹-4,637       -4.7%     28d   
  TITAN       68     CONSUMER_DISCRETIONARY  01-Jan-18   828.8       787.7       124    ₹-5,093       -5.0%     59d   
  THERMAX     42     CAPITAL_GOODS         01-Feb-18   1,231.6     1,167.9     80     ₹-5,095       -5.2%     28d   
  PCJEWELLER  —      CONSUMER_DISCRETIONARY  01-Feb-18   48.1        33.1        2061   ₹-30,940      -31.2%    28d   

  ENTRIES (6)
  [52w filter blocked 2: KIOCL(-44.9%), RCOM(-32.0%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  AARTIIND    8      CHEMICALS             2.319    0.32   +52.2%    +28.6%    277.2       344    ₹95,340     
  CHOLAHLDNG  10     FINANCIAL_SERVICES    2.114    -0.08  +68.1%    +21.3%    666.8       143    ₹95,355     
  JSWSTEEL    12     METALS                2.037    0.31   +65.3%    +19.7%    278.9       342    ₹95,370     
  SANOFI      15     PHARMA                1.960    0.20   +25.8%    +16.0%    3,641.0     26     ₹94,666     
  INFY        16     IT                    1.938    0.10   +18.5%    +18.1%    466.3       204    ₹95,135     
  INDIGO      17     CAPITAL_GOODS         1.926    0.34   +62.9%    +15.5%    1,317.1     72     ₹94,835     

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  LTTS        3      IT                    01-Dec-17   994.3       1,271.4     100    ₹27,703       +27.9%    
  JINDALSTEL  6      CHEMICALS             01-Jan-18   203.3       245.4       507    ₹21,322       +20.7%    
  3MINDIA     2      CAPITAL_GOODS         01-Jan-18   17,896.8    19,520.9    5      ₹8,121        +9.1%     
  BIOCON      5      PHARMA                01-Feb-18   305.3       309.1       325    ₹1,242        +1.3%     
  GILLETTE    21     FMCG                  01-Dec-17   5,834.9     5,902.2     17     ₹1,145        +1.2%     
  TECHM       13     IT                    01-Feb-18   462.4       463.1       214    ₹138          +0.1%     
  PFIZER      9      PHARMA                01-Feb-18   1,997.6     1,970.0     49     ₹-1,353       -1.4%     
  JUBLFOOD    22     CONSUMER_DISCRETIONARY  01-Feb-18   200.8       196.7       494    ₹-2,011       -2.0%     
  MPHASIS     14     IT                    01-Feb-18   722.3       701.9       137    ₹-2,790       -2.8%     
  VBL         7      FMCG                  01-Feb-18   39.1        37.3        2535   ₹-4,673       -4.7%     
  ADANIENT    11     ENERGY                01-Feb-18   114.8       106.8       864    ₹-6,884       -6.9%     
  LT          35     CAPITAL_GOODS         01-Feb-18   1,278.9     1,155.5     77     ₹-9,498       -9.6%     

  AFTER: Invested ₹1,785,486 | Cash ₹123,207 | Total ₹1,908,694 | Positions 18/20 | Slot ₹95,469

========================================================================
  REBALANCE #05  —  02 Apr 2018
  NAV: ₹1,862,738  |  Slot: ₹93,137  |  Cash: ₹123,207
========================================================================

  EXITS (9)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  GILLETTE    27     FMCG                  01-Dec-17   5,834.9     5,855.7     17     ₹353          +0.4%     122d  
  SANOFI      38     PHARMA                01-Mar-18   3,641.0     3,636.0     26     ₹-128         -0.1%     32d   
  INDIGO      34     CAPITAL_GOODS         01-Mar-18   1,317.1     1,313.7     72     ₹-246         -0.3%     32d   
  AARTIIND    33     CHEMICALS             01-Mar-18   277.2       274.3       344    ₹-984         -1.0%     32d   
  INFY        35     IT                    01-Mar-18   466.3       456.9       204    ₹-1,934       -2.0%     32d   
  CHOLAHLDNG  59     FINANCIAL_SERVICES    01-Mar-18   666.8       646.4       143    ₹-2,914       -3.1%     32d   
  BIOCON      36     PHARMA                01-Feb-18   305.3       295.2       325    ₹-3,302       -3.3%     60d   
  LT          29     CAPITAL_GOODS         01-Feb-18   1,278.9     1,173.7     77     ₹-8,094       -8.2%     60d   
  ADANIENT    —      ENERGY                01-Feb-18   114.8       86.3        864    ₹-24,615      -24.8%    60d   

  ENTRIES (9)
  [52w filter blocked 2: KIOCL(-55.2%), STRTECH(-21.2%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  ASHOKLEY    3      AUTO                  2.793    0.44   +72.1%    +24.9%    63.4        1469   ₹93,079     
  DBL         4      INFRASTRUCTURE        2.672    1.01   +219.8%   +10.8%    1,099.5     84     ₹92,360     
  BRITANNIA   6      FMCG                  2.359    0.13   +55.7%    +7.9%     2,259.4     41     ₹92,635     
  TITAN       9      CONSUMER_DISCRETIONARY  2.252    0.45   +111.8%   +11.2%    917.9       101    ₹92,705     
  CHOLAFIN    10     FINANCIAL_SERVICES    2.219    0.43   +49.0%    +14.9%    287.4       324    ₹93,113     
  AVANTIFEED  12     FMCG                  2.159    0.54   +240.1%   -5.2%     720.1       129    ₹92,891     
  INDUSINDBK  13     BANKING               2.123    0.31   +30.4%    +9.3%     1,720.1     54     ₹92,885     
  HDFC        14     FINANCIAL_SERVICES    2.081    0.40   +36.5%    +4.0%     450.6       206    ₹92,820     
  HDFCBANK    15     BANKING               2.081    0.40   +36.5%    +4.0%     450.6       206    ₹92,820     

  HOLDS (9)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    2      CONSUMER_DISCRETIONARY  01-Feb-18   200.8       231.1       494    ₹14,959       +15.1%    
  LTTS        7      IT                    01-Dec-17   994.3       1,119.2     100    ₹12,483       +12.6%    
  JINDALSTEL  22     CHEMICALS             01-Jan-18   203.3       221.9       507    ₹9,399        +9.1%     
  TECHM       5      IT                    01-Feb-18   462.4       484.6       214    ₹4,737        +4.8%     
  3MINDIA     25     CAPITAL_GOODS         01-Jan-18   17,896.8    18,555.8    5      ₹3,295        +3.7%     
  JSWSTEEL    16     METALS                01-Mar-18   278.9       270.7       342    ₹-2,794       -2.9%     
  MPHASIS     8      IT                    01-Feb-18   722.3       695.3       137    ₹-3,695       -3.7%     
  PFIZER      24     PHARMA                01-Feb-18   1,997.6     1,901.9     49     ₹-4,687       -4.8%     
  VBL         18     FMCG                  01-Feb-18   39.1        37.0        2535   ₹-5,325       -5.4%     

  AFTER: Invested ₹1,745,277 | Cash ₹116,469 | Total ₹1,861,746 | Positions 18/20 | Slot ₹93,137

========================================================================
  REBALANCE #06  —  02 May 2018
  NAV: ₹1,961,621  |  Slot: ₹98,081  |  Cash: ₹116,469
========================================================================

  EXITS (7)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LTTS        58     IT                    01-Dec-17   994.3       1,180.2     100    ₹18,585       +18.7%    152d  
  JINDALSTEL  —      CHEMICALS             01-Jan-18   203.3       235.1       507    ₹16,123       +15.6%    121d  
  INDUSINDBK  26     BANKING               02-Apr-18   1,720.1     1,780.8     54     ₹3,279        +3.5%     30d   
  HDFC        46     FINANCIAL_SERVICES    02-Apr-18   450.6       459.5       206    ₹1,841        +2.0%     30d   
  HDFCBANK    47     BANKING               02-Apr-18   450.6       459.5       206    ₹1,841        +2.0%     30d   
  VBL         89     FMCG                  01-Feb-18   39.1        38.6        2535   ₹-1,414       -1.4%     90d   
  PFIZER      61     PHARMA                01-Feb-18   1,997.6     1,955.0     49     ₹-2,085       -2.1%     90d   

  ENTRIES (8)
  [52w filter blocked 1: KIOCL(-52.6%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  PIDILITIND  2      CHEMICALS             3.184    0.48   +52.7%    +25.7%    530.4       184    ₹97,585     
  DMART       4      CONSUMER_DISCRETIONARY  3.133    0.28   +100.0%   +29.0%    1,495.8     65     ₹97,230     
  KOTAKBANK   10     BANKING               2.534    0.59   +39.7%    +15.1%    250.1       392    ₹98,024     
  AARTIIND    11     CHEMICALS             2.488    0.47   +62.7%    +23.3%    314.3       312    ₹98,059     
  HINDUNILVR  12     FMCG                  2.475    0.32   +58.8%    +7.2%     1,296.5     75     ₹97,237     
  SRF         13     CAPITAL_GOODS         2.457    0.74   +35.3%    +28.1%    455.9       215    ₹98,022     
  TCS         14     IT                    2.331    0.27   +54.4%    +12.1%    1,417.8     69     ₹97,832     
  M&M         15     AUTO                  2.187    0.62   +28.8%    +13.9%    802.1       122    ₹97,860     

  HOLDS (11)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    8      CONSUMER_DISCRETIONARY  01-Feb-18   200.8       250.1       494    ₹24,343       +24.5%    
  CHOLAFIN    7      FINANCIAL_SERVICES    02-Apr-18   287.4       333.6       324    ₹14,973       +16.1%    
  MPHASIS     6      IT                    01-Feb-18   722.3       824.0       137    ₹13,938       +14.1%    
  ASHOKLEY    3      AUTO                  02-Apr-18   63.4        69.1        1469   ₹8,390        +9.0%     
  TECHM       21     IT                    01-Feb-18   462.4       501.4       214    ₹8,341        +8.4%     
  3MINDIA     18     CAPITAL_GOODS         01-Jan-18   17,896.8    19,140.1    5      ₹6,217        +6.9%     
  BRITANNIA   5      FMCG                  02-Apr-18   2,259.4     2,400.1     41     ₹5,768        +6.2%     
  JSWSTEEL    24     METALS                01-Mar-18   278.9       287.7       342    ₹3,012        +3.2%     
  DBL         9      INFRASTRUCTURE        02-Apr-18   1,099.5     1,123.9     84     ₹2,047        +2.2%     
  TITAN       16     CONSUMER_DISCRETIONARY  02-Apr-18   917.9       937.7       101    ₹2,003        +2.2%     
  AVANTIFEED  17     FMCG                  02-Apr-18   720.1       728.2       129    ₹1,043        +1.1%     

  AFTER: Invested ₹1,910,674 | Cash ₹50,019 | Total ₹1,960,693 | Positions 19/20 | Slot ₹98,081

========================================================================
  REBALANCE #07  —  01 Jun 2018
  NAV: ₹1,891,965  |  Slot: ₹94,598  |  Cash: ₹50,019
========================================================================

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  3MINDIA     81     CAPITAL_GOODS         01-Jan-18   17,896.8    17,979.5    5      ₹413          +0.5%     151d  
  AARTIIND    47     CHEMICALS             02-May-18   314.3       290.2       312    ₹-7,506       -7.7%     30d   
  SRF         95     CAPITAL_GOODS         02-May-18   455.9       362.8       215    ₹-20,030      -20.4%    30d   
  DBL         55     INFRASTRUCTURE        02-Apr-18   1,099.5     847.7       84     ₹-21,149      -22.9%    60d   
  AVANTIFEED  183    FMCG                  02-Apr-18   720.1       466.8       129    ₹-32,678      -35.2%    60d   

  ENTRIES (4)
  [52w filter blocked 1: KIOCL(-58.7%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  DABUR       5      FMCG                  2.875    0.17   +39.8%    +19.0%    354.9       266    ₹94,416     
  COLPAL      12     FMCG                  2.756    0.12   +24.9%    +21.4%    1,055.6     89     ₹93,949     
  HDFC        13     FINANCIAL_SERVICES    2.694    0.53   +31.0%    +12.7%    495.6       190    ₹94,162     
  HDFCBANK    14     BANKING               2.694    0.53   +31.0%    +12.7%    495.6       190    ₹94,162     

  HOLDS (14)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    1      CONSUMER_DISCRETIONARY  01-Feb-18   200.8       245.3       494    ₹21,984       +22.2%    
  MPHASIS     6      IT                    01-Feb-18   722.3       868.5       137    ₹20,024       +20.2%    
  TECHM       7      IT                    01-Feb-18   462.4       530.6       214    ₹14,592       +14.7%    
  BRITANNIA   2      FMCG                  02-Apr-18   2,259.4     2,571.5     41     ₹12,795       +13.8%    
  JSWSTEEL    25     METALS                01-Mar-18   278.9       303.0       342    ₹8,241        +8.6%     
  HINDUNILVR  3      FMCG                  02-May-18   1,296.5     1,399.5     75     ₹7,722        +7.9%     
  CHOLAFIN    33     FINANCIAL_SERVICES    02-Apr-18   287.4       308.5       324    ₹6,831        +7.3%     
  KOTAKBANK   9      BANKING               02-May-18   250.1       262.2       392    ₹4,769        +4.9%     
  M&M         11     AUTO                  02-May-18   802.1       837.9       122    ₹4,359        +4.5%     
  DMART       8      CONSUMER_DISCRETIONARY  02-May-18   1,495.8     1,531.1     65     ₹2,291        +2.4%     
  ASHOKLEY    32     AUTO                  02-Apr-18   63.4        64.5        1469   ₹1,728        +1.9%     
  PIDILITIND  4      CHEMICALS             02-May-18   530.4       538.3       184    ₹1,460        +1.5%     
  TCS         19     IT                    02-May-18   1,417.8     1,415.4     69     ₹-168         -0.2%     
  TITAN       21     CONSUMER_DISCRETIONARY  02-Apr-18   917.9       875.0       101    ₹-4,330       -4.7%     

  AFTER: Invested ₹1,828,768 | Cash ₹62,750 | Total ₹1,891,518 | Positions 18/20 | Slot ₹94,598

========================================================================
  REBALANCE #08  —  02 Jul 2018
  NAV: ₹1,882,119  |  Slot: ₹94,106  |  Cash: ₹62,750
========================================================================

  [REGIME OFF] Nifty 200 9,109.0 < SMA200 9,211.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    1      CONSUMER_DISCRETIONARY  01-Feb-18   200.8       276.3       494    ₹37,303       +37.6%    
  MPHASIS     6      IT                    01-Feb-18   722.3       903.3       137    ₹24,805       +25.1%    
  BRITANNIA   2      FMCG                  02-Apr-18   2,259.4     2,757.5     41     ₹20,422       +22.0%    
  HINDUNILVR  3      FMCG                  02-May-18   1,296.5     1,458.5     75     ₹12,148       +12.5%    
  TECHM       15     IT                    01-Feb-18   462.4       497.4       214    ₹7,474        +7.6%     
  TCS         4      IT                    02-May-18   1,417.8     1,512.6     69     ₹6,535        +6.7%     
  KOTAKBANK   7      BANKING               02-May-18   250.1       266.2       392    ₹6,325        +6.5%     
  M&M         16     AUTO                  02-May-18   802.1       824.0       122    ₹2,664        +2.7%     
  JSWSTEEL    28     METALS                01-Mar-18   278.9       286.1       342    ₹2,466        +2.6%     
  CHOLAFIN    49     FINANCIAL_SERVICES    02-Apr-18   287.4       294.6       324    ₹2,335        +2.5%     
  DMART       11     CONSUMER_DISCRETIONARY  02-May-18   1,495.8     1,525.9     65     ₹1,953        +2.0%     
  DABUR       13     FMCG                  01-Jun-18   354.9       352.1       266    ₹-745         -0.8%     
  HDFC        17     FINANCIAL_SERVICES    01-Jun-18   495.6       486.8       190    ₹-1,666       -1.8%     
  HDFCBANK    18     BANKING               01-Jun-18   495.6       486.8       190    ₹-1,666       -1.8%     
  PIDILITIND  21     CHEMICALS             02-May-18   530.4       509.6       184    ₹-3,818       -3.9%     
  TITAN       31     CONSUMER_DISCRETIONARY  02-Apr-18   917.9       872.5       101    ₹-4,586       -4.9%     
  COLPAL      41     FMCG                  01-Jun-18   1,055.6     988.4       89     ₹-5,978       -6.4%     
  ASHOKLEY    85     AUTO                  02-Apr-18   63.4        54.5        1469   ₹-13,073      -14.0%    

  AFTER: Invested ₹1,819,369 | Cash ₹62,750 | Total ₹1,882,119 | Positions 18/20 | Slot ₹94,106

========================================================================
  REBALANCE #09  —  01 Aug 2018
  NAV: ₹1,958,068  |  Slot: ₹97,903  |  Cash: ₹62,750
========================================================================

  EXITS (7)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  JSWSTEEL    41     METALS                01-Mar-18   278.9       302.1       342    ₹7,948        +8.3%     153d  
  KOTAKBANK   35     BANKING               02-May-18   250.1       261.3       392    ₹4,408        +4.5%     91d   
  PIDILITIND  28     CHEMICALS             02-May-18   530.4       543.6       184    ₹2,442        +2.5%     91d   
  CHOLAFIN    98     FINANCIAL_SERVICES    02-Apr-18   287.4       285.5       324    ₹-625         -0.7%     121d  
  TITAN       44     CONSUMER_DISCRETIONARY  02-Apr-18   917.9       895.8       101    ₹-2,232       -2.4%     121d  
  COLPAL      69     FMCG                  01-Jun-18   1,055.6     950.4       89     ₹-9,368       -10.0%    61d   
  ASHOKLEY    161    AUTO                  02-Apr-18   63.4        51.8        1469   ₹-16,954      -18.2%    121d  

  ENTRIES (7)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  BAJFINANCE  2      FINANCIAL_SERVICES    3.384    0.74   +60.3%    +46.8%    265.6       368    ₹97,755     
  GLAXO       3      PHARMA                3.342    0.31   +32.0%    +36.8%    1,336.7     73     ₹97,582     
  SANOFI      6      PHARMA                3.073    0.36   +42.6%    +23.9%    4,304.8     22     ₹94,706     
  RELIANCE    7      OIL_GAS               3.054    0.79   +50.6%    +25.8%    530.2       184    ₹97,560     
  PAGEIND     9      TEXTILES              2.809    0.68   +82.6%    +26.9%    27,445.9    3      ₹82,338     
  BAJAJFINSV  11     FINANCIAL_SERVICES    2.742    0.82   +41.9%    +30.7%    696.8       140    ₹97,556     
  BATAINDIA   12     CONSUMER_DISCRETIONARY  2.720    0.68   +64.4%    +21.6%    878.9       111    ₹97,563     

  HOLDS (11)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  MPHASIS     5      IT                    01-Feb-18   722.3       1,006.5     137    ₹38,935       +39.3%    
  JUBLFOOD    10     CONSUMER_DISCRETIONARY  01-Feb-18   200.8       273.8       494    ₹36,049       +36.3%    
  BRITANNIA   1      FMCG                  02-Apr-18   2,259.4     2,882.3     41     ₹25,540       +27.6%    
  HINDUNILVR  4      FMCG                  02-May-18   1,296.5     1,539.1     75     ₹18,195       +18.7%    
  TCS         8      IT                    02-May-18   1,417.8     1,618.1     69     ₹13,814       +14.1%    
  TECHM       20     IT                    01-Feb-18   462.4       527.2       214    ₹13,847       +14.0%    
  DABUR       13     FMCG                  01-Jun-18   354.9       403.3       266    ₹12,850       +13.6%    
  DMART       16     CONSUMER_DISCRETIONARY  02-May-18   1,495.8     1,669.7     65     ₹11,300       +11.6%    
  M&M         27     AUTO                  02-May-18   802.1       871.0       122    ₹8,407        +8.6%     
  HDFC        24     FINANCIAL_SERVICES    01-Jun-18   495.6       506.9       190    ₹2,146        +2.3%     
  HDFCBANK    25     BANKING               01-Jun-18   495.6       506.9       190    ₹2,146        +2.3%     

  AFTER: Invested ₹1,910,932 | Cash ₹46,346 | Total ₹1,957,278 | Positions 18/20 | Slot ₹97,903

========================================================================
  REBALANCE #10  —  03 Sep 2018
  NAV: ₹2,027,149  |  Slot: ₹101,357  |  Cash: ₹46,346
========================================================================
  [SECTOR CAP≤4] dropped: LTTS

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  M&M         47     AUTO                  02-May-18   802.1       888.0       122    ₹10,470       +10.7%    124d  
  DMART       53     CONSUMER_DISCRETIONARY  02-May-18   1,495.8     1,604.7     65     ₹7,072        +7.3%     124d  
  HDFC        77     FINANCIAL_SERVICES    01-Jun-18   495.6       487.2       190    ₹-1,586       -1.7%     94d   
  HDFCBANK    78     BANKING               01-Jun-18   495.6       487.2       190    ₹-1,586       -1.7%     94d   
  BAJAJFINSV  54     FINANCIAL_SERVICES    01-Aug-18   696.8       663.2       140    ₹-4,707       -4.8%     33d   

  ENTRIES (5)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  PFIZER      3      PHARMA                3.146    0.43   +97.6%    +36.3%    3,065.4     33     ₹101,159    
  3MINDIA     8      CAPITAL_GOODS         2.819    0.44   +76.1%    +38.1%    24,137.1    4      ₹96,548     
  INFY        9      IT                    2.730    0.30   +60.8%    +18.6%    590.4       171    ₹100,951    
  GODREJCP    10     FMCG                  2.709    0.30   +56.4%    +26.5%    894.1       113    ₹101,035    
  HAVELLS     16     CAPITAL_GOODS         2.493    0.97   +48.7%    +34.5%    680.5       148    ₹100,719    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    6      CONSUMER_DISCRETIONARY  01-Feb-18   200.8       299.4       494    ₹48,699       +49.1%    
  MPHASIS     14     IT                    01-Feb-18   722.3       1,042.5     137    ₹43,862       +44.3%    
  BRITANNIA   13     FMCG                  02-Apr-18   2,259.4     2,916.1     41     ₹26,924       +29.1%    
  TECHM       25     IT                    01-Feb-18   462.4       577.1       214    ₹24,538       +24.8%    
  DABUR       12     FMCG                  01-Jun-18   354.9       435.1       266    ₹21,312       +22.6%    
  TCS         11     IT                    02-May-18   1,417.8     1,680.6     69     ₹18,130       +18.5%    
  HINDUNILVR  27     FMCG                  02-May-18   1,296.5     1,507.9     75     ₹15,852       +16.3%    
  PAGEIND     7      TEXTILES              01-Aug-18   27,445.9    31,213.8    3      ₹11,304       +13.7%    
  BATAINDIA   5      CONSUMER_DISCRETIONARY  01-Aug-18   878.9       993.8       111    ₹12,745       +13.1%    
  GLAXO       2      PHARMA                01-Aug-18   1,336.7     1,456.6     73     ₹8,754        +9.0%     
  SANOFI      1      PHARMA                01-Aug-18   4,304.8     4,680.3     22     ₹8,261        +8.7%     
  RELIANCE    4      OIL_GAS               01-Aug-18   530.2       546.6       184    ₹3,011        +3.1%     
  BAJFINANCE  18     FINANCIAL_SERVICES    01-Aug-18   265.6       265.8       368    ₹45           +0.0%     

  AFTER: Invested ₹1,990,581 | Cash ₹35,973 | Total ₹2,026,554 | Positions 18/20 | Slot ₹101,357

========================================================================
  REBALANCE #11  —  01 Oct 2018
  NAV: ₹1,869,708  |  Slot: ₹93,485  |  Cash: ₹35,973
========================================================================
  [SECTOR CAP≤4] dropped: LTTS, WIPRO

  [REGIME OFF] Nifty 200 9,165.5 < SMA200 9,384.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  MPHASIS     10     IT                    01-Feb-18   722.3       968.9       137    ₹33,785       +34.1%    
  TCS         1      IT                    02-May-18   1,417.8     1,846.5     69     ₹29,577       +30.2%    
  TECHM       9      IT                    01-Feb-18   462.4       587.9       214    ₹26,840       +27.1%    
  JUBLFOOD    32     CONSUMER_DISCRETIONARY  01-Feb-18   200.8       243.8       494    ₹21,256       +21.4%    
  DABUR       13     FMCG                  01-Jun-18   354.9       409.8       266    ₹14,587       +15.5%    
  BRITANNIA   45     FMCG                  02-Apr-18   2,259.4     2,595.3     41     ₹13,772       +14.9%    
  HINDUNILVR  34     FMCG                  02-May-18   1,296.5     1,457.5     75     ₹12,072       +12.4%    
  PAGEIND     8      TEXTILES              01-Aug-18   27,445.9    30,581.8    3      ₹9,408        +11.4%    
  SANOFI      5      PHARMA                01-Aug-18   4,304.8     4,484.0     22     ₹3,941        +4.2%     
  INFY        2      IT                    03-Sep-18   590.4       614.7       171    ₹4,156        +4.1%     
  RELIANCE    4      OIL_GAS               01-Aug-18   530.2       547.7       184    ₹3,220        +3.3%     
  BATAINDIA   21     CONSUMER_DISCRETIONARY  01-Aug-18   878.9       903.9       111    ₹2,773        +2.8%     
  GLAXO       44     PHARMA                01-Aug-18   1,336.7     1,217.7     73     ₹-8,689       -8.9%     
  3MINDIA     15     CAPITAL_GOODS         03-Sep-18   24,137.1    21,448.9    4      ₹-10,752      -11.1%    
  GODREJCP    55     FMCG                  03-Sep-18   894.1       745.4       113    ₹-16,804      -16.6%    
  PFIZER      17     PHARMA                03-Sep-18   3,065.4     2,549.0     33     ₹-17,040      -16.8%    
  HAVELLS     37     CAPITAL_GOODS         03-Sep-18   680.5       565.4       148    ₹-17,047      -16.9%    
  BAJFINANCE  77     FINANCIAL_SERVICES    01-Aug-18   265.6       215.5       368    ₹-18,465      -18.9%    

  AFTER: Invested ₹1,833,734 | Cash ₹35,973 | Total ₹1,869,708 | Positions 18/20 | Slot ₹93,485

========================================================================
  REBALANCE #12  —  01 Nov 2018
  NAV: ₹1,739,595  |  Slot: ₹86,980  |  Cash: ₹35,973
========================================================================
  [SECTOR CAP≤4] dropped: WIPRO, AUROPHARMA

  [REGIME OFF] Nifty 200 8,772.5 < SMA200 9,309.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TECHM       6      IT                    01-Feb-18   462.4       557.8       214    ₹20,406       +20.6%    
  MPHASIS     38     IT                    01-Feb-18   722.3       812.7       137    ₹12,385       +12.5%    
  TCS         13     IT                    02-May-18   1,417.8     1,588.0     69     ₹11,743       +12.0%    
  HINDUNILVR  27     FMCG                  02-May-18   1,296.5     1,434.8     75     ₹10,376       +10.7%    
  BRITANNIA   68     FMCG                  02-Apr-18   2,259.4     2,479.6     41     ₹9,028        +9.7%     
  JUBLFOOD    80     CONSUMER_DISCRETIONARY  01-Feb-18   200.8       217.0       494    ₹8,024        +8.1%     
  BATAINDIA   17     CONSUMER_DISCRETIONARY  01-Aug-18   878.9       948.3       111    ₹7,697        +7.9%     
  SANOFI      12     PHARMA                01-Aug-18   4,304.8     4,245.4     22     ₹-1,307       -1.4%     
  DABUR       47     FMCG                  01-Jun-18   354.9       345.8       266    ₹-2,436       -2.6%     
  PAGEIND     26     TEXTILES              01-Aug-18   27,445.9    26,278.7    3      ₹-3,502       -4.3%     
  INFY        16     IT                    03-Sep-18   590.4       554.8       171    ₹-6,076       -6.0%     
  HAVELLS     23     CAPITAL_GOODS         03-Sep-18   680.5       612.7       148    ₹-10,038      -10.0%    
  GLAXO       76     PHARMA                01-Aug-18   1,336.7     1,193.9     73     ₹-10,426      -10.7%    
  BAJFINANCE  —      FINANCIAL_SERVICES    01-Aug-18   265.6       235.9       368    ₹-10,939      -11.2%    
  RELIANCE    60     OIL_GAS               01-Aug-18   530.2       469.7       184    ₹-11,136      -11.4%    
  PFIZER      9      PHARMA                03-Sep-18   3,065.4     2,512.3     33     ₹-18,254      -18.0%    
  3MINDIA     54     CAPITAL_GOODS         03-Sep-18   24,137.1    18,484.4    4      ₹-22,611      -23.4%    
  GODREJCP    104    FMCG                  03-Sep-18   894.1       660.0       113    ₹-26,456      -26.2% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): GODREJCP

  AFTER: Invested ₹1,703,622 | Cash ₹35,973 | Total ₹1,739,595 | Positions 18/20 | Slot ₹86,980

========================================================================
  REBALANCE #13  —  03 Dec 2018
  NAV: ₹1,834,677  |  Slot: ₹91,734  |  Cash: ₹35,973
========================================================================

  [REGIME OFF] Nifty 200 9,126.9 < SMA200 9,242.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    43     CONSUMER_DISCRETIONARY  01-Feb-18   200.8       261.4       494    ₹29,915       +30.2%    
  HINDUNILVR  3      FMCG                  02-May-18   1,296.5     1,629.4     75     ₹24,966       +25.7%    
  BRITANNIA   50     FMCG                  02-Apr-18   2,259.4     2,748.7     41     ₹20,062       +21.7%    
  TECHM       20     IT                    01-Feb-18   462.4       551.6       214    ₹19,081       +19.3%    
  MPHASIS     65     IT                    01-Feb-18   722.3       834.0       137    ₹15,309       +15.5%    
  TCS         12     IT                    02-May-18   1,417.8     1,626.3     69     ₹14,383       +14.7%    
  BATAINDIA   22     CONSUMER_DISCRETIONARY  01-Aug-18   878.9       993.5       111    ₹12,714       +13.0%    
  DABUR       73     FMCG                  01-Jun-18   354.9       386.6       266    ₹8,409        +8.9%     
  SANOFI      7      PHARMA                01-Aug-18   4,304.8     4,503.1     22     ₹4,362        +4.6%     
  RELIANCE    —      OIL_GAS               01-Aug-18   530.2       514.3       184    ₹-2,937       -3.0%     
  HAVELLS     33     CAPITAL_GOODS         03-Sep-18   680.5       657.5       148    ₹-3,416       -3.4%     
  INFY        23     IT                    03-Sep-18   590.4       557.9       171    ₹-5,557       -5.5%     
  BAJFINANCE  —      FINANCIAL_SERVICES    01-Aug-18   265.6       244.7       368    ₹-7,688       -7.9%     
  PAGEIND     96     TEXTILES              01-Aug-18   27,445.9    24,450.9    3      ₹-8,985       -10.9% ⚠  
  GLAXO       101    PHARMA                01-Aug-18   1,336.7     1,159.2     73     ₹-12,957      -13.3% ⚠  
  3MINDIA     58     CAPITAL_GOODS         03-Sep-18   24,137.1    20,347.2    4      ₹-15,160      -15.7%    
  GODREJCP    86     FMCG                  03-Sep-18   894.1       725.9       113    ₹-19,010      -18.8% ⚠  
  PFIZER      77     PHARMA                03-Sep-18   3,065.4     2,400.9     33     ₹-21,930      -21.7% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): PFIZER, GODREJCP, PAGEIND, GLAXO

  AFTER: Invested ₹1,798,704 | Cash ₹35,973 | Total ₹1,834,677 | Positions 18/20 | Slot ₹91,734

========================================================================
  REBALANCE #14  —  01 Jan 2019
  NAV: ₹1,840,000  |  Slot: ₹92,000  |  Cash: ₹35,973
========================================================================

  [REGIME OFF] Nifty 200 9,197.9 < SMA200 9,229.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  HINDUNILVR  3      FMCG                  02-May-18   1,296.5     1,607.5     75     ₹23,326       +24.0%    
  JUBLFOOD    38     CONSUMER_DISCRETIONARY  01-Feb-18   200.8       245.8       494    ₹22,236       +22.4%    
  BRITANNIA   7      FMCG                  02-Apr-18   2,259.4     2,756.3     41     ₹20,373       +22.0%    
  TECHM       32     IT                    01-Feb-18   462.4       556.6       214    ₹20,141       +20.4%    
  BATAINDIA   4      CONSUMER_DISCRETIONARY  01-Aug-18   878.9       1,051.6     111    ₹19,167       +19.6%    
  MPHASIS     84     IT                    01-Feb-18   722.3       840.0       137    ₹16,127       +16.3%    
  DABUR       66     FMCG                  01-Jun-18   354.9       394.9       266    ₹10,625       +11.3%    
  TCS         47     IT                    02-May-18   1,417.8     1,561.0     69     ₹9,877        +10.1%    
  SANOFI      10     PHARMA                01-Aug-18   4,304.8     4,568.2     22     ₹5,794        +6.1%     
  BAJFINANCE  6      FINANCIAL_SERVICES    01-Aug-18   265.6       259.2       368    ₹-2,368       -2.4%     
  HAVELLS     19     CAPITAL_GOODS         03-Sep-18   680.5       660.0       148    ₹-3,037       -3.0%     
  GLAXO       51     PHARMA                01-Aug-18   1,336.7     1,257.5     73     ₹-5,788       -5.9%     
  RELIANCE    96     OIL_GAS               01-Aug-18   530.2       498.5       184    ₹-5,838       -6.0%     
  INFY        48     IT                    03-Sep-18   590.4       553.4       171    ₹-6,311       -6.3%     
  GODREJCP    42     FMCG                  03-Sep-18   894.1       759.7       113    ₹-15,188      -15.0%    
  PAGEIND     193    TEXTILES              01-Aug-18   27,445.9    22,927.0    3      ₹-13,557      -16.5% ⚠  
  PFIZER      43     PHARMA                03-Sep-18   3,065.4     2,491.3     33     ₹-18,947      -18.7%    
  3MINDIA     89     CAPITAL_GOODS         03-Sep-18   24,137.1    19,199.1    4      ₹-19,752      -20.5%    
  ⚠  WAZ < 0 (momentum below universe mean): PAGEIND

  AFTER: Invested ₹1,804,027 | Cash ₹35,973 | Total ₹1,840,000 | Positions 18/20 | Slot ₹92,000

========================================================================
  REBALANCE #15  —  01 Feb 2019
  NAV: ₹1,895,048  |  Slot: ₹94,752  |  Cash: ₹35,973
========================================================================
  [SECTOR CAP≤4] dropped: MARICO

  [REGIME OFF] Nifty 200 9,056.3 < SMA200 9,229.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  JUBLFOOD    15     CONSUMER_DISCRETIONARY  01-Feb-18   200.8       266.9       494    ₹32,633       +32.9%    
  BRITANNIA   1      FMCG                  02-Apr-18   2,259.4     2,887.7     41     ₹25,761       +27.8%    
  TECHM       41     IT                    01-Feb-18   462.4       577.8       214    ₹24,695       +25.0%    
  HINDUNILVR  7      FMCG                  02-May-18   1,296.5     1,605.9     75     ₹23,205       +23.9%    
  BATAINDIA   4      CONSUMER_DISCRETIONARY  01-Aug-18   878.9       1,068.3     111    ₹21,023       +21.5%    
  DABUR       6      FMCG                  01-Jun-18   354.9       423.0       266    ₹18,095       +19.2%    
  TCS         23     IT                    02-May-18   1,417.8     1,668.9     69     ₹17,321       +17.7%    
  MPHASIS     48     IT                    01-Feb-18   722.3       843.5       137    ₹16,602       +16.8%    
  SANOFI      2      PHARMA                01-Aug-18   4,304.8     4,707.5     22     ₹8,858        +9.4%     
  INFY        8      IT                    03-Sep-18   590.4       633.5       171    ₹7,373        +7.3%     
  RELIANCE    9      OIL_GAS               01-Aug-18   530.2       555.8       184    ₹4,713        +4.8%     
  HAVELLS     17     CAPITAL_GOODS         03-Sep-18   680.5       700.2       148    ₹2,917        +2.9%     
  BAJFINANCE  13     FINANCIAL_SERVICES    01-Aug-18   265.6       256.4       368    ₹-3,389       -3.5%     
  GLAXO       58     PHARMA                01-Aug-18   1,336.7     1,181.5     73     ₹-11,330      -11.6%    
  PFIZER      29     PHARMA                03-Sep-18   3,065.4     2,636.7     33     ₹-14,146      -14.0%    
  PAGEIND     126    TEXTILES              01-Aug-18   27,445.9    22,181.3    3      ₹-15,794      -19.2% ⚠  
  3MINDIA     65     CAPITAL_GOODS         03-Sep-18   24,137.1    18,852.7    4      ₹-21,137      -21.9%    
  GODREJCP    90     FMCG                  03-Sep-18   894.1       668.7       113    ₹-25,470      -25.2%    
  ⚠  WAZ < 0 (momentum below universe mean): PAGEIND

  AFTER: Invested ₹1,859,075 | Cash ₹35,973 | Total ₹1,895,048 | Positions 18/20 | Slot ₹94,752

========================================================================
  REBALANCE #16  —  01 Mar 2019
  NAV: ₹1,891,292  |  Slot: ₹94,565  |  Cash: ₹35,973
========================================================================
  [SECTOR CAP≤4] dropped: PFC

  [REGIME OFF] Nifty 200 9,037.5 < SMA200 9,188.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TECHM       6      IT                    01-Feb-18   462.4       643.8       214    ₹38,807       +39.2%    
  BATAINDIA   1      CONSUMER_DISCRETIONARY  01-Aug-18   878.9       1,195.4     111    ₹35,131       +36.0%    
  JUBLFOOD    44     CONSUMER_DISCRETIONARY  01-Feb-18   200.8       250.9       494    ₹24,750       +25.0%    
  BRITANNIA   45     FMCG                  02-Apr-18   2,259.4     2,733.4     41     ₹19,434       +21.0%    
  MPHASIS     56     IT                    01-Feb-18   722.3       863.1       137    ₹19,292       +19.5%    
  HINDUNILVR  33     FMCG                  02-May-18   1,296.5     1,548.2     75     ₹18,878       +19.4%    
  DABUR       12     FMCG                  01-Jun-18   354.9       416.8       266    ₹16,464       +17.4%    
  TCS         29     IT                    02-May-18   1,417.8     1,640.5     69     ₹15,361       +15.7%    
  INFY        8      IT                    03-Sep-18   590.4       620.8       171    ₹5,205        +5.2%     
  RELIANCE    22     OIL_GAS               01-Aug-18   530.2       545.2       184    ₹2,757        +2.8%     
  HAVELLS     19     CAPITAL_GOODS         03-Sep-18   680.5       672.1       148    ₹-1,244       -1.2%     
  SANOFI      54     PHARMA                01-Aug-18   4,304.8     4,234.8     22     ₹-1,540       -1.6%     
  BAJFINANCE  11     FINANCIAL_SERVICES    01-Aug-18   265.6       259.6       368    ₹-2,228       -2.3%     
  PFIZER      4      PHARMA                03-Sep-18   3,065.4     2,862.5     33     ₹-6,696       -6.6%     
  3MINDIA     48     CAPITAL_GOODS         03-Sep-18   24,137.1    21,377.7    4      ₹-11,037      -11.4%    
  GLAXO       80     PHARMA                01-Aug-18   1,336.7     1,115.8     73     ₹-16,125      -16.5%    
  PAGEIND     167    TEXTILES              01-Aug-18   27,445.9    20,638.6    3      ₹-20,422      -24.8% ⚠  
  GODREJCP    140    FMCG                  03-Sep-18   894.1       640.9       113    ₹-28,612      -28.3% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): GODREJCP, PAGEIND

  AFTER: Invested ₹1,855,319 | Cash ₹35,973 | Total ₹1,891,292 | Positions 18/20 | Slot ₹94,565

========================================================================
  REBALANCE #17  —  01 Apr 2019
  NAV: ₹1,942,945  |  Slot: ₹97,147  |  Cash: ₹35,973
========================================================================

  EXITS (9)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TECHM       43     IT                    01-Feb-18   462.4       608.0       214    ₹31,138       +31.5%    424d  
  BRITANNIA   74     FMCG                  02-Apr-18   2,259.4     2,709.0     41     ₹18,436       +19.9%    364d  
  HINDUNILVR  97     FMCG                  02-May-18   1,296.5     1,508.3     75     ₹15,889       +16.3%    334d  
  MPHASIS     107    IT                    01-Feb-18   722.3       824.3       137    ₹13,970       +14.1%    424d  
  DABUR       115    FMCG                  01-Jun-18   354.9       375.3       266    ₹5,421        +5.7%     304d  
  SANOFI      138    PHARMA                01-Aug-18   4,304.8     4,202.2     22     ₹-2,258       -2.4%     243d  
  PAGEIND     72     TEXTILES              01-Aug-18   27,445.9    23,686.5    3      ₹-11,278      -13.7%    243d  
  GLAXO       174    PHARMA                01-Aug-18   1,336.7     1,090.9     73     ₹-17,945      -18.4%    243d  
  GODREJCP    205    FMCG                  03-Sep-18   894.1       638.9       113    ₹-28,842      -28.5%    210d  

  ENTRIES (9)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  IPCALAB     3      PHARMA                2.785    0.44   +46.5%    +24.1%    474.3       204    ₹96,749     
  AXISBANK    4      BANKING               2.716    0.16   +45.5%    +24.0%    761.5       127    ₹96,707     
  TITAN       6      CONSUMER_DISCRETIONARY  2.625    -0.06  +29.0%    +26.2%    1,094.0     88     ₹96,272     
  MANAPPURAM  7      FINANCIAL_SERVICES    2.599    0.23   +22.5%    +39.6%    109.8       884    ₹97,081     
  VBL         8      FMCG                  2.565    0.29   +46.0%    +21.6%    52.0        1868   ₹97,103     
  RBLBANK     9      BANKING               2.563    0.45   +47.7%    +17.8%    659.7       147    ₹96,973     
  MUTHOOTFIN  10     FINANCIAL_SERVICES    2.554    0.08   +55.7%    +24.5%    537.4       180    ₹96,729     
  DIVISLAB    11     PHARMA                2.550    0.02   +59.9%    +17.1%    1,641.9     59     ₹96,870     
  TORNTPHARM  13     PHARMA                2.485    0.15   +55.0%    +10.0%    868.5       111    ₹96,402     

  HOLDS (9)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BATAINDIA   1      CONSUMER_DISCRETIONARY  01-Aug-18   878.9       1,295.6     111    ₹46,254       +47.4%    
  JUBLFOOD    36     CONSUMER_DISCRETIONARY  01-Feb-18   200.8       287.0       494    ₹42,569       +42.9%    
  TCS         25     IT                    02-May-18   1,417.8     1,670.3     69     ₹17,418       +17.8%    
  RELIANCE    2      OIL_GAS               01-Aug-18   530.2       618.9       184    ₹16,323       +16.7%    
  BAJFINANCE  5      FINANCIAL_SERVICES    01-Aug-18   265.6       292.8       368    ₹10,004       +10.2%    
  HAVELLS     18     CAPITAL_GOODS         03-Sep-18   680.5       736.8       148    ₹8,330        +8.3%     
  INFY        15     IT                    03-Sep-18   590.4       631.8       171    ₹7,094        +7.0%     
  PFIZER      12     PHARMA                03-Sep-18   3,065.4     2,870.6     33     ₹-6,427       -6.4%     
  3MINDIA     31     CAPITAL_GOODS         03-Sep-18   24,137.1    22,569.9    4      ₹-6,269       -6.5%     

  AFTER: Invested ₹1,895,460 | Cash ₹46,451 | Total ₹1,941,911 | Positions 18/20 | Slot ₹97,147

========================================================================
  REBALANCE #18  —  02 May 2019
  NAV: ₹1,923,890  |  Slot: ₹96,194  |  Cash: ₹46,451
========================================================================

  EXITS (6)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  JUBLFOOD    58     CONSUMER_DISCRETIONARY  01-Feb-18   200.8       264.5       494    ₹31,463       +31.7%    455d  
  INFY        40     IT                    03-Sep-18   590.4       611.5       171    ₹3,617        +3.6%     241d  
  MUTHOOTFIN  34     FINANCIAL_SERVICES    01-Apr-19   537.4       531.1       180    ₹-1,136       -1.2%     31d   
  MANAPPURAM  55     FINANCIAL_SERVICES    01-Apr-19   109.8       103.1       884    ₹-5,905       -6.1%     31d   
  TORNTPHARM  90     PHARMA                01-Apr-19   868.5       808.6       111    ₹-6,649       -6.9%     31d   
  3MINDIA     54     CAPITAL_GOODS         03-Sep-18   24,137.1    21,465.0    4      ₹-10,688      -11.1%    241d  

  ENTRIES (6)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  ICICIGI     2      FINANCIAL_SERVICES    3.048    0.20   +43.7%    +32.7%    1,057.0     91     ₹96,190     
  GILLETTE    3      FMCG                  2.984    0.08   +15.6%    +16.3%    6,727.6     14     ₹94,186     
  RELAXO      4      CONSUMER_DISCRETIONARY  2.830    0.23   +31.4%    +20.5%    434.5       221    ₹96,019     
  HDFC        9      FINANCIAL_SERVICES    2.643    0.24   +22.3%    +11.9%    553.2       173    ₹95,697     
  HDFCBANK    10     BANKING               2.643    0.24   +22.3%    +11.9%    553.2       173    ₹95,697     
  BAJAJFINSV  13     FINANCIAL_SERVICES    2.408    0.03   +40.8%    +16.7%    755.5       127    ₹95,950     

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BATAINDIA   1      CONSUMER_DISCRETIONARY  01-Aug-18   878.9       1,369.8     111    ₹54,485       +55.8%    
  TCS         5      IT                    02-May-18   1,417.8     1,821.3     69     ₹27,841       +28.5%    
  RELIANCE    8      OIL_GAS               01-Aug-18   530.2       624.8       184    ₹17,404       +17.8%    
  BAJFINANCE  7      FINANCIAL_SERVICES    01-Aug-18   265.6       305.5       368    ₹14,682       +15.0%    
  HAVELLS     24     CAPITAL_GOODS         03-Sep-18   680.5       732.4       148    ₹7,676        +7.6%     
  TITAN       21     CONSUMER_DISCRETIONARY  01-Apr-19   1,094.0     1,111.4     88     ₹1,530        +1.6%     
  IPCALAB     6      PHARMA                01-Apr-19   474.3       474.4       204    ₹25           +0.0%     
  DIVISLAB    18     PHARMA                01-Apr-19   1,641.9     1,633.3     59     ₹-506         -0.5%     
  VBL         26     FMCG                  01-Apr-19   52.0        51.2        1868   ₹-1,451       -1.5%     
  AXISBANK    11     BANKING               01-Apr-19   761.5       748.3       127    ₹-1,674       -1.7%     
  RBLBANK     23     BANKING               01-Apr-19   659.7       642.8       147    ₹-2,475       -2.6%     
  PFIZER      12     PHARMA                03-Sep-18   3,065.4     2,733.2     33     ₹-10,964      -10.8%    

  AFTER: Invested ₹1,853,573 | Cash ₹69,636 | Total ₹1,923,209 | Positions 18/20 | Slot ₹96,194

========================================================================
  REBALANCE #19  —  03 Jun 2019
  NAV: ₹1,965,275  |  Slot: ₹98,264  |  Cash: ₹69,636
========================================================================

  EXITS (7)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TCS         38     IT                    02-May-18   1,417.8     1,843.5     69     ₹29,367       +30.0%    397d  
  RELIANCE    35     OIL_GAS               01-Aug-18   530.2       604.9       184    ₹13,734       +14.1%    306d  
  HAVELLS     44     CAPITAL_GOODS         03-Sep-18   680.5       731.4       148    ₹7,529        +7.5%     273d  
  VBL         34     FMCG                  01-Apr-19   52.0        54.8        1868   ₹5,263        +5.4%     63d   
  RELAXO      45     CONSUMER_DISCRETIONARY  02-May-19   434.5       416.0       221    ₹-4,084       -4.3%     32d   
  DIVISLAB    70     PHARMA                01-Apr-19   1,641.9     1,539.9     59     ₹-6,018       -6.2%     63d   
  PFIZER      82     PHARMA                03-Sep-18   3,065.4     2,716.4     33     ₹-11,518      -11.4%    273d  

  ENTRIES (8)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  NAUKRI      2      TELECOM               3.007    0.05   +97.5%    +39.0%    451.4       217    ₹97,951     
  HONAUT      3      CAPITAL_GOODS         2.861    0.33   +44.5%    +22.9%    26,282.6    3      ₹78,848     
  GUJGASLTD   4      ENERGY                2.770    0.57   +10.2%    +55.1%    177.2       554    ₹98,157     
  INDIGO      6      CAPITAL_GOODS         2.683    0.33   +49.2%    +51.3%    1,683.2     58     ₹97,627     
  ATUL        7      CHEMICALS             2.678    0.11   +42.0%    +19.8%    3,884.6     25     ₹97,115     
  SHREECEM    8      CEMENT                2.651    0.55   +34.2%    +32.5%    21,301.5    4      ₹85,206     
  SBIN        10     BANKING               2.593    0.49   +40.0%    +31.6%    322.2       305    ₹98,263     
  ABB         11     CAPITAL_GOODS         2.549    0.28   +32.3%    +28.3%    1,401.9     70     ₹98,132     

  HOLDS (11)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BATAINDIA   17     CONSUMER_DISCRETIONARY  01-Aug-18   878.9       1,272.7     111    ₹43,710       +44.8%    
  BAJFINANCE  5      FINANCIAL_SERVICES    01-Aug-18   265.6       342.5       368    ₹28,300       +28.9%    
  TITAN       18     CONSUMER_DISCRETIONARY  01-Apr-19   1,094.0     1,236.5     88     ₹12,537       +13.0%    
  ICICIGI     1      FINANCIAL_SERVICES    02-May-19   1,057.0     1,168.4     91     ₹10,131       +10.5%    
  BAJAJFINSV  9      FINANCIAL_SERVICES    02-May-19   755.5       832.2       127    ₹9,739        +10.1%    
  AXISBANK    14     BANKING               01-Apr-19   761.5       808.3       127    ₹5,943        +6.1%     
  HDFC        12     FINANCIAL_SERVICES    02-May-19   553.2       576.9       173    ₹4,115        +4.3%     
  HDFCBANK    13     BANKING               02-May-19   553.2       576.9       173    ₹4,115        +4.3%     
  RBLBANK     30     BANKING               01-Apr-19   659.7       675.0       147    ₹2,254        +2.3%     
  GILLETTE    21     FMCG                  02-May-19   6,727.6     6,604.1     14     ₹-1,728       -1.8%     
  IPCALAB     33     PHARMA                01-Apr-19   474.3       451.0       204    ₹-4,749       -4.9%     

  AFTER: Invested ₹1,925,403 | Cash ₹38,980 | Total ₹1,964,383 | Positions 19/20 | Slot ₹98,264

========================================================================
  REBALANCE #20  —  01 Jul 2019
  NAV: ₹1,954,317  |  Slot: ₹97,716  |  Cash: ₹38,980
========================================================================

  EXITS (6)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SBIN        24     BANKING               03-Jun-19   322.2       327.7       305    ₹1,686        +1.7%     28d   
  ICICIGI     25     FINANCIAL_SERVICES    02-May-19   1,057.0     1,059.7     91     ₹241          +0.3%     60d   
  RBLBANK     83     BANKING               01-Apr-19   659.7       631.9       147    ₹-4,079       -4.2%     91d   
  IPCALAB     53     PHARMA                01-Apr-19   474.3       452.9       204    ₹-4,351       -4.5%     91d   
  INDIGO      61     CAPITAL_GOODS         03-Jun-19   1,683.2     1,573.5     58     ₹-6,364       -6.5%     28d   
  GUJGASLTD   56     ENERGY                03-Jun-19   177.2       162.9       554    ₹-7,932       -8.1%     28d   

  ENTRIES (6)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  ADANIPOWER  1      ENERGY                3.152    1.20   +230.4%   +26.2%    12.1        8102   ₹97,710     
  PIIND       4      CHEMICALS             2.828    0.28   +53.6%    +17.0%    1,170.1     83     ₹97,122     
  SRF         5      CAPITAL_GOODS         2.813    0.45   +62.3%    +22.5%    589.6       165    ₹97,276     
  SIEMENS     11     CAPITAL_GOODS         2.618    0.62   +33.0%    +25.8%    754.7       129    ₹97,356     
  TRENT       12     CONSUMER_DISCRETIONARY  2.605    0.45   +41.4%    +21.2%    449.7       217    ₹97,590     
  WIPRO       14     IT                    2.511    0.18   +41.9%    +10.6%    129.5       754    ₹97,609     

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BATAINDIA   7      CONSUMER_DISCRETIONARY  01-Aug-18   878.9       1,351.4     111    ₹52,448       +53.8%    
  BAJFINANCE  3      FINANCIAL_SERVICES    01-Aug-18   265.6       360.9       368    ₹35,048       +35.9%    
  TITAN       6      CONSUMER_DISCRETIONARY  01-Apr-19   1,094.0     1,292.1     88     ₹17,431       +18.1%    
  BAJAJFINSV  10     FINANCIAL_SERVICES    02-May-19   755.5       849.4       127    ₹11,928       +12.4%    
  HDFC        18     FINANCIAL_SERVICES    02-May-19   553.2       587.3       173    ₹5,898        +6.2%     
  HDFCBANK    19     BANKING               02-May-19   553.2       587.3       173    ₹5,898        +6.2%     
  AXISBANK    21     BANKING               01-Apr-19   761.5       806.1       127    ₹5,665        +5.9%     
  ABB         9      CAPITAL_GOODS         03-Jun-19   1,401.9     1,408.9     70     ₹493          +0.5%     
  ATUL        2      CHEMICALS             03-Jun-19   3,884.6     3,893.8     25     ₹231          +0.2%     
  GILLETTE    13     FMCG                  02-May-19   6,727.6     6,690.7     14     ₹-517         -0.5%     
  SHREECEM    22     CEMENT                03-Jun-19   21,301.5    20,903.6    4      ₹-1,591       -1.9%     
  NAUKRI      8      TELECOM               03-Jun-19   451.4       438.3       217    ₹-2,829       -2.9%     
  HONAUT      16     CAPITAL_GOODS         03-Jun-19   26,282.6    24,614.7    3      ₹-5,004       -6.3%     

  AFTER: Invested ₹1,936,841 | Cash ₹16,782 | Total ₹1,953,623 | Positions 19/20 | Slot ₹97,716

========================================================================
  REBALANCE #21  —  01 Aug 2019
  NAV: ₹1,755,239  |  Slot: ₹87,762  |  Cash: ₹16,782
========================================================================
  [SECTOR CAP≤4] dropped: HDFCLIFE

  [REGIME OFF] Nifty 200 8,935.7 < SMA200 9,243.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BATAINDIA   20     CONSUMER_DISCRETIONARY  01-Aug-18   878.9       1,221.7     111    ₹38,041       +39.0%    
  BAJFINANCE  33     FINANCIAL_SERVICES    01-Aug-18   265.6       314.9       368    ₹18,129       +18.5%    
  ADANIPOWER  —      ENERGY                01-Jul-19   12.1        11.9        8102   ₹-1,458       -1.5%     
  NAUKRI      2      TELECOM               03-Jun-19   451.4       433.1       217    ₹-3,968       -4.1%     
  WIPRO       56     IT                    01-Jul-19   129.5       124.1       754    ₹-4,015       -4.1%     
  HDFC        102    FINANCIAL_SERVICES    02-May-19   553.2       526.1       173    ₹-4,680       -4.9%     
  HDFCBANK    103    BANKING               02-May-19   553.2       526.1       173    ₹-4,680       -4.9%     
  GILLETTE    75     FMCG                  02-May-19   6,727.6     6,389.0     14     ₹-4,740       -5.0%     
  TITAN       68     CONSUMER_DISCRETIONARY  01-Apr-19   1,094.0     1,037.1     88     ₹-5,005       -5.2%     
  BAJAJFINSV  85     FINANCIAL_SERVICES    02-May-19   755.5       706.2       127    ₹-6,261       -6.5%     
  PIIND       7      CHEMICALS             01-Jul-19   1,170.1     1,093.1     83     ₹-6,397       -6.6%     
  TRENT       14     CONSUMER_DISCRETIONARY  01-Jul-19   449.7       418.3       217    ₹-6,828       -7.0%     
  SHREECEM    36     CEMENT                03-Jun-19   21,301.5    19,507.9    4      ₹-7,174       -8.4%     
  ATUL        12     CHEMICALS             03-Jun-19   3,884.6     3,533.6     25     ₹-8,774       -9.0%     
  AXISBANK    74     BANKING               01-Apr-19   761.5       666.5       127    ₹-12,059      -12.5%    
  SRF         4      CAPITAL_GOODS         01-Jul-19   589.6       515.0       165    ₹-12,297      -12.6%    
  HONAUT      49     CAPITAL_GOODS         03-Jun-19   26,282.6    22,664.6    3      ₹-10,854      -13.8%    
  ABB         87     CAPITAL_GOODS         03-Jun-19   1,401.9     1,190.1     70     ₹-14,824      -15.1%    
  SIEMENS     64     CAPITAL_GOODS         01-Jul-19   754.7       635.0       129    ₹-15,436      -15.9%    

  AFTER: Invested ₹1,738,458 | Cash ₹16,782 | Total ₹1,755,239 | Positions 19/20 | Slot ₹87,762

========================================================================
  REBALANCE #22  —  03 Sep 2019
  NAV: ₹1,770,452  |  Slot: ₹88,523  |  Cash: ₹16,782
========================================================================

  [REGIME OFF] Nifty 200 8,802.3 < SMA200 9,267.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BATAINDIA   1      CONSUMER_DISCRETIONARY  01-Aug-18   878.9       1,444.0     111    ₹62,723       +64.3%    
  BAJFINANCE  49     FINANCIAL_SERVICES    01-Aug-18   265.6       318.5       368    ₹19,438       +19.9%    
  TRENT       6      CONSUMER_DISCRETIONARY  01-Jul-19   449.7       447.7       217    ₹-428         -0.4%     
  GILLETTE    33     FMCG                  02-May-19   6,727.6     6,606.9     14     ₹-1,689       -1.8%     
  PIIND       3      CHEMICALS             01-Jul-19   1,170.1     1,133.0     83     ₹-3,082       -3.2%     
  ADANIPOWER  —      ENERGY                01-Jul-19   12.1        11.5        8102   ₹-4,699       -4.8%     
  TITAN       84     CONSUMER_DISCRETIONARY  01-Apr-19   1,094.0     1,039.4     88     ₹-4,803       -5.0%     
  HDFC        81     FINANCIAL_SERVICES    02-May-19   553.2       523.5       173    ₹-5,125       -5.4%     
  HDFCBANK    82     BANKING               02-May-19   553.2       523.5       173    ₹-5,125       -5.4%     
  BAJAJFINSV  105    FINANCIAL_SERVICES    02-May-19   755.5       700.7       127    ₹-6,966       -7.3% ⚠   
  HONAUT      48     CAPITAL_GOODS         03-Jun-19   26,282.6    24,236.5    3      ₹-6,138       -7.8%     
  SRF         26     CAPITAL_GOODS         01-Jul-19   589.6       532.9       165    ₹-9,346       -9.6%     
  WIPRO       56     IT                    01-Jul-19   129.5       116.4       754    ₹-9,813       -10.1%    
  NAUKRI      25     TELECOM               03-Jun-19   451.4       398.2       217    ₹-11,546      -11.8%    
  ATUL        83     CHEMICALS             03-Jun-19   3,884.6     3,411.3     25     ₹-11,832      -12.2%    
  SIEMENS     60     CAPITAL_GOODS         01-Jul-19   754.7       656.7       129    ₹-12,637      -13.0%    
  AXISBANK    126    BANKING               01-Apr-19   761.5       643.0       127    ₹-15,044      -15.6% ⚠  
  ABB         97     CAPITAL_GOODS         03-Jun-19   1,401.9     1,170.0     70     ₹-16,235      -16.5%    
  SHREECEM    124    CEMENT                03-Jun-19   21,301.5    17,370.6    4      ₹-15,723      -18.5% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): BAJAJFINSV, SHREECEM, AXISBANK

  AFTER: Invested ₹1,753,670 | Cash ₹16,782 | Total ₹1,770,452 | Positions 19/20 | Slot ₹88,523

========================================================================
  REBALANCE #23  —  01 Oct 2019
  NAV: ₹1,955,629  |  Slot: ₹97,781  |  Cash: ₹16,782
========================================================================

  [REGIME OFF] Nifty 200 9,236.4 < SMA200 9,278.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BATAINDIA   3      CONSUMER_DISCRETIONARY  01-Aug-18   878.9       1,630.0     111    ₹83,369       +85.5%    
  BAJFINANCE  25     FINANCIAL_SERVICES    01-Aug-18   265.6       390.2       368    ₹45,833       +46.9%    
  SIEMENS     10     CAPITAL_GOODS         01-Jul-19   754.7       875.4       129    ₹15,572       +16.0%    
  TITAN       29     CONSUMER_DISCRETIONARY  01-Apr-19   1,094.0     1,256.2     88     ₹14,270       +14.8%    
  BAJAJFINSV  52     FINANCIAL_SERVICES    02-May-19   755.5       843.5       127    ₹11,170       +11.6%    
  HDFC        30     FINANCIAL_SERVICES    02-May-19   553.2       591.4       173    ₹6,618        +6.9%     
  HDFCBANK    31     BANKING               02-May-19   553.2       591.4       173    ₹6,618        +6.9%     
  TRENT       27     CONSUMER_DISCRETIONARY  01-Jul-19   449.7       475.3       217    ₹5,551        +5.7%     
  PIIND       11     CHEMICALS             01-Jul-19   1,170.1     1,235.2     83     ₹5,396        +5.6%     
  HONAUT      18     CAPITAL_GOODS         03-Jun-19   26,282.6    27,751.2    3      ₹4,406        +5.6%     
  ADANIPOWER  —      ENERGY                01-Jul-19   12.1        12.4        8102   ₹2,350        +2.4%     
  ATUL        40     CHEMICALS             03-Jun-19   3,884.6     3,886.1     25     ₹38           +0.0%     
  GILLETTE    112    FMCG                  02-May-19   6,727.6     6,289.6     14     ₹-6,132       -6.5% ⚠   
  ABB         —      CAPITAL_GOODS         03-Jun-19   1,401.9     1,307.0     70     ₹-6,642       -6.8%     
  NAUKRI      71     TELECOM               03-Jun-19   451.4       403.6       217    ₹-10,379      -10.6%    
  SRF         73     CAPITAL_GOODS         01-Jul-19   589.6       524.9       165    ₹-10,662      -11.0%    
  AXISBANK    126    BANKING               01-Apr-19   761.5       676.3       127    ₹-10,813      -11.2% ⚠  
  SHREECEM    127    CEMENT                03-Jun-19   21,301.5    18,134.6    4      ₹-12,667      -14.9% ⚠  
  WIPRO       167    IT                    01-Jul-19   129.5       107.2       754    ₹-16,787      -17.2% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): GILLETTE, AXISBANK, SHREECEM, WIPRO

  AFTER: Invested ₹1,938,848 | Cash ₹16,782 | Total ₹1,955,629 | Positions 19/20 | Slot ₹97,781

========================================================================
  REBALANCE #24  —  01 Nov 2019
  NAV: ₹2,075,911  |  Slot: ₹103,796  |  Cash: ₹16,782
========================================================================
  [SECTOR CAP≤4] dropped: SBILIFE

  EXITS (11)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  NAUKRI      45     TELECOM               03-Jun-19   451.4       511.5       217    ₹13,054       +13.3%    151d  
  ADANIPOWER  67     ENERGY                01-Jul-19   12.1        13.6        8102   ₹12,315       +12.6%    123d  
  BAJAJFINSV  41     FINANCIAL_SERVICES    02-May-19   755.5       836.1       127    ₹10,238       +10.7%    183d  
  HDFC        56     FINANCIAL_SERVICES    02-May-19   553.2       587.3       173    ₹5,901        +6.2%     183d  
  HDFCBANK    57     BANKING               02-May-19   553.2       587.3       173    ₹5,901        +6.2%     183d  
  GILLETTE    50     FMCG                  02-May-19   6,727.6     7,107.5     14     ₹5,319        +5.6%     183d  
  AXISBANK    92     BANKING               01-Apr-19   761.5       745.2       127    ₹-2,068       -2.1%     214d  
  SRF         54     CAPITAL_GOODS         01-Jul-19   589.6       567.9       165    ₹-3,572       -3.7%     123d  
  WIPRO       147    IT                    01-Jul-19   129.5       118.7       754    ₹-8,099       -8.3%     123d  
  ABB         —      CAPITAL_GOODS         03-Jun-19   1,401.9     1,283.6     70     ₹-8,284       -8.4%     151d  
  SHREECEM    126    CEMENT                03-Jun-19   21,301.5    19,509.7    4      ₹-7,167       -8.4%     151d  

  ENTRIES (10)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  MANAPPURAM  1      FINANCIAL_SERVICES    3.568    0.18   +147.1%   +52.5%    152.5       680    ₹103,722    
  BERGEPAINT  2      CHEMICALS             3.477    0.24   +74.7%    +55.1%    411.5       252    ₹103,693    
  HDFCAMC     3      FINANCIAL_SERVICES    3.465    0.43   +122.8%   +35.3%    1,336.5     77     ₹102,909    
  ABBOTINDIA  5      PHARMA                3.120    0.24   +60.5%    +36.5%    10,661.6    9      ₹95,955     
  BPCL        6      OIL_GAS               2.979    0.15   +92.7%    +54.7%    166.5       623    ₹103,760    
  NESTLEIND   7      FMCG                  2.962    0.11   +58.0%    +31.0%    696.3       149    ₹103,752    
  WHIRLPOOL   9      CONSUMER_DURABLES     2.925    0.23   +52.9%    +45.2%    2,199.4     47     ₹103,370    
  NAM-INDIA   10     FINANCIAL_SERVICES    2.845    0.27   +131.0%   +56.5%    296.7       349    ₹103,543    
  HINDUNILVR  11     FMCG                  2.830    0.13   +42.1%    +26.6%    1,969.3     52     ₹102,405    
  RELAXO      12     CONSUMER_DISCRETIONARY  2.823    0.13   +53.0%    +33.4%    534.0       194    ₹103,605    

  HOLDS (8)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BATAINDIA   4      CONSUMER_DISCRETIONARY  01-Aug-18   878.9       1,634.0     111    ₹83,816       +85.9%    
  BAJFINANCE  20     FINANCIAL_SERVICES    01-Aug-18   265.6       397.8       368    ₹48,618       +49.7%    
  SIEMENS     8      CAPITAL_GOODS         01-Jul-19   754.7       948.2       129    ₹24,967       +25.6%    
  TRENT       17     CONSUMER_DISCRETIONARY  01-Jul-19   449.7       543.6       217    ₹20,377       +20.9%    
  TITAN       24     CONSUMER_DISCRETIONARY  01-Apr-19   1,094.0     1,277.0     88     ₹16,108       +16.7%    
  PIIND       14     CHEMICALS             01-Jul-19   1,170.1     1,362.2     83     ₹15,943       +16.4%    
  ATUL        31     CHEMICALS             03-Jun-19   3,884.6     4,234.9     25     ₹8,757        +9.0%     
  HONAUT      21     CAPITAL_GOODS         03-Jun-19   26,282.6    28,037.0    3      ₹5,263        +6.7%     

  AFTER: Invested ₹2,010,185 | Cash ₹64,507 | Total ₹2,074,692 | Positions 18/20 | Slot ₹103,796

========================================================================
  REBALANCE #25  —  02 Dec 2019
  NAV: ₹2,028,518  |  Slot: ₹101,426  |  Cash: ₹64,507
========================================================================
  [SECTOR CAP≤4] dropped: AAVAS, BAJAJFINSV

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TRENT       42     CONSUMER_DISCRETIONARY  01-Jul-19   449.7       522.7       217    ₹15,841       +16.2%    154d  
  TITAN       95     CONSUMER_DISCRETIONARY  01-Apr-19   1,094.0     1,131.8     88     ₹3,324        +3.5%     245d  
  HONAUT      54     CAPITAL_GOODS         03-Jun-19   26,282.6    26,924.4    3      ₹1,925        +2.4%     182d  
  ATUL        53     CHEMICALS             03-Jun-19   3,884.6     3,949.5     25     ₹1,622        +1.7%     182d  
  HINDUNILVR  47     FMCG                  01-Nov-19   1,969.3     1,846.3     52     ₹-6,398       -6.2%     31d   

  ENTRIES (5)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  GLAXO       2      PHARMA                3.696    0.21   +25.7%    +39.2%    1,424.7     71     ₹101,155    
  PFIZER      4      PHARMA                3.467    0.37   +49.0%    +44.6%    3,665.5     27     ₹98,968     
  LALPATHLAB  7      HEALTHCARE            3.023    0.18   +93.3%    +36.6%    761.8       133    ₹101,319    
  IGL         12     OIL_GAS               2.572    0.12   +56.4%    +26.9%    185.5       546    ₹101,271    
  GUJGASLTD   16     ENERGY                2.354    0.08   +74.3%    +21.1%    206.7       490    ₹101,259    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BATAINDIA   30     CONSUMER_DISCRETIONARY  01-Aug-18   878.9       1,526.7     111    ₹71,900       +73.7%    
  BAJFINANCE  28     FINANCIAL_SERVICES    01-Aug-18   265.6       386.1       368    ₹44,324       +45.3%    
  PIIND       6      CHEMICALS             01-Jul-19   1,170.1     1,478.3     83     ₹25,578       +26.3%    
  HDFCAMC     1      FINANCIAL_SERVICES    01-Nov-19   1,336.5     1,531.1     77     ₹14,984       +14.6%    
  SIEMENS     19     CAPITAL_GOODS         01-Jul-19   754.7       841.1       129    ₹11,141       +11.4%    
  ABBOTINDIA  3      PHARMA                01-Nov-19   10,661.6    11,580.2    9      ₹8,267        +8.6%     
  RELAXO      10     CONSUMER_DISCRETIONARY  01-Nov-19   534.0       575.8       194    ₹8,107        +7.8%     
  NAM-INDIA   14     FINANCIAL_SERVICES    01-Nov-19   296.7       293.6       349    ₹-1,062       -1.0%     
  BERGEPAINT  9      CHEMICALS             01-Nov-19   411.5       400.4       252    ₹-2,794       -2.7%     
  NESTLEIND   20     FMCG                  01-Nov-19   696.3       677.6       149    ₹-2,789       -2.7%     
  WHIRLPOOL   8      CONSUMER_DURABLES     01-Nov-19   2,199.4     2,121.8     47     ₹-3,644       -3.5%     
  BPCL        11     OIL_GAS               01-Nov-19   166.5       160.3       623    ₹-3,866       -3.7%     
  MANAPPURAM  13     FINANCIAL_SERVICES    01-Nov-19   152.5       139.6       680    ₹-8,784       -8.5%     

  AFTER: Invested ₹1,979,441 | Cash ₹48,479 | Total ₹2,027,920 | Positions 18/20 | Slot ₹101,426

========================================================================
  REBALANCE #26  —  01 Jan 2020
  NAV: ₹2,089,098  |  Slot: ₹104,455  |  Cash: ₹48,479
========================================================================
  [SECTOR CAP≤4] dropped: AAVAS, CRISIL, SBILIFE

  EXITS (3)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BATAINDIA   40     CONSUMER_DISCRETIONARY  01-Aug-18   878.9       1,637.8     111    ₹84,238       +86.3%    518d  
  SIEMENS     54     CAPITAL_GOODS         01-Jul-19   754.7       845.9       129    ₹11,767       +12.1%    184d  
  BPCL        59     OIL_GAS               01-Nov-19   166.5       157.7       623    ₹-5,544       -5.3%     61d   

  ENTRIES (4)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  IPCALAB     12     PHARMA                2.628    0.09   +46.7%    +23.3%    556.7       187    ₹104,112    
  COROMANDEL  13     CHEMICALS             2.620    -0.01  +20.4%    +28.8%    492.0       212    ₹104,303    
  BHARTIARTL  16     TELECOM               2.556    -0.28  +55.3%    +29.8%    433.3       241    ₹104,429    
  SRF         18     CAPITAL_GOODS         2.479    -0.09  +55.7%    +25.0%    673.0       155    ₹104,309    

  HOLDS (15)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BAJFINANCE  38     FINANCIAL_SERVICES    01-Aug-18   265.6       413.5       368    ₹54,425       +55.7%    
  PIIND       17     CHEMICALS             01-Jul-19   1,170.1     1,424.0     83     ₹21,070       +21.7%    
  GUJGASLTD   1      ENERGY                02-Dec-19   206.7       242.9       490    ₹17,757       +17.5%    
  ABBOTINDIA  3      PHARMA                01-Nov-19   10,661.6    12,056.1    9      ₹12,550       +13.1%    
  RELAXO      4      CONSUMER_DISCRETIONARY  01-Nov-19   534.0       603.5       194    ₹13,465       +13.0%    
  HDFCAMC     10     FINANCIAL_SERVICES    01-Nov-19   1,336.5     1,429.8     77     ₹7,183        +7.0%     
  WHIRLPOOL   14     CONSUMER_DURABLES     01-Nov-19   2,199.4     2,293.0     47     ₹4,399        +4.3%     
  MANAPPURAM  8      FINANCIAL_SERVICES    01-Nov-19   152.5       157.7       680    ₹3,496        +3.4%     
  IGL         7      OIL_GAS               02-Dec-19   185.5       189.6       546    ₹2,264        +2.2%     
  BERGEPAINT  15     CHEMICALS             01-Nov-19   411.5       418.7       252    ₹1,825        +1.8%     
  PFIZER      5      PHARMA                02-Dec-19   3,665.5     3,697.5     27     ₹865          +0.9%     
  NAM-INDIA   9      FINANCIAL_SERVICES    01-Nov-19   296.7       295.7       349    ₹-335         -0.3%     
  NESTLEIND   36     FMCG                  01-Nov-19   696.3       690.6       149    ₹-856         -0.8%     
  GLAXO       20     PHARMA                02-Dec-19   1,424.7     1,393.7     71     ₹-2,201       -2.2%     
  LALPATHLAB  35     HEALTHCARE            02-Dec-19   761.8       733.0       133    ₹-3,827       -3.8%     

  AFTER: Invested ₹2,068,632 | Cash ₹19,971 | Total ₹2,088,603 | Positions 19/20 | Slot ₹104,455

========================================================================
  REBALANCE #27  —  03 Feb 2020
  NAV: ₹2,234,444  |  Slot: ₹111,722  |  Cash: ₹19,971
========================================================================

  EXITS (4)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJFINANCE  33     FINANCIAL_SERVICES    01-Aug-18   265.6       426.0       368    ₹59,030       +60.4%    551d  
  GLAXO       72     PHARMA                02-Dec-19   1,424.7     1,406.1     71     ₹-1,322       -1.3%     63d   
  PFIZER      42     PHARMA                02-Dec-19   3,665.5     3,595.7     27     ₹-1,884       -1.9%     63d   
  NAM-INDIA   31     FINANCIAL_SERVICES    01-Nov-19   296.7       289.0       349    ₹-2,674       -2.6%     94d   

  ENTRIES (4)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  AAVAS       2      FINANCIAL_SERVICES    3.795    0.22   +137.5%   +29.8%    1,988.8     56     ₹111,373    
  AUBANK      4      FINANCIAL_SERVICES    3.682    0.36   +71.7%    +52.0%    519.4       215    ₹111,666    
  TATACONSUM  9      FMCG                  2.826    0.18   +81.9%    +23.6%    358.8       311    ₹111,588    
  JUBLFOOD    11     CONSUMER_DISCRETIONARY  2.668    0.34   +66.3%    +25.1%    385.8       289    ₹111,496    

  HOLDS (15)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  RELAXO      1      CONSUMER_DISCRETIONARY  01-Nov-19   534.0       706.9       194    ₹33,533       +32.4%    
  GUJGASLTD   3      ENERGY                02-Dec-19   206.7       269.8       490    ₹30,934       +30.5%    
  PIIND       12     CHEMICALS             01-Jul-19   1,170.1     1,520.1     83     ₹29,044       +29.9%    
  IGL         6      OIL_GAS               02-Dec-19   185.5       227.3       546    ₹22,813       +22.5%    
  COROMANDEL  5      CHEMICALS             01-Jan-20   492.0       585.9       212    ₹19,916       +19.1%    
  BHARTIARTL  8      TELECOM               01-Jan-20   433.3       487.6       241    ₹13,074       +12.5%    
  BERGEPAINT  10     CHEMICALS             01-Nov-19   411.5       461.5       252    ₹12,602       +12.2%    
  SRF         7      CAPITAL_GOODS         01-Jan-20   673.0       741.4       155    ₹10,610       +10.2%    
  ABBOTINDIA  24     PHARMA                01-Nov-19   10,661.6    11,694.8    9      ₹9,299        +9.7%     
  NESTLEIND   19     FMCG                  01-Nov-19   696.3       761.7       149    ₹9,740        +9.4%     
  LALPATHLAB  14     HEALTHCARE            02-Dec-19   761.8       832.1       133    ₹9,352        +9.2%     
  MANAPPURAM  26     FINANCIAL_SERVICES    01-Nov-19   152.5       163.8       680    ₹7,639        +7.4%     
  WHIRLPOOL   27     CONSUMER_DURABLES     01-Nov-19   2,199.4     2,361.3     47     ₹7,612        +7.4%     
  IPCALAB     13     PHARMA                01-Jan-20   556.7       583.3       187    ₹4,962        +4.8%     
  HDFCAMC     16     FINANCIAL_SERVICES    01-Nov-19   1,336.5     1,383.8     77     ₹3,642        +3.5%     

  AFTER: Invested ₹2,206,025 | Cash ₹27,890 | Total ₹2,233,915 | Positions 19/20 | Slot ₹111,722

========================================================================
  REBALANCE #28  —  02 Mar 2020
  NAV: ₹2,196,316  |  Slot: ₹109,816  |  Cash: ₹27,890
========================================================================
  [SECTOR CAP≤4] dropped: MUTHOOTFIN

  [REGIME OFF] Nifty 200 9,178.8 < SMA200 9,535.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ABBOTINDIA  1      PHARMA                01-Nov-19   10,661.6    14,123.8    9      ₹31,160       +32.5%    
  GUJGASLTD   3      ENERGY                02-Dec-19   206.7       265.9       490    ₹29,018       +28.7%    
  PIIND       27     CHEMICALS             01-Jul-19   1,170.1     1,496.3     83     ₹27,072       +27.9%    
  RELAXO      7      CONSUMER_DISCRETIONARY  01-Nov-19   534.0       679.2       194    ₹28,166       +27.2%    
  IPCALAB     8      PHARMA                01-Jan-20   556.7       704.0       187    ₹27,537       +26.4%    
  COROMANDEL  11     CHEMICALS             01-Jan-20   492.0       572.6       212    ₹17,095       +16.4%    
  BHARTIARTL  19     TELECOM               01-Jan-20   433.3       495.5       241    ₹14,986       +14.4%    
  SRF         12     CAPITAL_GOODS         01-Jan-20   673.0       765.8       155    ₹14,390       +13.8%    
  AUBANK      4      FINANCIAL_SERVICES    03-Feb-20   519.4       575.0       215    ₹11,959       +10.7%    
  BERGEPAINT  9      CHEMICALS             01-Nov-19   411.5       452.2       252    ₹10,262       +9.9%     
  NESTLEIND   14     FMCG                  01-Nov-19   696.3       752.8       149    ₹8,410        +8.1%     
  IGL         38     OIL_GAS               02-Dec-19   185.5       192.8       546    ₹3,981        +3.9%     
  LALPATHLAB  41     HEALTHCARE            02-Dec-19   761.8       786.4       133    ₹3,271        +3.2%     
  HDFCAMC     23     FINANCIAL_SERVICES    01-Nov-19   1,336.5     1,373.0     77     ₹2,813        +2.7%     
  WHIRLPOOL   30     CONSUMER_DURABLES     01-Nov-19   2,199.4     2,175.0     47     ₹-1,143       -1.1%     
  MANAPPURAM  55     FINANCIAL_SERVICES    01-Nov-19   152.5       143.2       680    ₹-6,319       -6.1%     
  AAVAS       16     FINANCIAL_SERVICES    03-Feb-20   1,988.8     1,858.4     56     ₹-7,302       -6.6%     
  TATACONSUM  20     FMCG                  03-Feb-20   358.8       320.9       311    ₹-11,796      -10.6%    
  JUBLFOOD    67     CONSUMER_DISCRETIONARY  03-Feb-20   385.8       329.1       289    ₹-16,387      -14.7%    

  AFTER: Invested ₹2,168,426 | Cash ₹27,890 | Total ₹2,196,316 | Positions 19/20 | Slot ₹109,816

========================================================================
  REBALANCE #29  —  01 Apr 2020
  NAV: ₹1,769,230  |  Slot: ₹88,461  |  Cash: ₹27,890
========================================================================
  [SECTOR CAP≤4] dropped: DRREDDY, TORNTPHARM, DIVISLAB

  [REGIME OFF] Nifty 200 6,761.9 < SMA200 9,328.1 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ABBOTINDIA  1      PHARMA                01-Nov-19   10,661.6    14,325.7    9      ₹32,976       +34.4%    
  IPCALAB     3      PHARMA                01-Jan-20   556.7       686.1       187    ₹24,196       +23.2%    
  RELAXO      8      CONSUMER_DISCRETIONARY  01-Nov-19   534.0       574.3       194    ₹7,813        +7.5%     
  NESTLEIND   4      FMCG                  01-Nov-19   696.3       731.5       149    ₹5,240        +5.1%     
  GUJGASLTD   11     ENERGY                02-Dec-19   206.7       216.2       490    ₹4,673        +4.6%     
  COROMANDEL  15     CHEMICALS             01-Jan-20   492.0       497.7       212    ₹1,203        +1.2%     
  PIIND       32     CHEMICALS             01-Jul-19   1,170.1     1,175.0     83     ₹405          +0.4%     
  BERGEPAINT  9      CHEMICALS             01-Nov-19   411.5       392.5       252    ₹-4,776       -4.6%     
  IGL         20     OIL_GAS               02-Dec-19   185.5       174.2       546    ₹-6,184       -6.1%     
  BHARTIARTL  14     TELECOM               01-Jan-20   433.3       402.8       241    ₹-7,360       -7.0%     
  LALPATHLAB  18     HEALTHCARE            02-Dec-19   761.8       666.1       133    ₹-12,723      -12.6%    
  WHIRLPOOL   41     CONSUMER_DURABLES     01-Nov-19   2,199.4     1,742.9     47     ₹-21,452      -20.8%    
  SRF         40     CAPITAL_GOODS         01-Jan-20   673.0       520.3       155    ₹-23,657      -22.7%    
  TATACONSUM  19     FMCG                  03-Feb-20   358.8       265.2       311    ₹-29,116      -26.1%    
  HDFCAMC     27     FINANCIAL_SERVICES    01-Nov-19   1,336.5     983.5       77     ₹-27,176      -26.4%    
  JUBLFOOD    39     CONSUMER_DISCRETIONARY  03-Feb-20   385.8       273.9       289    ₹-32,352      -29.0%    
  AAVAS       70     FINANCIAL_SERVICES    03-Feb-20   1,988.8     1,192.0     56     ₹-44,621      -40.1%    
  MANAPPURAM  —      FINANCIAL_SERVICES    01-Nov-19   152.5       83.7        680    ₹-46,824      -45.1%    
  AUBANK      95     FINANCIAL_SERVICES    03-Feb-20   519.4       239.5       215    ₹-60,179      -53.9%    

  AFTER: Invested ₹1,741,340 | Cash ₹27,890 | Total ₹1,769,230 | Positions 19/20 | Slot ₹88,461

========================================================================
  REBALANCE #30  —  04 May 2020
  NAV: ₹2,019,626  |  Slot: ₹100,981  |  Cash: ₹27,890
========================================================================
  [SECTOR CAP≤4] dropped: ALKEM, PFIZER, AJANTPHARM, DIVISLAB, CIPLA, APLLTD, TORNTPHARM, BIOCON

  [REGIME OFF] Nifty 200 7,596.9 < SMA200 9,130.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ABBOTINDIA  1      PHARMA                01-Nov-19   10,661.6    16,694.5    9      ₹54,295       +56.6%    
  IPCALAB     2      PHARMA                01-Jan-20   556.7       792.6       187    ₹44,097       +42.4%    
  PIIND       17     CHEMICALS             01-Jul-19   1,170.1     1,489.1     83     ₹26,472       +27.3%    
  BHARTIARTL  14     TELECOM               01-Jan-20   433.3       509.0       241    ₹18,246       +17.5%    
  NESTLEIND   5      FMCG                  01-Nov-19   696.3       815.7       149    ₹17,785       +17.1%    
  GUJGASLTD   25     ENERGY                02-Dec-19   206.7       239.5       490    ₹16,074       +15.9%    
  IGL         22     OIL_GAS               02-Dec-19   185.5       209.4       546    ₹13,087       +12.9%    
  RELAXO      36     CONSUMER_DISCRETIONARY  01-Nov-19   534.0       597.4       194    ₹12,285       +11.9%    
  COROMANDEL  26     CHEMICALS             01-Jan-20   492.0       529.7       212    ₹7,990        +7.7%     
  SRF         18     CAPITAL_GOODS         01-Jan-20   673.0       713.5       155    ₹6,284        +6.0%     
  LALPATHLAB  29     HEALTHCARE            02-Dec-19   761.8       735.8       133    ₹-3,462       -3.4%     
  BERGEPAINT  27     CHEMICALS             01-Nov-19   411.5       390.6       252    ₹-5,257       -5.1%     
  TATACONSUM  20     FMCG                  03-Feb-20   358.8       317.2       311    ₹-12,930      -11.6%    
  WHIRLPOOL   44     CONSUMER_DURABLES     01-Nov-19   2,199.4     1,930.8     47     ₹-12,621      -12.2%    
  HDFCAMC     30     FINANCIAL_SERVICES    01-Nov-19   1,336.5     1,149.6     77     ₹-14,389      -14.0%    
  JUBLFOOD    50     CONSUMER_DISCRETIONARY  03-Feb-20   385.8       306.5       289    ₹-22,921      -20.6%    
  MANAPPURAM  —      FINANCIAL_SERVICES    01-Nov-19   152.5       108.2       680    ₹-30,132      -29.1%    
  AAVAS       96     FINANCIAL_SERVICES    03-Feb-20   1,988.8     1,130.6     56     ₹-48,059      -43.2% ⚠  
  AUBANK      143    FINANCIAL_SERVICES    03-Feb-20   519.4       257.2       215    ₹-56,360      -50.5% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): AAVAS, AUBANK

  AFTER: Invested ₹1,991,736 | Cash ₹27,890 | Total ₹2,019,626 | Positions 19/20 | Slot ₹100,981

========================================================================
  REBALANCE #31  —  01 Jun 2020
  NAV: ₹2,059,457  |  Slot: ₹102,973  |  Cash: ₹27,890
========================================================================
  [SECTOR CAP≤4] dropped: APLLTD, ZYDUSLIFE, DIVISLAB, BIOCON, SANOFI, LUPIN

  [REGIME OFF] Nifty 200 8,020.1 < SMA200 8,960.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ABBOTINDIA  1      PHARMA                01-Nov-19   10,661.6    15,437.7    9      ₹42,985       +44.8%    
  IPCALAB     11     PHARMA                01-Jan-20   556.7       749.6       187    ₹36,056       +34.6%    
  PIIND       26     CHEMICALS             01-Jul-19   1,170.1     1,552.3     83     ₹31,719       +32.7%    
  RELAXO      18     CONSUMER_DISCRETIONARY  01-Nov-19   534.0       698.2       194    ₹31,838       +30.7%    
  BHARTIARTL  15     TELECOM               01-Jan-20   433.3       534.4       241    ₹24,362       +23.3%    
  COROMANDEL  10     CHEMICALS             01-Jan-20   492.0       605.6       212    ₹24,088       +23.1%    
  NESTLEIND   9      FMCG                  01-Nov-19   696.3       802.9       149    ₹15,884       +15.3%    
  GUJGASLTD   42     ENERGY                02-Dec-19   206.7       233.9       490    ₹13,364       +13.2%    
  IGL         19     OIL_GAS               02-Dec-19   185.5       209.8       546    ₹13,293       +13.1%    
  SRF         53     CAPITAL_GOODS         01-Jan-20   673.0       726.6       155    ₹8,307        +8.0%     
  BERGEPAINT  32     CHEMICALS             01-Nov-19   411.5       397.9       252    ₹-3,426       -3.3%     
  LALPATHLAB  35     HEALTHCARE            02-Dec-19   761.8       730.4       133    ₹-4,180       -4.1%     
  TATACONSUM  28     FMCG                  03-Feb-20   358.8       343.7       311    ₹-4,713       -4.2%     
  WHIRLPOOL   39     CONSUMER_DURABLES     01-Nov-19   2,199.4     1,998.5     47     ₹-9,441       -9.1%     
  HDFCAMC     41     FINANCIAL_SERVICES    01-Nov-19   1,336.5     1,194.4     77     ₹-10,938      -10.6%    
  JUBLFOOD    54     CONSUMER_DISCRETIONARY  03-Feb-20   385.8       331.8       289    ₹-15,594      -14.0%    
  MANAPPURAM  —      FINANCIAL_SERVICES    01-Nov-19   152.5       117.8       680    ₹-23,645      -22.8%    
  AAVAS       191    FINANCIAL_SERVICES    03-Feb-20   1,988.8     1,063.1     56     ₹-51,842      -46.5% ⚠  
  AUBANK      216    FINANCIAL_SERVICES    03-Feb-20   519.4       204.0       215    ₹-67,805      -60.7% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): AAVAS, AUBANK

  AFTER: Invested ₹2,031,567 | Cash ₹27,890 | Total ₹2,059,457 | Positions 19/20 | Slot ₹102,973

========================================================================
  REBALANCE #32  —  01 Jul 2020
  NAV: ₹2,141,248  |  Slot: ₹107,062  |  Cash: ₹27,890
========================================================================
  [SECTOR CAP≤4] dropped: LUPIN, CIPLA, AUROPHARMA, DRREDDY

  [REGIME OFF] Nifty 200 8,554.7 < SMA200 8,895.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  GUJGASLTD   15     ENERGY                02-Dec-19   206.7       311.5       490    ₹51,354       +50.7%    
  IPCALAB     19     PHARMA                01-Jan-20   556.7       809.5       187    ₹47,264       +45.4%    
  COROMANDEL  8      CHEMICALS             01-Jan-20   492.0       700.0       212    ₹44,093       +42.3%    
  ABBOTINDIA  30     PHARMA                01-Nov-19   10,661.6    14,456.3    9      ₹34,152       +35.6%    
  PIIND       35     CHEMICALS             01-Jul-19   1,170.1     1,508.1     83     ₹28,051       +28.9%    
  BHARTIARTL  44     TELECOM               01-Jan-20   433.3       535.5       241    ₹24,616       +23.6%    
  RELAXO      60     CONSUMER_DISCRETIONARY  01-Nov-19   534.0       617.6       194    ₹16,203       +15.6%    
  NESTLEIND   46     FMCG                  01-Nov-19   696.3       787.9       149    ₹13,640       +13.1%    
  IGL         61     OIL_GAS               02-Dec-19   185.5       195.6       546    ₹5,539        +5.5%     
  SRF         67     CAPITAL_GOODS         01-Jan-20   673.0       705.2       155    ₹4,990        +4.8%     
  TATACONSUM  29     FMCG                  03-Feb-20   358.8       366.8       311    ₹2,487        +2.2%     
  LALPATHLAB  81     HEALTHCARE            02-Dec-19   761.8       756.8       133    ₹-668         -0.7%     
  BERGEPAINT  76     CHEMICALS             01-Nov-19   411.5       401.3       252    ₹-2,556       -2.5%     
  MANAPPURAM  —      FINANCIAL_SERVICES    01-Nov-19   152.5       141.8       680    ₹-7,288       -7.0%     
  WHIRLPOOL   88     CONSUMER_DURABLES     01-Nov-19   2,199.4     2,015.5     47     ₹-8,640       -8.4%     
  JUBLFOOD    59     CONSUMER_DISCRETIONARY  03-Feb-20   385.8       336.0       289    ₹-14,384      -12.9%    
  HDFCAMC     90     FINANCIAL_SERVICES    01-Nov-19   1,336.5     1,114.2     77     ₹-17,115      -16.6%    
  AAVAS       105    FINANCIAL_SERVICES    03-Feb-20   1,988.8     1,315.4     56     ₹-37,708      -33.9% ⚠  
  AUBANK      166    FINANCIAL_SERVICES    03-Feb-20   519.4       277.9       215    ₹-51,926      -46.5% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): AAVAS, AUBANK

  AFTER: Invested ₹2,113,358 | Cash ₹27,890 | Total ₹2,141,248 | Positions 19/20 | Slot ₹107,062

========================================================================
  REBALANCE #33  —  03 Aug 2020
  NAV: ₹2,239,694  |  Slot: ₹111,985  |  Cash: ₹27,890
========================================================================

  EXITS (13)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ABBOTINDIA  96     PHARMA                01-Nov-19   10,661.6    14,688.6    9      ₹36,242       +37.8%    276d  
  BHARTIARTL  91     TELECOM               01-Jan-20   433.3       523.2       241    ₹21,667       +20.7%    215d  
  SRF         107    CAPITAL_GOODS         01-Jan-20   673.0       751.3       155    ₹12,142       +11.6%    215d  
  NESTLEIND   119    FMCG                  01-Nov-19   696.3       775.0       149    ₹11,725       +11.3%    276d  
  RELAXO      115    CONSUMER_DISCRETIONARY  01-Nov-19   534.0       584.3       194    ₹9,758        +9.4%     276d  
  BERGEPAINT  43     CHEMICALS             01-Nov-19   411.5       426.3       252    ₹3,745        +3.6%     276d  
  IGL         189    OIL_GAS               02-Dec-19   185.5       175.4       546    ₹-5,490       -5.4%     245d  
  WHIRLPOOL   99     CONSUMER_DURABLES     01-Nov-19   2,199.4     2,080.9     47     ₹-5,568       -5.4%     276d  
  MANAPPURAM  —      FINANCIAL_SERVICES    01-Nov-19   152.5       142.6       680    ₹-6,740       -6.5%     276d  
  JUBLFOOD    79     CONSUMER_DISCRETIONARY  03-Feb-20   385.8       344.3       289    ₹-12,006      -10.8%    182d  
  HDFCAMC     163    FINANCIAL_SERVICES    01-Nov-19   1,336.5     1,103.8     77     ₹-17,918      -17.4%    276d  
  AAVAS       110    FINANCIAL_SERVICES    03-Feb-20   1,988.8     1,407.2     56     ₹-32,567      -29.2%    182d  
  AUBANK      57     FINANCIAL_SERVICES    03-Feb-20   519.4       354.9       215    ₹-35,352      -31.7%    182d  

  ENTRIES (12)
  [52w filter blocked 2: ADANIGREEN(-30.2%), IDBI(-30.0%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  JUBILANT    2      CONSUMER_DISCRETIONARY  3.751    0.79   +100.3%   +127.5%   618.2       181    ₹111,891    
  TATACOMM    3      TELECOM               3.375    0.34   +170.8%   +78.0%    740.4       151    ₹111,807    
  HAL         4      CAPITAL_GOODS         2.874    0.34   +40.8%    +77.5%    399.6       280    ₹111,892    
  SYNGENE     5      HEALTHCARE            2.827    0.44   +48.5%    +51.9%    474.9       235    ₹111,603    
  MPHASIS     7      IT                    2.643    0.54   +30.2%    +66.8%    1,024.9     109    ₹111,719    
  MUTHOOTFIN  8      FINANCIAL_SERVICES    2.612    1.12   +115.6%   +64.0%    1,172.9     95     ₹111,426    
  ESCORTS     9      AUTO                  2.396    1.18   +119.5%   +59.2%    1,064.6     105    ₹111,780    
  BALKRISIND  10     AUTO_ANCILLARY        2.322    0.94   +81.9%    +47.5%    1,249.0     89     ₹111,157    
  NATCOPHARM  11     PHARMA                2.270    0.48   +60.8%    +34.8%    772.9       144    ₹111,296    
  JKCEMENT    12     CEMENT                2.159    0.59   +56.3%    +39.1%    1,471.4     76     ₹111,824    
  WIPRO       13     IT                    2.140    0.61   +6.9%     +52.9%    129.8       862    ₹111,862    
  DRREDDY     15     PHARMA                2.054    0.40   +75.3%    +18.9%    877.5       127    ₹111,439    

  HOLDS (6)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  IPCALAB     14     PHARMA                01-Jan-20   556.7       927.0       187    ₹69,231       +66.5%    
  PIIND       35     CHEMICALS             01-Jul-19   1,170.1     1,809.4     83     ₹53,058       +54.6%    
  COROMANDEL  6      CHEMICALS             01-Jan-20   492.0       737.5       212    ₹52,047       +49.9%    
  GUJGASLTD   16     ENERGY                02-Dec-19   206.7       291.7       490    ₹41,681       +41.2%    
  LALPATHLAB  34     HEALTHCARE            02-Dec-19   761.8       894.8       133    ₹17,691       +17.5%    
  TATACONSUM  19     FMCG                  03-Feb-20   358.8       414.1       311    ₹17,203       +15.4%    

  AFTER: Invested ₹2,210,309 | Cash ₹27,794 | Total ₹2,238,103 | Positions 18/20 | Slot ₹111,985

========================================================================
  REBALANCE #34  —  01 Sep 2020
  NAV: ₹2,253,090  |  Slot: ₹112,654  |  Cash: ₹27,794
========================================================================

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  GUJGASLTD   49     ENERGY                02-Dec-19   206.7       299.4       490    ₹45,466       +44.9%    274d  
  LALPATHLAB  45     HEALTHCARE            02-Dec-19   761.8       861.8       133    ₹13,302       +13.1%    274d  
  BALKRISIND  40     AUTO_ANCILLARY        03-Aug-20   1,249.0     1,276.1     89     ₹2,413        +2.2%     29d   
  ESCORTS     —      AUTO                  03-Aug-20   1,064.6     1,076.3     105    ₹1,231        +1.1%     29d   
  WIPRO       46     IT                    03-Aug-20   129.8       125.2       862    ₹-3,953       -3.5%     29d   

  ENTRIES (5)
  [52w filter blocked 1: CHOLAHLDNG(-23.2%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  ADANIGREEN  1      ENERGY                8.327    0.48   +1028.1%  +80.5%    494.6       227    ₹112,286    
  EMAMILTD    3      FMCG                  3.176    0.69   +26.9%    +93.7%    323.5       348    ₹112,569    
  DIVISLAB    4      PHARMA                2.771    0.48   +107.1%   +33.5%    3,134.3     35     ₹109,700    
  JSWSTEEL    12     METALS                1.907    1.19   +31.4%    +51.0%    271.5       414    ₹112,397    
  ATUL        13     CHEMICALS             1.882    0.74   +68.2%    +28.7%    5,854.6     19     ₹111,237    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  IPCALAB     6      PHARMA                01-Jan-20   556.7       989.1       187    ₹80,848       +77.7%    
  PIIND       35     CHEMICALS             01-Jul-19   1,170.1     1,828.7     83     ₹54,661       +56.3%    
  TATACONSUM  5      FMCG                  03-Feb-20   358.8       523.3       311    ₹51,144       +45.8%    
  COROMANDEL  9      CHEMICALS             01-Jan-20   492.0       699.9       212    ₹44,078       +42.3%    
  TATACOMM    2      TELECOM               03-Aug-20   740.4       811.0       151    ₹10,661       +9.5%     
  HAL         10     CAPITAL_GOODS         03-Aug-20   399.6       398.7       280    ₹-244         -0.2%     
  JKCEMENT    16     CEMENT                03-Aug-20   1,471.4     1,465.0     76     ₹-483         -0.4%     
  SYNGENE     7      HEALTHCARE            03-Aug-20   474.9       469.1       235    ₹-1,365       -1.2%     
  MPHASIS     27     IT                    03-Aug-20   1,024.9     1,009.9     109    ₹-1,645       -1.5%     
  DRREDDY     39     PHARMA                03-Aug-20   877.5       843.5       127    ₹-4,320       -3.9%     
  NATCOPHARM  15     PHARMA                03-Aug-20   772.9       736.2       144    ₹-5,277       -4.7%     
  JUBILANT    8      CONSUMER_DISCRETIONARY  03-Aug-20   618.2       563.4       181    ₹-9,907       -8.9%     
  MUTHOOTFIN  31     FINANCIAL_SERVICES    03-Aug-20   1,172.9     1,060.1     95     ₹-10,713      -9.6%     

  AFTER: Invested ₹2,187,648 | Cash ₹64,779 | Total ₹2,252,427 | Positions 18/20 | Slot ₹112,654

========================================================================
  REBALANCE #35  —  01 Oct 2020
  NAV: ₹2,394,115  |  Slot: ₹119,706  |  Cash: ₹64,779
========================================================================
  [SECTOR CAP≤4] dropped: NAVINFLUOR

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  JKCEMENT    63     CEMENT                03-Aug-20   1,471.4     1,496.8     76     ₹1,936        +1.7%     59d   
  JSWSTEEL    —      METALS                01-Sep-20   271.5       267.0       414    ₹-1,862       -1.7%     30d   
  MUTHOOTFIN  77     FINANCIAL_SERVICES    03-Aug-20   1,172.9     1,055.2     95     ₹-11,178      -10.0%    59d   
  HAL         167    CAPITAL_GOODS         03-Aug-20   399.6       357.3       280    ₹-11,857      -10.6%    59d   
  JUBILANT    —      CONSUMER_DISCRETIONARY  03-Aug-20   618.2       527.1       181    ₹-16,487      -14.7%    59d   

  ENTRIES (4)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  ADANIENT    2      ENERGY                3.179    1.08   +103.2%   +87.7%    307.4       389    ₹119,580    
  HCLTECH     6      IT                    2.492    0.77   +57.3%    +41.3%    646.5       185    ₹119,607    
  WIPRO       8      IT                    2.352    0.66   +32.6%    +40.6%    144.3       829    ₹119,616    
  APOLLOHOSP  10     HEALTHCARE            2.326    0.71   +46.7%    +48.4%    2,062.8     58     ₹119,641    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  IPCALAB     4      PHARMA                01-Jan-20   556.7       1,106.3     187    ₹102,769      +98.7%    
  PIIND       34     CHEMICALS             01-Jul-19   1,170.1     1,913.5     83     ₹61,695       +63.5%    
  COROMANDEL  29     CHEMICALS             01-Jan-20   492.0       749.7       212    ₹54,623       +52.4%    
  ADANIGREEN  1      ENERGY                01-Sep-20   494.6       747.1       227    ₹57,306       +51.0%    
  TATACONSUM  22     FMCG                  03-Feb-20   358.8       485.7       311    ₹39,454       +35.4%    
  MPHASIS     7      IT                    03-Aug-20   1,024.9     1,215.0     109    ₹20,722       +18.5%    
  SYNGENE     11     HEALTHCARE            03-Aug-20   474.9       555.8       235    ₹19,005       +17.0%    
  NATCOPHARM  12     PHARMA                03-Aug-20   772.9       900.8       144    ₹18,424       +16.6%    
  DRREDDY     5      PHARMA                03-Aug-20   877.5       990.8       127    ₹14,389       +12.9%    
  TATACOMM    13     TELECOM               03-Aug-20   740.4       781.6       151    ₹6,219        +5.6%     
  ATUL        17     CHEMICALS             01-Sep-20   5,854.6     6,002.6     19     ₹2,812        +2.5%     
  EMAMILTD    18     FMCG                  01-Sep-20   323.5       314.3       348    ₹-3,200       -2.8%     
  DIVISLAB    3      PHARMA                01-Sep-20   3,134.3     2,973.1     35     ₹-5,643       -5.1%     

  AFTER: Invested ₹2,287,798 | Cash ₹105,749 | Total ₹2,393,547 | Positions 17/20 | Slot ₹119,706

========================================================================
  REBALANCE #36  —  02 Nov 2020
  NAV: ₹2,442,893  |  Slot: ₹122,145  |  Cash: ₹105,749
========================================================================

  EXITS (4)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COROMANDEL  93     CHEMICALS             01-Jan-20   492.0       678.1       212    ₹39,458       +37.8%    306d  
  TATACONSUM  31     FMCG                  03-Feb-20   358.8       471.0       311    ₹34,900       +31.3%    273d  
  NATCOPHARM  28     PHARMA                03-Aug-20   772.9       861.9       144    ₹12,815       +11.5%    91d   
  ATUL        33     CHEMICALS             01-Sep-20   5,854.6     5,835.3     19     ₹-366         -0.3%     62d   

  ENTRIES (5)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  NAVINFLUOR  3      PHARMA                2.474    0.69   +146.8%   +24.7%    2,141.1     57     ₹122,040    
  TIINDIA     6      AUTO_ANCILLARY        2.145    0.56   +79.3%    +30.5%    652.0       187    ₹121,927    
  HAVELLS     7      CAPITAL_GOODS         2.114    0.65   +14.1%    +32.9%    752.1       162    ₹121,838    
  JKCEMENT    8      CEMENT                2.094    0.59   +63.7%    +20.4%    1,770.4     68     ₹120,389    
  ASHOKLEY    9      AUTO                  2.083    1.08   +5.4%     +64.9%    37.4        3266   ₹122,123    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  IPCALAB     4      PHARMA                01-Jan-20   556.7       1,130.2     187    ₹107,230      +103.0%   
  PIIND       12     CHEMICALS             01-Jul-19   1,170.1     2,192.7     83     ₹84,869       +87.4%    
  ADANIGREEN  1      ENERGY                01-Sep-20   494.6       859.2       227    ₹82,753       +73.7%    
  TATACOMM    5      TELECOM               03-Aug-20   740.4       892.5       151    ₹22,966       +20.5%    
  MPHASIS     16     IT                    03-Aug-20   1,024.9     1,216.6     109    ₹20,892       +18.7%    
  ADANIENT    2      ENERGY                01-Oct-20   307.4       340.3       389    ₹12,786       +10.7%    
  SYNGENE     19     HEALTHCARE            03-Aug-20   474.9       525.2       235    ₹11,818       +10.6%    
  DRREDDY     22     PHARMA                03-Aug-20   877.5       941.7       127    ₹8,152        +7.3%     
  WIPRO       27     IT                    01-Oct-20   144.3       154.3       829    ₹8,311        +6.9%     
  HCLTECH     15     IT                    01-Oct-20   646.5       657.1       185    ₹1,949        +1.6%     
  APOLLOHOSP  18     HEALTHCARE            01-Oct-20   2,062.8     2,042.6     58     ₹-1,171       -1.0%     
  EMAMILTD    10     FMCG                  01-Sep-20   323.5       314.1       348    ₹-3,247       -2.9%     
  DIVISLAB    11     PHARMA                01-Sep-20   3,134.3     2,957.3     35     ₹-6,194       -5.6%     

  AFTER: Invested ₹2,420,229 | Cash ₹21,941 | Total ₹2,442,171 | Positions 18/20 | Slot ₹122,145

========================================================================
  REBALANCE #37  —  01 Dec 2020
  NAV: ₹2,735,292  |  Slot: ₹136,765  |  Cash: ₹21,941
========================================================================

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  EMAMILTD    50     FMCG                  01-Sep-20   323.5       401.4       348    ₹27,135       +24.1%    91d   
  MPHASIS     48     IT                    03-Aug-20   1,024.9     1,173.6     109    ₹16,207       +14.5%    120d  
  ASHOKLEY    75     AUTO                  02-Nov-20   37.4        42.4        3266   ₹16,198       +13.3%    29d   
  DRREDDY     37     PHARMA                03-Aug-20   877.5       936.2       127    ₹7,457        +6.7%     120d  
  HCLTECH     30     IT                    01-Oct-20   646.5       666.4       185    ₹3,678        +3.1%     61d   

  ENTRIES (5)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  DALBHARAT   6      CHEMICALS             2.294    0.03   +31.8%    +52.0%    1,126.4     121    ₹136,296    
  MRF         7      AUTO_ANCILLARY        2.285    -0.26  +24.6%    +36.6%    78,892.1    1      ₹78,892     
  APOLLOTYRE  9      AUTO_ANCILLARY        2.115    -0.01  +11.5%    +52.7%    174.0       785    ₹136,605    
  TATACHEM    10     CHEMICALS             2.112    0.04   +41.8%    +30.6%    377.5       362    ₹136,672    
  INFY        13     IT                    2.089    -0.22  +66.1%    +25.8%    1,001.7     136    ₹136,227    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ADANIGREEN  1      ENERGY                01-Sep-20   494.6       1,134.7     227    ₹145,291      +129.4%   
  IPCALAB     25     PHARMA                01-Jan-20   556.7       1,120.5     187    ₹105,415      +101.3%   
  PIIND       19     CHEMICALS             01-Jul-19   1,170.1     2,294.6     83     ₹93,331       +96.1%    
  ADANIENT    5      ENERGY                01-Oct-20   307.4       420.9       389    ₹44,142       +36.9%    
  TATACOMM    17     TELECOM               03-Aug-20   740.4       968.7       151    ₹34,461       +30.8%    
  NAVINFLUOR  2      PHARMA                02-Nov-20   2,141.1     2,641.7     57     ₹28,536       +23.4%    
  TIINDIA     26     AUTO_ANCILLARY        02-Nov-20   652.0       802.3       187    ₹28,094       +23.0%    
  SYNGENE     8      HEALTHCARE            03-Aug-20   474.9       564.7       235    ₹21,093       +18.9%    
  APOLLOHOSP  4      HEALTHCARE            01-Oct-20   2,062.8     2,431.2     58     ₹21,369       +17.9%    
  JKCEMENT    3      CEMENT                02-Nov-20   1,770.4     2,018.6     68     ₹16,874       +14.0%    
  WIPRO       12     IT                    01-Oct-20   144.3       162.6       829    ₹15,207       +12.7%    
  DIVISLAB    11     PHARMA                01-Sep-20   3,134.3     3,512.3     35     ₹13,229       +12.1%    
  HAVELLS     16     CAPITAL_GOODS         02-Nov-20   752.1       792.3       162    ₹6,516        +5.3%     

  AFTER: Invested ₹2,689,912 | Cash ₹44,639 | Total ₹2,734,550 | Positions 18/20 | Slot ₹136,765

========================================================================
  REBALANCE #38  —  01 Jan 2021
  NAV: ₹2,796,755  |  Slot: ₹139,838  |  Cash: ₹44,639
========================================================================

  EXITS (6)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  IPCALAB     102    PHARMA                01-Jan-20   556.7       1,074.3     187    ₹96,776       +93.0%    366d  
  PIIND       58     CHEMICALS             01-Jul-19   1,170.1     2,227.4     83     ₹87,755       +90.4%    550d  
  TIINDIA     45     AUTO_ANCILLARY        02-Nov-20   652.0       790.9       187    ₹25,964       +21.3%    60d   
  APOLLOHOSP  69     HEALTHCARE            01-Oct-20   2,062.8     2,383.7     58     ₹18,612       +15.6%    92d   
  APOLLOTYRE  79     AUTO_ANCILLARY        01-Dec-20   174.0       167.4       785    ₹-5,187       -3.8%     31d   
  MRF         59     AUTO_ANCILLARY        01-Dec-20   78,892.1    75,379.4    1      ₹-3,513       -4.5%     31d   

  ENTRIES (6)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  SAIL        2      METALS                3.467    -0.15  +73.7%    +118.2%   63.2        2211   ₹139,831    
  ATGL        3      OIL_GAS               3.366    0.16   +132.3%   +94.9%    376.3       371    ₹139,611    
  TATASTEEL   6      METALS                2.721    -0.14  +39.8%    +76.2%    55.4        2525   ₹139,792    
  LTTS        9      IT                    2.424    -0.19  +66.5%    +52.5%    2,258.9     61     ₹137,794    
  ASIANPAINT  10     CHEMICALS             2.348    -0.15  +56.5%    +36.2%    2,656.7     52     ₹138,147    
  SRF         12     CAPITAL_GOODS         2.097    -0.12  +65.8%    +38.1%    1,114.9     125    ₹139,363    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ADANIGREEN  1      ENERGY                01-Sep-20   494.6       1,065.9     227    ₹129,685      +115.5%   
  ADANIENT    5      ENERGY                01-Oct-20   307.4       490.0       389    ₹71,014       +59.4%    
  TATACOMM    8      TELECOM               03-Aug-20   740.4       1,020.5     151    ₹42,295       +37.8%    
  SYNGENE     25     HEALTHCARE            03-Aug-20   474.9       627.9       235    ₹35,945       +32.2%    
  WIPRO       31     IT                    01-Oct-20   144.3       178.9       829    ₹28,676       +24.0%    
  NAVINFLUOR  11     PHARMA                02-Nov-20   2,141.1     2,597.5     57     ₹26,019       +21.3%    
  DIVISLAB    7      PHARMA                01-Sep-20   3,134.3     3,733.7     35     ₹20,980       +19.1%    
  TATACHEM    4      CHEMICALS             01-Dec-20   377.5       447.5       362    ₹25,313       +18.5%    
  HAVELLS     13     CAPITAL_GOODS         02-Nov-20   752.1       875.5       162    ₹20,000       +16.4%    
  INFY        18     IT                    01-Dec-20   1,001.7     1,109.6     136    ₹14,678       +10.8%    
  JKCEMENT    24     CEMENT                02-Nov-20   1,770.4     1,880.8     68     ₹7,508        +6.2%     
  DALBHARAT   33     CHEMICALS             01-Dec-20   1,126.4     1,070.6     121    ₹-6,758       -5.0%     

  AFTER: Invested ₹2,707,945 | Cash ₹87,819 | Total ₹2,795,764 | Positions 18/20 | Slot ₹139,838

========================================================================
  REBALANCE #39  —  01 Feb 2021
  NAV: ₹2,781,345  |  Slot: ₹139,067  |  Cash: ₹87,819
========================================================================

  EXITS (6)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TATACOMM    40     TELECOM               03-Aug-20   740.4       916.7       151    ₹26,612       +23.8%    182d  
  JKCEMENT    46     CEMENT                02-Nov-20   1,770.4     2,116.9     68     ₹23,558       +19.6%    91d   
  SYNGENE     48     HEALTHCARE            03-Aug-20   474.9       557.7       235    ₹19,448       +17.4%    182d  
  NAVINFLUOR  68     PHARMA                02-Nov-20   2,141.1     2,310.0     57     ₹9,628        +7.9%     91d   
  SRF         52     CAPITAL_GOODS         01-Jan-21   1,114.9     1,081.4     125    ₹-4,189       -3.0%     31d   
  ASIANPAINT  101    CHEMICALS             01-Jan-21   2,656.7     2,343.0     52     ₹-16,313      -11.8%    31d   

  ENTRIES (6)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  CUMMINSIND  6      CAPITAL_GOODS         2.518    0.22   +27.6%    +68.2%    673.0       206    ₹138,632    
  SFL         7      CONSUMER_DISCRETIONARY  2.471    0.12   +33.5%    +50.3%    983.1       141    ₹138,621    
  CROMPTON    8      CONSUMER_DURABLES     2.329    -0.05  +62.7%    +40.6%    400.6       347    ₹139,010    
  HINDZINC    9      CHEMICALS             2.236    0.10   +75.0%    +42.0%    188.7       737    ₹139,047    
  ADANIPORTS  11     INFRASTRUCTURE        2.156    -0.22  +47.1%    +50.8%    526.9       263    ₹138,563    
  BAJAJ-AUTO  12     AUTO                  2.125    -0.00  +40.5%    +42.5%    3,598.2     38     ₹136,733    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ADANIGREEN  1      ENERGY                01-Sep-20   494.6       1,024.2     227    ₹120,219      +107.1%   
  ADANIENT    3      ENERGY                01-Oct-20   307.4       535.9       389    ₹88,885       +74.3%    
  HAVELLS     4      CAPITAL_GOODS         02-Nov-20   752.1       1,024.8     162    ₹44,187       +36.3%    
  WIPRO       19     IT                    01-Oct-20   144.3       194.7       829    ₹41,801       +34.9%    
  TATACHEM    5      CHEMICALS             01-Dec-20   377.5       460.6       362    ₹30,072       +22.0%    
  INFY        32     IT                    01-Dec-20   1,001.7     1,110.0     136    ₹14,732       +10.8%    
  DIVISLAB    30     PHARMA                01-Sep-20   3,134.3     3,359.8     35     ₹7,894        +7.2%     
  DALBHARAT   29     CHEMICALS             01-Dec-20   1,126.4     1,192.6     121    ₹8,013        +5.9%     
  ATGL        2      OIL_GAS               01-Jan-21   376.3       388.6       371    ₹4,557        +3.3%     
  LTTS        17     IT                    01-Jan-21   2,258.9     2,321.4     61     ₹3,814        +2.8%     
  TATASTEEL   15     METALS                01-Jan-21   55.4        54.8        2525   ₹-1,522       -1.1%     
  SAIL        10     METALS                01-Jan-21   63.2        54.0        2211   ₹-20,458      -14.6%    

  AFTER: Invested ₹2,722,039 | Cash ₹58,320 | Total ₹2,780,359 | Positions 18/20 | Slot ₹139,067

========================================================================
  REBALANCE #40  —  01 Mar 2021
  NAV: ₹3,196,862  |  Slot: ₹159,843  |  Cash: ₹58,320
========================================================================

  EXITS (6)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TATASTEEL   44     METALS                01-Jan-21   55.4        62.9        2525   ₹18,977       +13.6%    59d   
  INFY        56     IT                    01-Dec-20   1,001.7     1,115.4     136    ₹15,462       +11.4%    90d   
  DIVISLAB    118    PHARMA                01-Sep-20   3,134.3     3,357.8     35     ₹7,824        +7.1%     181d  
  SFL         50     CONSUMER_DISCRETIONARY  01-Feb-21   983.1       1,004.2     141    ₹2,968        +2.1%     28d   
  CROMPTON    55     CONSUMER_DURABLES     01-Feb-21   400.6       371.8       347    ₹-9,997       -7.2%     28d   
  BAJAJ-AUTO  74     AUTO                  01-Feb-21   3,598.2     3,335.3     38     ₹-9,991       -7.3%     28d   

  ENTRIES (5)
  [52w filter blocked 1: BANKINDIA(-20.2%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  THERMAX     6      CAPITAL_GOODS         2.742    -0.03  +49.7%    +56.5%    1,370.5     116    ₹158,983    
  GUJGASLTD   7      ENERGY                2.626    0.03   +88.2%    +52.8%    510.5       313    ₹159,799    
  TATAPOWER   8      ENERGY                2.444    0.09   +104.8%   +50.8%    94.1        1699   ₹159,792    
  SUNDARMFIN  10     FINANCIAL_SERVICES    2.229    0.05   +63.2%    +50.3%    2,503.4     63     ₹157,715    
  NATIONALUM  11     METALS                2.224    -0.17  +68.3%    +54.5%    45.1        3540   ₹159,812    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ADANIENT    2      ENERGY                01-Oct-20   307.4       849.4       389    ₹210,851      +176.3%   
  ADANIGREEN  1      ENERGY                01-Sep-20   494.6       1,141.0     227    ₹146,721      +130.7%   
  TATACHEM    3      CHEMICALS             01-Dec-20   377.5       693.1       362    ₹114,245      +83.6%    
  HAVELLS     12     CAPITAL_GOODS         02-Nov-20   752.1       1,073.0     162    ₹51,994       +42.7%    
  ATGL        4      OIL_GAS               01-Jan-21   376.3       524.8       371    ₹55,077       +39.5%    
  WIPRO       40     IT                    01-Oct-20   144.3       191.4       829    ₹39,082       +32.7%    
  ADANIPORTS  5      INFRASTRUCTURE        01-Feb-21   526.9       672.6       263    ₹38,333       +27.7%    
  DALBHARAT   29     CHEMICALS             01-Dec-20   1,126.4     1,437.0     121    ₹37,577       +27.6%    
  CUMMINSIND  27     CAPITAL_GOODS         01-Feb-21   673.0       744.7       206    ₹14,772       +10.7%    
  LTTS        23     IT                    01-Jan-21   2,258.9     2,416.9     61     ₹9,639        +7.0%     
  SAIL        9      METALS                01-Jan-21   63.2        67.6        2211   ₹9,671        +6.9%     
  HINDZINC    16     CHEMICALS             01-Feb-21   188.7       196.1       737    ₹5,490        +3.9%     

  AFTER: Invested ₹3,109,318 | Cash ₹86,598 | Total ₹3,195,916 | Positions 17/20 | Slot ₹159,843

========================================================================
  REBALANCE #41  —  01 Apr 2021
  NAV: ₹3,562,853  |  Slot: ₹178,143  |  Cash: ₹86,598
========================================================================

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HAVELLS     73     CAPITAL_GOODS         02-Nov-20   752.1       1,021.2     162    ₹43,601       +35.8%    150d  
  WIPRO       75     IT                    01-Oct-20   144.3       192.4       829    ₹39,847       +33.3%    182d  
  LTTS        62     IT                    01-Jan-21   2,258.9     2,550.6     61     ₹17,791       +12.9%    90d   
  NATIONALUM  —      METALS                01-Mar-21   45.1        46.0        3540   ₹3,031        +1.9%     31d   
  HINDZINC    65     CHEMICALS             01-Feb-21   188.7       183.1       737    ₹-4,093       -2.9%     59d   

  ENTRIES (4)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  DEEPAKNTR   5      CHEMICALS             3.258    0.00   +338.0%   +77.5%    1,618.8     110    ₹178,067    
  GRASIM      6      CEMENT                3.043    0.06   +217.5%   +55.7%    1,412.7     126    ₹178,002    
  TATAELXSI   7      IT                    2.997    0.02   +341.6%   +49.7%    2,645.7     67     ₹177,262    
  DIXON       8      CONSUMER_DURABLES     2.914    0.03   +429.0%   +34.2%    3,580.2     49     ₹175,432    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ADANIENT    2      ENERGY                01-Oct-20   307.4       1,104.5     389    ₹310,077      +259.3%   
  ATGL        1      OIL_GAS               01-Jan-21   376.3       1,059.3     371    ₹253,378      +181.5%   
  ADANIGREEN  4      ENERGY                01-Sep-20   494.6       1,160.1     227    ₹151,046      +134.5%   
  TATACHEM    3      CHEMICALS             01-Dec-20   377.5       726.4       362    ₹126,277      +92.4%    
  ADANIPORTS  11     INFRASTRUCTURE        01-Feb-21   526.9       714.7       263    ₹49,400       +35.7%    
  DALBHARAT   12     CHEMICALS             01-Dec-20   1,126.4     1,526.2     121    ₹48,377       +35.5%    
  CUMMINSIND  14     CAPITAL_GOODS         01-Feb-21   673.0       832.8       206    ₹32,917       +23.7%    
  SAIL        31     METALS                01-Jan-21   63.2        72.3        2211   ₹20,063       +14.3%    
  TATAPOWER   22     ENERGY                01-Mar-21   94.1        100.9       1699   ₹11,670       +7.3%     
  GUJGASLTD   20     ENERGY                01-Mar-21   510.5       524.7       313    ₹4,422        +2.8%     
  SUNDARMFIN  29     FINANCIAL_SERVICES    01-Mar-21   2,503.4     2,514.3     63     ₹683          +0.4%     
  THERMAX     23     CAPITAL_GOODS         01-Mar-21   1,370.5     1,300.7     116    ₹-8,098       -5.1%     

  AFTER: Invested ₹3,406,735 | Cash ₹155,277 | Total ₹3,562,011 | Positions 16/20 | Slot ₹178,143

========================================================================
  REBALANCE #42  —  03 May 2021
  NAV: ₹3,842,640  |  Slot: ₹192,132  |  Cash: ₹155,277
========================================================================
  [SECTOR CAP≤4] dropped: VEDL

  EXITS (3)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  CUMMINSIND  26     CAPITAL_GOODS         01-Feb-21   673.0       779.0       206    ₹21,833       +15.7%    91d   
  GUJGASLTD   25     ENERGY                01-Mar-21   510.5       509.9       313    ₹-210         -0.1%     63d   
  SUNDARMFIN  27     FINANCIAL_SERVICES    01-Mar-21   2,503.4     2,399.1     63     ₹-6,571       -4.2%     63d   

  ENTRIES (3)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  JSWSTEEL    3      METALS                4.261    0.22   +345.4%   +92.2%    684.6       280    ₹191,700    
  TATASTEEL   7      METALS                3.341    0.45   +286.5%   +70.1%    91.7        2096   ₹192,123    
  PERSISTENT  8      IT                    3.185    0.23   +357.7%   +43.2%    1,036.6     185    ₹191,767    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ADANIENT    2      ENERGY                01-Oct-20   307.4       1,252.5     389    ₹367,646      +307.4%   
  ATGL        1      OIL_GAS               01-Jan-21   376.3       1,235.6     371    ₹318,792      +228.3%   
  ADANIGREEN  20     ENERGY                01-Sep-20   494.6       1,036.5     227    ₹123,000      +109.5%   
  TATACHEM    12     CHEMICALS             01-Dec-20   377.5       730.2       362    ₹127,644      +93.4%    
  SAIL        4      METALS                01-Jan-21   63.2        110.2       2211   ₹103,776      +74.2%    
  ADANIPORTS  18     INFRASTRUCTURE        01-Feb-21   526.9       739.1       263    ₹55,821       +40.3%    
  DALBHARAT   19     CHEMICALS             01-Dec-20   1,126.4     1,512.6     121    ₹46,732       +34.3%    
  TATAELXSI   11     IT                    01-Apr-21   2,645.7     3,374.4     67     ₹48,821       +27.5%    
  DIXON       6      CONSUMER_DURABLES     01-Apr-21   3,580.2     4,241.7     49     ₹32,411       +18.5%    
  DEEPAKNTR   5      CHEMICALS             01-Apr-21   1,618.8     1,871.8     110    ₹27,828       +15.6%    
  THERMAX     16     CAPITAL_GOODS         01-Mar-21   1,370.5     1,500.1     116    ₹15,025       +9.5%     
  TATAPOWER   21     ENERGY                01-Mar-21   94.1        95.9        1699   ₹3,183        +2.0%     
  GRASIM      17     CEMENT                01-Apr-21   1,412.7     1,373.9     126    ₹-4,889       -2.7%     

  AFTER: Invested ₹3,791,756 | Cash ₹50,201 | Total ₹3,841,957 | Positions 16/20 | Slot ₹192,132

========================================================================
  REBALANCE #43  —  01 Jun 2021
  NAV: ₹4,045,535  |  Slot: ₹202,277  |  Cash: ₹50,201
========================================================================

  EXITS (3)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TATACHEM    75     CHEMICALS             01-Dec-20   377.5       658.6       362    ₹101,740      +74.4%    182d  
  TATAPOWER   41     ENERGY                01-Mar-21   94.1        100.5       1699   ₹11,017       +6.9%     92d   
  THERMAX     54     CAPITAL_GOODS         01-Mar-21   1,370.5     1,384.1     116    ₹1,570        +1.0%     92d   

  ENTRIES (3)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  LAURUSLABS  5      PHARMA                3.598    0.43   +472.8%   +48.1%    524.5       385    ₹201,927    
  JSWENERGY   8      ENERGY                2.924    0.33   +221.7%   +73.0%    122.2       1655   ₹202,252    
  BSE         9      FINANCIAL_SERVICES    2.906    0.40   +155.2%   +60.8%    97.3        2078   ₹202,209    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ADANIENT    2      ENERGY                01-Oct-20   307.4       1,412.8     389    ₹430,006      +359.6%   
  ATGL        1      OIL_GAS               01-Jan-21   376.3       1,438.7     371    ₹394,155      +282.3%   
  ADANIGREEN  15     ENERGY                01-Sep-20   494.6       1,271.1     227    ₹176,243      +157.0%   
  SAIL        10     METALS                01-Jan-21   63.2        104.5       2211   ₹91,286       +65.3%    
  DALBHARAT   11     CHEMICALS             01-Dec-20   1,126.4     1,775.6     121    ₹78,551       +57.6%    
  ADANIPORTS  27     INFRASTRUCTURE        01-Feb-21   526.9       774.8       263    ₹65,216       +47.1%    
  TATAELXSI   6      IT                    01-Apr-21   2,645.7     3,444.9     67     ₹53,549       +30.2%    
  PERSISTENT  4      IT                    03-May-21   1,036.6     1,202.4     185    ₹30,671       +16.0%    
  DIXON       23     CONSUMER_DURABLES     01-Apr-21   3,580.2     4,098.5     49     ₹25,393       +14.5%    
  DEEPAKNTR   13     CHEMICALS             01-Apr-21   1,618.8     1,730.5     110    ₹12,284       +6.9%     
  TATASTEEL   7      METALS                03-May-21   91.7        94.8        2096   ₹6,523        +3.4%     
  GRASIM      25     CEMENT                01-Apr-21   1,412.7     1,403.1     126    ₹-1,218       -0.7%     
  JSWSTEEL    3      METALS                03-May-21   684.6       657.4       280    ₹-7,620       -4.0%     

  AFTER: Invested ₹4,031,947 | Cash ₹12,868 | Total ₹4,044,815 | Positions 16/20 | Slot ₹202,277

========================================================================
  REBALANCE #44  —  01 Jul 2021
  NAV: ₹4,102,797  |  Slot: ₹205,140  |  Cash: ₹12,868
========================================================================

  EXITS (3)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIGREEN  119    ENERGY                01-Sep-20   494.6       1,068.7     227    ₹130,309      +116.1%   303d  
  ADANIPORTS  149    INFRASTRUCTURE        01-Feb-21   526.9       687.3       263    ₹42,195       +30.5%    150d  
  GRASIM      54     CEMENT                01-Apr-21   1,412.7     1,457.2     126    ₹5,605        +3.1%     91d   

  ENTRIES (3)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  ZYDUSLIFE   10     PHARMA                2.351    0.29   +81.0%    +47.1%    616.5       332    ₹204,677    
  SRF         13     CAPITAL_GOODS         2.246    0.59   +104.8%   +36.0%    1,447.1     141    ₹204,035    
  COFORGE     14     IT                    2.237    0.52   +198.7%   +42.2%    790.4       259    ₹204,720    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ADANIENT    2      ENERGY                01-Oct-20   307.4       1,487.6     389    ₹459,095      +383.9%   
  ATGL        12     OIL_GAS               01-Jan-21   376.3       967.8       371    ₹219,451      +157.2%   
  SAIL        8      METALS                01-Jan-21   63.2        109.5       2211   ₹102,251      +73.1%    
  DALBHARAT   25     CHEMICALS             01-Dec-20   1,126.4     1,848.9     121    ₹87,422       +64.1%    
  TATAELXSI   3      IT                    01-Apr-21   2,645.7     4,102.5     67     ₹97,609       +55.1%    
  JSWENERGY   4      ENERGY                01-Jun-21   122.2       169.1       1655   ₹77,610       +38.4%    
  PERSISTENT  5      IT                    03-May-21   1,036.6     1,427.2     185    ₹72,258       +37.7%    
  LAURUSLABS  1      PHARMA                01-Jun-21   524.5       656.5       385    ₹50,813       +25.2%    
  DIXON       11     CONSUMER_DURABLES     01-Apr-21   3,580.2     4,401.9     49     ₹40,261       +22.9%    
  DEEPAKNTR   21     CHEMICALS             01-Apr-21   1,618.8     1,850.4     110    ₹25,476       +14.3%    
  TATASTEEL   7      METALS                03-May-21   91.7        102.4       2096   ₹22,526       +11.7%    
  BSE         9      FINANCIAL_SERVICES    01-Jun-21   97.3        93.2        2078   ₹-8,542       -4.2%     
  JSWSTEEL    6      METALS                03-May-21   684.6       644.2       280    ₹-11,318      -5.9%     

  AFTER: Invested ₹4,096,401 | Cash ₹5,667 | Total ₹4,102,068 | Positions 16/20 | Slot ₹205,140

========================================================================
  REBALANCE #45  —  02 Aug 2021
  NAV: ₹4,476,158  |  Slot: ₹223,808  |  Cash: ₹5,667
========================================================================

  EXITS (4)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ATGL        54     OIL_GAS               01-Jan-21   376.3       887.4       371    ₹189,600      +135.8%   213d  
  DEEPAKNTR   46     CHEMICALS             01-Apr-21   1,618.8     2,032.0     110    ₹45,454       +25.5%    123d  
  DIXON       69     CONSUMER_DURABLES     01-Apr-21   3,580.2     4,339.1     49     ₹37,186       +21.2%    123d  
  ZYDUSLIFE   147    PHARMA                01-Jul-21   616.5       574.2       332    ₹-14,048      -6.9%     32d   

  ENTRIES (4)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  CRISIL      10     FINANCIAL_SERVICES    2.522    0.05   +82.2%    +58.8%    2,800.0     79     ₹221,201    
  GUJGASLTD   11     ENERGY                2.488    0.41   +167.8%   +41.7%    722.7       309    ₹223,325    
  ABBOTINDIA  13     PHARMA                2.272    -0.01  +37.1%    +32.3%    18,684.8    11     ₹205,533    
  MPHASIS     14     IT                    2.239    0.36   +130.0%   +47.1%    2,352.2     95     ₹223,458    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ADANIENT    2      ENERGY                01-Oct-20   307.4       1,435.6     389    ₹438,883      +367.0%   
  JSWENERGY   1      ENERGY                01-Jun-21   122.2       238.1       1655   ₹191,878      +94.9%    
  SAIL        38     METALS                01-Jan-21   63.2        120.3       2211   ₹126,182      +90.2%    
  DALBHARAT   9      CHEMICALS             01-Dec-20   1,126.4     2,114.2     121    ₹119,519      +87.7%    
  TATAELXSI   8      IT                    01-Apr-21   2,645.7     4,082.3     67     ₹96,252       +54.3%    
  PERSISTENT  4      IT                    03-May-21   1,036.6     1,515.7     185    ₹88,628       +46.2%    
  BSE         3      FINANCIAL_SERVICES    01-Jun-21   97.3        133.3       2078   ₹74,731       +37.0%    
  TATASTEEL   5      METALS                03-May-21   91.7        124.1       2096   ₹68,000       +35.4%    
  LAURUSLABS  7      PHARMA                01-Jun-21   524.5       644.6       385    ₹46,263       +22.9%    
  COFORGE     6      IT                    01-Jul-21   790.4       969.2       259    ₹46,303       +22.6%    
  SRF         12     CAPITAL_GOODS         01-Jul-21   1,447.1     1,773.4     141    ₹46,020       +22.6%    
  JSWSTEEL    20     METALS                03-May-21   684.6       713.8       280    ₹8,151        +4.3%     

  AFTER: Invested ₹4,388,029 | Cash ₹87,092 | Total ₹4,475,121 | Positions 16/20 | Slot ₹223,808

========================================================================
  REBALANCE #46  —  01 Sep 2021
  NAV: ₹4,548,237  |  Slot: ₹227,412  |  Cash: ₹87,092
========================================================================
  [SECTOR CAP≤4] dropped: TECHM, LTTS

  EXITS (3)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SAIL        80     METALS                01-Jan-21   63.2        103.5       2211   ₹89,093       +63.7%    243d  
  JSWSTEEL    83     METALS                03-May-21   684.6       646.8       280    ₹-10,598      -5.5%     121d  
  CRISIL      47     FINANCIAL_SERVICES    02-Aug-21   2,800.0     2,569.8     79     ₹-18,184      -8.2%     30d   

  ENTRIES (3)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  ATGL        2      OIL_GAS               3.515    0.63   +717.5%   -4.3%     1,509.2     150    ₹226,375    
  APOLLOHOSP  4      HEALTHCARE            3.378    0.44   +206.1%   +54.0%    4,977.0     45     ₹223,967    
  BAJAJFINSV  9      FINANCIAL_SERVICES    2.687    0.93   +157.1%   +42.0%    1,675.3     135    ₹226,161    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ADANIENT    6      ENERGY                01-Oct-20   307.4       1,562.4     389    ₹488,198      +408.3%   
  JSWENERGY   1      ENERGY                01-Jun-21   122.2       249.7       1655   ₹210,977      +104.3%   
  DALBHARAT   17     CHEMICALS             01-Dec-20   1,126.4     2,094.6     121    ₹117,152      +86.0%    
  TATAELXSI   3      IT                    01-Apr-21   2,645.7     4,687.2     67     ₹136,779      +77.2%    
  PERSISTENT  8      IT                    03-May-21   1,036.6     1,610.3     185    ₹106,138      +55.3%    
  TATASTEEL   11     METALS                03-May-21   91.7        124.2       2096   ₹68,212       +35.5%    
  SRF         5      CAPITAL_GOODS         01-Jul-21   1,447.1     1,958.2     141    ₹72,071       +35.3%    
  BSE         31     FINANCIAL_SERVICES    01-Jun-21   97.3        126.6       2078   ₹60,796       +30.1%    
  LAURUSLABS  23     PHARMA                01-Jun-21   524.5       652.7       385    ₹49,372       +24.5%    
  COFORGE     15     IT                    01-Jul-21   790.4       976.6       259    ₹48,232       +23.6%    
  MPHASIS     10     IT                    02-Aug-21   2,352.2     2,538.1     95     ₹17,664       +7.9%     
  ABBOTINDIA  33     PHARMA                02-Aug-21   18,684.8    18,947.9    11     ₹2,894        +1.4%     
  GUJGASLTD   37     ENERGY                02-Aug-21   722.7       674.6       309    ₹-14,872      -6.7%     

  AFTER: Invested ₹4,524,605 | Cash ₹22,828 | Total ₹4,547,433 | Positions 16/20 | Slot ₹227,412

========================================================================
  REBALANCE #47  —  01 Oct 2021
  NAV: ₹4,790,790  |  Slot: ₹239,540  |  Cash: ₹22,828
========================================================================

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIENT    —      ENERGY                01-Oct-20   307.4       1,456.1     389    ₹446,844      +373.7%   365d  
  COFORGE     56     IT                    01-Jul-21   790.4       997.8       259    ₹53,712       +26.2%    92d   
  LAURUSLABS  119    PHARMA                01-Jun-21   524.5       609.9       385    ₹32,877       +16.3%    122d  
  APOLLOHOSP  55     HEALTHCARE            01-Sep-21   4,977.0     4,404.5     45     ₹-25,763      -11.5%    30d   
  GUJGASLTD   136    ENERGY                02-Aug-21   722.7       589.5       309    ₹-41,176      -18.4%    60d   

  ENTRIES (6)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  IRCTC       3      CONSUMER_DISCRETIONARY  3.418    0.73   +181.5%   +86.7%    725.8       330    ₹239,506    
  OIL         6      OIL_GAS               3.162    0.55   +196.4%   +54.8%    132.1       1813   ₹239,412    
  LINDEINDIA  7      CHEMICALS             3.102    0.36   +263.3%   +53.5%    2,601.0     92     ₹239,295    
  LTTS        9      IT                    2.728    0.81   +194.6%   +61.4%    4,375.8     54     ₹236,294    
  HATSUN      10     FMCG                  2.633    0.64   +153.0%   +60.0%    1,365.3     175    ₹238,920    
  BAJAJHLDNG  11     FINANCIAL_SERVICES    2.465    0.47   +99.7%    +34.8%    4,488.4     53     ₹237,884    

  HOLDS (11)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  JSWENERGY   1      ENERGY                01-Jun-21   122.2       376.7       1655   ₹421,234      +208.3%   
  TATAELXSI   4      IT                    01-Apr-21   2,645.7     5,590.4     67     ₹197,294      +111.3%   
  DALBHARAT   36     CHEMICALS             01-Dec-20   1,126.4     2,070.5     121    ₹114,231      +83.8%    
  PERSISTENT  20     IT                    03-May-21   1,036.6     1,760.2     185    ₹133,863      +69.8%    
  SRF         5      CAPITAL_GOODS         01-Jul-21   1,447.1     2,186.9     141    ₹104,320      +51.1%    
  BSE         30     FINANCIAL_SERVICES    01-Jun-21   97.3        129.4       2078   ₹66,606       +32.9%    
  TATASTEEL   22     METALS                03-May-21   91.7        114.2       2096   ₹47,301       +24.6%    
  MPHASIS     15     IT                    02-Aug-21   2,352.2     2,770.0     95     ₹39,695       +17.8%    
  ABBOTINDIA  27     PHARMA                02-Aug-21   18,684.8    20,842.3    11     ₹23,732       +11.5%    
  BAJAJFINSV  8      FINANCIAL_SERVICES    01-Sep-21   1,675.3     1,714.3     135    ₹5,275        +2.3%     
  ATGL        2      OIL_GAS               01-Sep-21   1,509.2     1,422.0     150    ₹-13,073      -5.8%     

  AFTER: Invested ₹4,759,261 | Cash ₹29,830 | Total ₹4,789,091 | Positions 17/20 | Slot ₹239,540

========================================================================
  REBALANCE #48  —  01 Nov 2021
  NAV: ₹4,789,026  |  Slot: ₹239,451  |  Cash: ₹29,830
========================================================================

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DALBHARAT   82     CHEMICALS             01-Dec-20   1,126.4     1,973.5     121    ₹102,501      +75.2%    335d  
  BSE         47     FINANCIAL_SERVICES    01-Jun-21   97.3        143.5       2078   ₹96,013       +47.5%    153d  
  TATASTEEL   —      METALS                03-May-21   91.7        120.1       2096   ₹59,670       +31.1%    182d  
  BAJAJFINSV  —      FINANCIAL_SERVICES    01-Sep-21   1,675.3     1,753.1     135    ₹10,501       +4.6%     61d   
  ABBOTINDIA  120    PHARMA                02-Aug-21   18,684.8    18,879.9    11     ₹2,146        +1.0%     91d   

  ENTRIES (5)
  [52w filter blocked 1: DMART(-22.6%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  TITAN       6      CONSUMER_DISCRETIONARY  2.743    0.87   +98.4%    +40.3%    2,375.1     100    ₹237,513    
  SOLARINDS   9      CHEMICALS             2.605    0.65   +135.7%   +42.3%    2,402.7     99     ₹237,866    
  IOC         13     OIL_GAS               2.392    0.84   +90.0%    +29.5%    65.7        3644   ₹239,442    
  GRASIM      14     CEMENT                2.334    0.83   +130.7%   +16.0%    1,748.1     136    ₹237,738    
  TIINDIA     16     AUTO_ANCILLARY        2.305    0.60   +132.9%   +29.3%    1,436.6     166    ₹238,473    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  JSWENERGY   2      ENERGY                01-Jun-21   122.2       334.7       1655   ₹351,715      +173.9%   
  TATAELXSI   5      IT                    01-Apr-21   2,645.7     5,775.4     67     ₹209,691      +118.3%   
  PERSISTENT  4      IT                    03-May-21   1,036.6     1,957.1     185    ₹170,299      +88.8%    
  SRF         17     CAPITAL_GOODS         01-Jul-21   1,447.1     2,092.4     141    ₹90,995       +44.6%    
  MPHASIS     10     IT                    02-Aug-21   2,352.2     3,039.8     95     ₹65,321       +29.2%    
  IRCTC       3      CONSUMER_DISCRETIONARY  01-Oct-21   725.8       818.3       330    ₹30,535       +12.7%    
  LTTS        15     IT                    01-Oct-21   4,375.8     4,629.8     54     ₹13,717       +5.8%     
  BAJAJHLDNG  12     FINANCIAL_SERVICES    01-Oct-21   4,488.4     4,513.3     53     ₹1,323        +0.6%     
  HATSUN      7      FMCG                  01-Oct-21   1,365.3     1,340.3     175    ₹-4,361       -1.8%     
  ATGL        1      OIL_GAS               01-Sep-21   1,509.2     1,433.3     150    ₹-11,387      -5.0%     
  OIL         8      OIL_GAS               01-Oct-21   132.1       119.0       1813   ₹-23,649      -9.9%     
  LINDEINDIA  11     CHEMICALS             01-Oct-21   2,601.0     2,333.4     92     ₹-24,617      -10.3%    

  AFTER: Invested ₹4,717,074 | Cash ₹70,537 | Total ₹4,787,612 | Positions 17/20 | Slot ₹239,451

========================================================================
  REBALANCE #49  —  01 Dec 2021
  NAV: ₹4,784,572  |  Slot: ₹239,229  |  Cash: ₹70,537
========================================================================

  EXITS (2)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SRF         54     CAPITAL_GOODS         01-Jul-21   1,447.1     1,988.5     141    ₹76,342       +37.4%    153d  
  IOC         41     OIL_GAS               01-Nov-21   65.7        60.9        3644   ₹-17,411      -7.3%     30d   

  ENTRIES (2)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  BSE         7      FINANCIAL_SERVICES    2.882    0.81   +193.8%   +38.0%    175.3       1364   ₹239,113    
  ZEEL        9      MEDIA                 2.693    1.08   +82.7%    +97.6%    324.0       738    ₹239,076    

  HOLDS (15)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  JSWENERGY   3      ENERGY                01-Jun-21   122.2       295.0       1655   ₹286,000      +141.4%   
  TATAELXSI   4      IT                    01-Apr-21   2,645.7     5,687.7     67     ₹203,814      +115.0%   
  PERSISTENT  2      IT                    03-May-21   1,036.6     2,032.5     185    ₹184,246      +96.1%    
  MPHASIS     29     IT                    02-Aug-21   2,352.2     2,756.6     95     ₹38,421       +17.2%    
  SOLARINDS   1      CHEMICALS             01-Nov-21   2,402.7     2,813.8     99     ₹40,698       +17.1%    
  LTTS        5      IT                    01-Oct-21   4,375.8     5,055.3     54     ₹36,691       +15.5%    
  TIINDIA     23     AUTO_ANCILLARY        01-Nov-21   1,436.6     1,632.3     166    ₹32,494       +13.6%    
  BAJAJHLDNG  11     FINANCIAL_SERVICES    01-Oct-21   4,488.4     4,913.8     53     ₹22,549       +9.5%     
  ATGL        6      OIL_GAS               01-Sep-21   1,509.2     1,624.3     150    ₹17,268       +7.6%     
  IRCTC       8      CONSUMER_DISCRETIONARY  01-Oct-21   725.8       776.7       330    ₹16,803       +7.0%     
  TITAN       14     CONSUMER_DISCRETIONARY  01-Nov-21   2,375.1     2,329.6     100    ₹-4,550       -1.9%     
  LINDEINDIA  17     CHEMICALS             01-Oct-21   2,601.0     2,548.7     92     ₹-4,812       -2.0%     
  GRASIM      18     CEMENT                01-Nov-21   1,748.1     1,641.2     136    ₹-14,538      -6.1%     
  HATSUN      37     FMCG                  01-Oct-21   1,365.3     1,267.9     175    ₹-17,033      -7.1%     
  OIL         12     OIL_GAS               01-Oct-21   132.1       115.3       1813   ₹-30,439      -12.7%    

  AFTER: Invested ₹4,689,815 | Cash ₹94,189 | Total ₹4,784,004 | Positions 17/20 | Slot ₹239,229

========================================================================
  REBALANCE #50  —  03 Jan 2022
  NAV: ₹4,938,039  |  Slot: ₹246,902  |  Cash: ₹94,189
========================================================================
  [SECTOR CAP≤4] dropped: TECHM, INFY

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LINDEINDIA  39     CHEMICALS             01-Oct-21   2,601.0     2,491.4     92     ₹-10,089      -4.2%     94d   
  ZEEL        95     MEDIA                 01-Dec-21   324.0       310.4       738    ₹-10,001      -4.2%     33d   
  GRASIM      47     CEMENT                01-Nov-21   1,748.1     1,623.6     136    ₹-16,930      -7.1%     63d   
  HATSUN      103    FMCG                  01-Oct-21   1,365.3     1,209.7     175    ₹-27,222      -11.4%    94d   
  OIL         120    OIL_GAS               01-Oct-21   132.1       106.9       1813   ₹-45,599      -19.0%    94d   

  ENTRIES (4)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  RAJESHEXPO  4      CONSUMER_DISCRETIONARY  3.293    0.30   +75.1%    +45.7%    851.1       290    ₹246,825    
  SCHAEFFLER  6      AUTO_ANCILLARY        3.072    0.47   +108.6%   +24.8%    1,757.9     140    ₹246,105    
  MAXHEALTH   8      HEALTHCARE            3.008    0.35   +202.4%   +21.2%    432.6       570    ₹246,599    
  THERMAX     9      CAPITAL_GOODS         2.548    0.40   +100.3%   +35.2%    1,787.7     138    ₹246,707    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  JSWENERGY   15     ENERGY                01-Jun-21   122.2       292.4       1655   ₹281,710      +139.3%   
  PERSISTENT  1      IT                    03-May-21   1,036.6     2,353.9     185    ₹243,697      +127.1%   
  TATAELXSI   10     IT                    01-Apr-21   2,645.7     5,697.6     67     ₹204,474      +115.4%   
  MPHASIS     18     IT                    02-Aug-21   2,352.2     3,132.9     95     ₹74,164       +33.2%    
  TIINDIA     7      AUTO_ANCILLARY        01-Nov-21   1,436.6     1,877.1     166    ₹73,122       +30.7%    
  LTTS        12     IT                    01-Oct-21   4,375.8     5,407.4     54     ₹55,706       +23.6%    
  BSE         2      FINANCIAL_SERVICES    01-Dec-21   175.3       202.3       1364   ₹36,781       +15.4%    
  ATGL        3      OIL_GAS               01-Sep-21   1,509.2     1,741.9     150    ₹34,904       +15.4%    
  BAJAJHLDNG  17     FINANCIAL_SERVICES    01-Oct-21   4,488.4     5,068.6     53     ₹30,754       +12.9%    
  IRCTC       21     CONSUMER_DISCRETIONARY  01-Oct-21   725.8       808.9       330    ₹27,441       +11.5%    
  TITAN       26     CONSUMER_DISCRETIONARY  01-Nov-21   2,375.1     2,491.2     100    ₹11,603       +4.9%     
  SOLARINDS   29     CHEMICALS             01-Nov-21   2,402.7     2,373.7     99     ₹-2,868       -1.2%     

  AFTER: Invested ₹4,745,488 | Cash ₹191,380 | Total ₹4,936,868 | Positions 16/20 | Slot ₹246,902

========================================================================
  REBALANCE #51  —  01 Feb 2022
  NAV: ₹4,913,502  |  Slot: ₹245,675  |  Cash: ₹191,380
========================================================================

  EXITS (4)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MPHASIS     49     IT                    02-Aug-21   2,352.2     2,880.5     95     ₹50,190       +22.5%    183d  
  LTTS        55     IT                    01-Oct-21   4,375.8     4,442.6     54     ₹3,607        +1.5%     123d  
  TITAN       52     CONSUMER_DISCRETIONARY  01-Nov-21   2,375.1     2,400.3     100    ₹2,517        +1.1%     92d   
  SOLARINDS   62     CHEMICALS             01-Nov-21   2,402.7     2,286.3     99     ₹-11,520      -4.8%     92d   

  ENTRIES (4)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  ADANIGREEN  5      ENERGY                3.557    0.66   +84.7%    +62.4%    1,904.0     129    ₹245,616    
  LINDEINDIA  8      CHEMICALS             2.922    0.40   +196.7%   +15.1%    2,686.0     91     ₹244,426    
  SRF         10     CAPITAL_GOODS         2.725    0.93   +128.2%   +15.3%    2,413.0     101    ₹243,710    
  ONGC        13     OIL_GAS               2.356    0.96   +100.5%   +15.7%    131.8       1864   ₹245,635    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TATAELXSI   4      IT                    01-Apr-21   2,645.7     7,216.5     67     ₹306,242      +172.8%   
  JSWENERGY   9      ENERGY                01-Jun-21   122.2       301.0       1655   ₹295,873      +146.3%   
  PERSISTENT  7      IT                    03-May-21   1,036.6     2,183.0     185    ₹212,093      +110.6%   
  ATGL        1      OIL_GAS               01-Sep-21   1,509.2     1,862.6     150    ₹53,011       +23.4%    
  BSE         2      FINANCIAL_SERVICES    01-Dec-21   175.3       211.4       1364   ₹49,222       +20.6%    
  TIINDIA     12     AUTO_ANCILLARY        01-Nov-21   1,436.6     1,676.5     166    ₹39,820       +16.7%    
  THERMAX     3      CAPITAL_GOODS         03-Jan-22   1,787.7     2,038.4     138    ₹34,588       +14.0%    
  IRCTC       18     CONSUMER_DISCRETIONARY  01-Oct-21   725.8       819.7       330    ₹30,993       +12.9%    
  BAJAJHLDNG  19     FINANCIAL_SERVICES    01-Oct-21   4,488.4     4,949.8     53     ₹24,457       +10.3%    
  SCHAEFFLER  6      AUTO_ANCILLARY        03-Jan-22   1,757.9     1,790.7     140    ₹4,587        +1.9%     
  RAJESHEXPO  11     CONSUMER_DISCRETIONARY  03-Jan-22   851.1       825.3       290    ₹-7,498       -3.0%     
  MAXHEALTH   21     HEALTHCARE            03-Jan-22   432.6       362.4       570    ₹-40,060      -16.2%    

  AFTER: Invested ₹4,721,582 | Cash ₹190,757 | Total ₹4,912,339 | Positions 16/20 | Slot ₹245,675

========================================================================
  REBALANCE #52  —  02 Mar 2022
  NAV: ₹4,673,467  |  Slot: ₹233,673  |  Cash: ₹190,757
========================================================================

  [REGIME OFF] Nifty 200 14,199.8 < SMA200 14,510.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (16)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  JSWENERGY   1      ENERGY                01-Jun-21   122.2       331.7       1655   ₹346,697      +171.4%   
  TATAELXSI   5      IT                    01-Apr-21   2,645.7     6,346.4     67     ₹247,945      +139.9%   
  PERSISTENT  13     IT                    03-May-21   1,036.6     1,886.5     185    ₹157,226      +82.0%    
  BSE         2      FINANCIAL_SERVICES    01-Dec-21   175.3       213.2       1364   ₹51,727       +21.6%    
  ATGL        7      OIL_GAS               01-Sep-21   1,509.2     1,666.0     150    ₹23,531       +10.4%    
  BAJAJHLDNG  36     FINANCIAL_SERVICES    01-Oct-21   4,488.4     4,833.5     53     ₹18,293       +7.7%     
  IRCTC       —      CONSUMER_DISCRETIONARY  01-Oct-21   725.8       775.6       330    ₹16,429       +6.9%     
  TIINDIA     64     AUTO_ANCILLARY        01-Nov-21   1,436.6     1,508.8     166    ₹11,990       +5.0%     
  THERMAX     27     CAPITAL_GOODS         03-Jan-22   1,787.7     1,809.5     138    ₹2,998        +1.2%     
  SCHAEFFLER  3      AUTO_ANCILLARY        03-Jan-22   1,757.9     1,771.1     140    ₹1,850        +0.8%     
  ADANIGREEN  6      ENERGY                01-Feb-22   1,904.0     1,882.6     129    ₹-2,767       -1.1%     
  ONGC        12     OIL_GAS               01-Feb-22   131.8       126.9       1864   ₹-9,029       -3.7%     
  LINDEINDIA  23     CHEMICALS             01-Feb-22   2,686.0     2,586.4     91     ₹-9,061       -3.7%     
  SRF         4      CAPITAL_GOODS         01-Feb-22   2,413.0     2,321.2     101    ₹-9,271       -3.8%     
  MAXHEALTH   17     HEALTHCARE            03-Jan-22   432.6       370.7       570    ₹-35,293      -14.3%    
  RAJESHEXPO  80     CONSUMER_DISCRETIONARY  03-Jan-22   851.1       682.8       290    ₹-48,812      -19.8%    

  AFTER: Invested ₹4,482,710 | Cash ₹190,757 | Total ₹4,673,467 | Positions 16/20 | Slot ₹233,673

========================================================================
  REBALANCE #53  —  01 Apr 2022
  NAV: ₹5,253,384  |  Slot: ₹262,669  |  Cash: ₹190,757
========================================================================

  EXITS (6)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BSE         —      FINANCIAL_SERVICES    01-Dec-21   175.3       297.8       1364   ₹167,020      +69.8%    121d  
  TIINDIA     85     AUTO_ANCILLARY        01-Nov-21   1,436.6     1,607.0     166    ₹28,286       +11.9%    151d  
  THERMAX     45     CAPITAL_GOODS         03-Jan-22   1,787.7     1,928.1     138    ₹19,371       +7.9%     88d   
  IRCTC       —      CONSUMER_DISCRETIONARY  01-Oct-21   725.8       765.4       330    ₹13,075       +5.5%     182d  
  RAJESHEXPO  79     CONSUMER_DISCRETIONARY  03-Jan-22   851.1       687.4       290    ₹-47,480      -19.2%    88d   
  MAXHEALTH   96     HEALTHCARE            03-Jan-22   432.6       349.2       570    ₹-47,578      -19.3%    88d   

  ENTRIES (6)
  [52w filter blocked 2: TTML(-39.7%), TRIDENT(-23.7%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  FLUOROCHEM  3      CHEMICALS             3.370    1.04   +388.0%   +22.2%    2,785.9     94     ₹261,873    
  HAL         8      CAPITAL_GOODS         2.407    0.84   +59.8%    +28.3%    723.8       362    ₹262,017    
  COALINDIA   9      METALS                2.360    0.81   +62.7%    +31.2%    137.5       1910   ₹262,628    
  SOLARINDS   12     CHEMICALS             2.236    0.85   +123.9%   +21.2%    2,851.5     92     ₹262,337    
  GAIL        13     OIL_GAS               2.221    0.74   +35.0%    +31.3%    93.1        2819   ₹262,580    
  TRENT       14     CONSUMER_DISCRETIONARY  2.201    0.96   +67.6%    +22.9%    1,263.3     207    ₹261,499    

  HOLDS (10)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TATAELXSI   2      IT                    01-Apr-21   2,645.7     8,616.2     67     ₹400,027      +225.7%   
  JSWENERGY   7      ENERGY                01-Jun-21   122.2       302.4       1655   ₹298,220      +147.4%   
  PERSISTENT  21     IT                    03-May-21   1,036.6     2,292.8     185    ₹232,395      +121.2%   
  ATGL        10     OIL_GAS               01-Sep-21   1,509.2     2,247.0     150    ₹110,674      +48.9%    
  LINDEINDIA  4      CHEMICALS             01-Feb-22   2,686.0     3,817.8     91     ₹102,993      +42.1%    
  BAJAJHLDNG  31     FINANCIAL_SERVICES    01-Oct-21   4,488.4     5,115.1     53     ₹33,214       +14.0%    
  SRF         11     CAPITAL_GOODS         01-Feb-22   2,413.0     2,590.2     101    ₹17,899       +7.3%     
  SCHAEFFLER  23     AUTO_ANCILLARY        03-Jan-22   1,757.9     1,848.8     140    ₹12,723       +5.2%     
  ADANIGREEN  5      ENERGY                01-Feb-22   1,904.0     1,945.1     129    ₹5,302        +2.2%     
  ONGC        18     OIL_GAS               01-Feb-22   131.8       130.8       1864   ₹-1,769       -0.7%     

  AFTER: Invested ₹5,045,644 | Cash ₹205,872 | Total ₹5,251,516 | Positions 16/20 | Slot ₹262,669

========================================================================
  REBALANCE #54  —  02 May 2022
  NAV: ₹5,238,612  |  Slot: ₹261,931  |  Cash: ₹205,872
========================================================================

  [REGIME OFF] Nifty 200 14,736.3 < SMA200 14,788.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (16)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TATAELXSI   17     IT                    01-Apr-21   2,645.7     7,417.0     67     ₹319,675      +180.3%   
  JSWENERGY   19     ENERGY                01-Jun-21   122.2       298.2       1655   ₹291,341      +144.0%   
  PERSISTENT  32     IT                    03-May-21   1,036.6     2,013.4     185    ₹180,712      +94.2%    
  ATGL        10     OIL_GAS               01-Sep-21   1,509.2     2,485.4     150    ₹146,439      +64.7%    
  ADANIGREEN  2      ENERGY                01-Feb-22   1,904.0     2,830.4     129    ₹119,499      +48.7%    
  LINDEINDIA  11     CHEMICALS             01-Feb-22   2,686.0     3,461.1     91     ₹70,532       +28.9%    
  SCHAEFFLER  3      AUTO_ANCILLARY        03-Jan-22   1,757.9     2,196.9     140    ₹61,458       +25.0%    
  BAJAJHLDNG  41     FINANCIAL_SERVICES    01-Oct-21   4,488.4     5,045.5     53     ₹29,526       +12.4%    
  HAL         23     CAPITAL_GOODS         01-Apr-22   723.8       748.0       362    ₹8,749        +3.3%     
  SRF         —      CAPITAL_GOODS         01-Feb-22   2,413.0     2,474.6     101    ₹6,220        +2.6%     
  SOLARINDS   7      CHEMICALS             01-Apr-22   2,851.5     2,900.5     92     ₹4,512        +1.7%     
  COALINDIA   22     METALS                01-Apr-22   137.5       138.6       1910   ₹2,116        +0.8%     
  FLUOROCHEM  5      CHEMICALS             01-Apr-22   2,785.9     2,720.4     94     ₹-6,156       -2.4%     
  GAIL        35     OIL_GAS               01-Apr-22   93.1        90.6        2819   ₹-7,292       -2.8%     
  TRENT       21     CONSUMER_DISCRETIONARY  01-Apr-22   1,263.3     1,215.2     207    ₹-9,947       -3.8%     
  ONGC        62     OIL_GAS               01-Feb-22   131.8       121.8       1864   ₹-18,612      -7.6%     

  AFTER: Invested ₹5,032,739 | Cash ₹205,872 | Total ₹5,238,612 | Positions 16/20 | Slot ₹261,931

========================================================================
  REBALANCE #55  —  01 Jun 2022
  NAV: ₹5,016,138  |  Slot: ₹250,807  |  Cash: ₹205,872
========================================================================

  [REGIME OFF] Nifty 200 14,082.9 < SMA200 14,803.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (16)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TATAELXSI   5      IT                    01-Apr-21   2,645.7     8,319.7     67     ₹380,155      +214.5%   
  JSWENERGY   50     ENERGY                01-Jun-21   122.2       260.6       1655   ₹229,074      +113.3%   
  PERSISTENT  63     IT                    03-May-21   1,036.6     1,815.3     185    ₹144,055      +75.1%    
  ATGL        8      OIL_GAS               01-Sep-21   1,509.2     2,323.3     150    ₹122,121      +53.9%    
  SCHAEFFLER  2      AUTO_ANCILLARY        03-Jan-22   1,757.9     2,324.5     140    ₹79,322       +32.2%    
  HAL         3      CAPITAL_GOODS         01-Apr-22   723.8       899.0       362    ₹63,437       +24.2%    
  LINDEINDIA  25     CHEMICALS             01-Feb-22   2,686.0     3,049.0     91     ₹33,032       +13.5%    
  BAJAJHLDNG  56     FINANCIAL_SERVICES    01-Oct-21   4,488.4     4,785.6     53     ₹15,755       +6.6%     
  COALINDIA   23     METALS                01-Apr-22   137.5       144.8       1910   ₹13,967       +5.3%     
  FLUOROCHEM  4      CHEMICALS             01-Apr-22   2,785.9     2,902.7     94     ₹10,977       +4.2%     
  SRF         —      CAPITAL_GOODS         01-Feb-22   2,413.0     2,368.5     101    ₹-4,491       -1.8%     
  ADANIGREEN  79     ENERGY                01-Feb-22   1,904.0     1,848.4     129    ₹-7,172       -2.9%     
  GAIL        45     OIL_GAS               01-Apr-22   93.1        85.6        2819   ₹-21,146      -8.1%     
  SOLARINDS   22     CHEMICALS             01-Apr-22   2,851.5     2,611.8     92     ₹-22,055      -8.4%     
  ONGC        81     OIL_GAS               01-Feb-22   131.8       116.7       1864   ₹-28,196      -11.5%    
  TRENT       52     CONSUMER_DISCRETIONARY  01-Apr-22   1,263.3     1,106.1     207    ₹-32,536      -12.4%    

  AFTER: Invested ₹4,810,265 | Cash ₹205,872 | Total ₹5,016,138 | Positions 16/20 | Slot ₹250,807

========================================================================
  REBALANCE #56  —  01 Jul 2022
  NAV: ₹4,715,459  |  Slot: ₹235,773  |  Cash: ₹205,872
========================================================================
  [SECTOR CAP≤4] dropped: COROMANDEL

  [REGIME OFF] Nifty 200 13,394.4 < SMA200 14,710.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (16)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TATAELXSI   12     IT                    01-Apr-21   2,645.7     7,920.9     67     ₹353,442      +199.4%   
  JSWENERGY   93     ENERGY                01-Jun-21   122.2       201.2       1655   ₹130,697      +64.6% ⚠  
  ATGL        10     OIL_GAS               01-Sep-21   1,509.2     2,386.3     150    ₹131,576      +58.1%    
  PERSISTENT  119    IT                    03-May-21   1,036.6     1,608.9     185    ₹105,872      +55.2% ⚠  
  SCHAEFFLER  3      AUTO_ANCILLARY        03-Jan-22   1,757.9     2,187.0     140    ₹60,076       +24.4%    
  LINDEINDIA  24     CHEMICALS             01-Feb-22   2,686.0     3,216.6     91     ₹48,286       +19.8%    
  HAL         8      CAPITAL_GOODS         01-Apr-22   723.8       824.4       362    ₹36,424       +13.9%    
  ADANIGREEN  21     ENERGY                01-Feb-22   1,904.0     1,963.7     129    ₹7,695        +3.1%     
  FLUOROCHEM  7      CHEMICALS             01-Apr-22   2,785.9     2,753.3     94     ₹-3,064       -1.2%     
  COALINDIA   36     METALS                01-Apr-22   137.5       135.4       1910   ₹-4,091       -1.6%     
  BAJAJHLDNG  51     FINANCIAL_SERVICES    01-Oct-21   4,488.4     4,353.1     53     ₹-7,169       -3.0%     
  SOLARINDS   18     CHEMICALS             01-Apr-22   2,851.5     2,713.6     92     ₹-12,685      -4.8%     
  SRF         —      CAPITAL_GOODS         01-Feb-22   2,413.0     2,135.5     101    ₹-28,020      -11.5%    
  TRENT       82     CONSUMER_DISCRETIONARY  01-Apr-22   1,263.3     1,070.3     207    ₹-39,945      -15.3%    
  GAIL        126    OIL_GAS               01-Apr-22   93.1        76.1        2819   ₹-48,125      -18.3% ⚠  
  ONGC        95     OIL_GAS               01-Feb-22   131.8       102.1       1864   ₹-55,348      -22.5% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): JSWENERGY, ONGC, PERSISTENT, GAIL

  AFTER: Invested ₹4,509,587 | Cash ₹205,872 | Total ₹4,715,459 | Positions 16/20 | Slot ₹235,773

========================================================================
  REBALANCE #57  —  01 Aug 2022
  NAV: ₹5,416,811  |  Slot: ₹270,841  |  Cash: ₹205,872
========================================================================
  [SECTOR CAP≤4] dropped: SIEMENS, CUMMINSIND, BEL

  EXITS (9)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  JSWENERGY   156    ENERGY                01-Jun-21   122.2       242.4       1655   ₹198,999      +98.4%    426d  
  PERSISTENT  —      IT                    03-May-21   1,036.6     1,780.3     185    ₹137,582      +71.7%    455d  
  ADANIGREEN  41     ENERGY                01-Feb-22   1,904.0     2,273.7     129    ₹47,691       +19.4%    181d  
  BAJAJHLDNG  61     FINANCIAL_SERVICES    01-Oct-21   4,488.4     5,007.5     53     ₹27,512       +11.6%    304d  
  TRENT       43     CONSUMER_DISCRETIONARY  01-Apr-22   1,263.3     1,297.1     207    ₹6,995        +2.7%     122d  
  SRF         —      CAPITAL_GOODS         01-Feb-22   2,413.0     2,427.5     101    ₹1,468        +0.6%     181d  
  SOLARINDS   74     CHEMICALS             01-Apr-22   2,851.5     2,684.1     92     ₹-15,399      -5.9%     122d  
  GAIL        110    OIL_GAS               01-Apr-22   93.1        85.2        2819   ₹-22,261      -8.5%     122d  
  ONGC        118    OIL_GAS               01-Feb-22   131.8       107.8       1864   ₹-44,676      -18.2%    181d  

  ENTRIES (10)
  [52w filter blocked 1: TTML(-60.6%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  TVSMOTOR    2      AUTO                  3.505    0.78   +62.0%    +47.7%    911.5       297    ₹270,713    
  M&M         3      AUTO                  3.397    0.97   +71.6%    +39.5%    1,206.3     224    ₹270,215    
  CGPOWER     4      CAPITAL_GOODS         3.366    0.86   +183.4%   +24.5%    222.0       1219   ₹270,640    
  ABB         7      CAPITAL_GOODS         2.783    0.49   +62.0%    +36.3%    2,691.6     100    ₹269,158    
  SKFINDIA    8      CAPITAL_GOODS         2.773    0.68   +55.4%    +37.0%    4,275.1     63     ₹269,329    
  ITC         9      FMCG                  2.722    0.73   +54.3%    +21.3%    260.9       1038   ₹270,817    
  VBL         10     FMCG                  2.687    0.63   +83.6%    +27.4%    183.1       1479   ₹270,834    
  BLUEDART    11     LOGISTICS             2.647    0.74   +58.4%    +26.6%    8,621.4     31     ₹267,263    
  AIAENG      13     METALS                2.352    0.50   +25.3%    +30.1%    2,434.5     111    ₹270,230    
  MRF         21     AUTO_ANCILLARY        2.012    0.67   +9.2%     +21.2%    86,534.8    3      ₹259,604    

  HOLDS (7)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TATAELXSI   16     IT                    01-Apr-21   2,645.7     8,473.4     67     ₹390,455      +220.3%   
  ATGL        1      OIL_GAS               01-Sep-21   1,509.2     3,214.4     150    ₹255,780      +113.0%   
  SCHAEFFLER  5      AUTO_ANCILLARY        03-Jan-22   1,757.9     2,742.8     140    ₹137,880      +56.0%    
  LINDEINDIA  18     CHEMICALS             01-Feb-22   2,686.0     3,657.0     91     ₹88,362       +36.2%    
  HAL         6      CAPITAL_GOODS         01-Apr-22   723.8       963.6       362    ₹86,789       +33.1%    
  FLUOROCHEM  17     CHEMICALS             01-Apr-22   2,785.9     3,235.5     94     ₹42,267       +16.1%    
  COALINDIA   25     METALS                01-Apr-22   137.5       157.1       1910   ₹37,528       +14.3%    

  AFTER: Invested ₹5,408,551 | Cash ₹5,068 | Total ₹5,413,619 | Positions 17/20 | Slot ₹270,841

========================================================================
  REBALANCE #58  —  01 Sep 2022
  NAV: ₹5,799,125  |  Slot: ₹289,956  |  Cash: ₹5,068
========================================================================
  [SECTOR CAP≤4] dropped: BEL, GRINDWELL, ZFCVINDIA

  EXITS (4)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TATAELXSI   52     IT                    01-Apr-21   2,645.7     8,764.6     67     ₹409,965      +231.3%   518d  
  LINDEINDIA  60     CHEMICALS             01-Feb-22   2,686.0     3,368.2     91     ₹62,079       +25.4%    212d  
  AIAENG      42     METALS                01-Aug-22   2,434.5     2,582.5     111    ₹16,424       +6.1%     31d   
  MRF         57     AUTO_ANCILLARY        01-Aug-22   86,534.8    84,735.9    3      ₹-5,397       -2.1%     31d   

  ENTRIES (4)
  [52w filter blocked 2: ADANIGREEN(-21.5%), TTML(-56.0%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  FEDERALBNK  10     BANKING               2.551    1.20   +54.6%    +35.3%    116.5       2489   ₹289,880    
  TIINDIA     11     AUTO_ANCILLARY        2.541    0.83   +78.8%    +41.4%    2,251.6     128    ₹288,199    
  SOLARINDS   16     CHEMICALS             2.377    0.56   +96.2%    +26.9%    3,315.3     87     ₹288,432    
  PIDILITIND  17     CHEMICALS             2.316    0.78   +27.1%    +28.4%    1,387.6     208    ₹288,627    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ATGL        1      OIL_GAS               01-Sep-21   1,509.2     3,635.5     150    ₹318,955      +140.9%   
  SCHAEFFLER  3      AUTO_ANCILLARY        03-Jan-22   1,757.9     3,085.3     140    ₹185,839      +75.5%    
  HAL         14     CAPITAL_GOODS         01-Apr-22   723.8       1,098.4     362    ₹135,588      +51.7%    
  COALINDIA   19     METALS                01-Apr-22   137.5       172.5       1910   ₹66,784       +25.4%    
  ABB         4      CAPITAL_GOODS         01-Aug-22   2,691.6     3,336.7     100    ₹64,509       +24.0%    
  FLUOROCHEM  31     CHEMICALS             01-Apr-22   2,785.9     3,363.8     94     ₹54,322       +20.7%    
  VBL         6      FMCG                  01-Aug-22   183.1       206.6       1479   ₹34,715       +12.8%    
  SKFINDIA    7      CAPITAL_GOODS         01-Aug-22   4,275.1     4,707.1     63     ₹27,217       +10.1%    
  TVSMOTOR    2      AUTO                  01-Aug-22   911.5       998.5       297    ₹25,840       +9.5%     
  M&M         9      AUTO                  01-Aug-22   1,206.3     1,278.7     224    ₹16,221       +6.0%     
  ITC         13     FMCG                  01-Aug-22   260.9       269.5       1038   ₹8,895        +3.3%     
  CGPOWER     5      CAPITAL_GOODS         01-Aug-22   222.0       224.0       1219   ₹2,469        +0.9%     
  BLUEDART    30     LOGISTICS             01-Aug-22   8,621.4     8,626.0     31     ₹144          +0.1%     

  AFTER: Invested ₹5,514,601 | Cash ₹283,152 | Total ₹5,797,753 | Positions 17/20 | Slot ₹289,956

========================================================================
  REBALANCE #59  —  03 Oct 2022
  NAV: ₹5,794,904  |  Slot: ₹289,745  |  Cash: ₹283,152
========================================================================
  [SECTOR CAP≤4] dropped: ZFCVINDIA, BEL

  EXITS (1)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  FEDERALBNK  —      BANKING               01-Sep-22   116.5       114.1       2489   ₹-5,981       -2.1%     32d   

  ENTRIES (1)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  BAJAJHLDNG  6      FINANCIAL_SERVICES    2.813    0.83   +38.5%    +44.7%    6,299.2     45     ₹283,464    

  HOLDS (16)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ATGL        11     OIL_GAS               01-Sep-21   1,509.2     3,097.6     150    ₹238,270      +105.3%   
  SCHAEFFLER  1      AUTO_ANCILLARY        03-Jan-22   1,757.9     3,084.4     140    ₹185,710      +75.5%    
  HAL         5      CAPITAL_GOODS         01-Apr-22   723.8       1,092.9     362    ₹133,602      +51.0%    
  FLUOROCHEM  4      CHEMICALS             01-Apr-22   2,785.9     3,968.5     94     ₹111,170      +42.5%    
  TIINDIA     2      AUTO_ANCILLARY        01-Sep-22   2,251.6     2,688.0     128    ₹55,864       +19.4%    
  SOLARINDS   3      CHEMICALS             01-Sep-22   3,315.3     3,959.1     87     ₹56,007       +19.4%    
  COALINDIA   33     METALS                01-Apr-22   137.5       161.0       1910   ₹44,828       +17.1%    
  VBL         7      FMCG                  01-Aug-22   183.1       212.1       1479   ₹42,924       +15.8%    
  ABB         8      CAPITAL_GOODS         01-Aug-22   2,691.6     2,935.8     100    ₹24,424       +9.1%     
  TVSMOTOR    13     AUTO                  01-Aug-22   911.5       979.2       297    ₹20,114       +7.4%     
  SKFINDIA    15     CAPITAL_GOODS         01-Aug-22   4,275.1     4,531.2     63     ₹16,135       +6.0%     
  ITC         19     FMCG                  01-Aug-22   260.9       275.2       1038   ₹14,884       +5.5%     
  BLUEDART    25     LOGISTICS             01-Aug-22   8,621.4     8,901.2     31     ₹8,674        +3.2%     
  CGPOWER     12     CAPITAL_GOODS         01-Aug-22   222.0       226.8       1219   ₹5,840        +2.2%     
  M&M         21     AUTO                  01-Aug-22   1,206.3     1,219.6     224    ₹2,969        +1.1%     
  PIDILITIND  28     CHEMICALS             01-Sep-22   1,387.6     1,306.9     208    ₹-16,789      -5.8%     

  AFTER: Invested ₹5,511,316 | Cash ₹283,251 | Total ₹5,794,567 | Positions 17/20 | Slot ₹289,745

========================================================================
  REBALANCE #60  —  01 Nov 2022
  NAV: ₹5,994,706  |  Slot: ₹299,735  |  Cash: ₹283,251
========================================================================

  EXITS (4)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ABB         25     CAPITAL_GOODS         01-Aug-22   2,691.6     3,090.1     100    ₹39,849       +14.8%    92d   
  SKFINDIA    41     CAPITAL_GOODS         01-Aug-22   4,275.1     4,292.9     63     ₹1,123        +0.4%     92d   
  PIDILITIND  52     CHEMICALS             01-Sep-22   1,387.6     1,300.8     208    ₹-18,068      -6.3%     61d   
  BLUEDART    129    LOGISTICS             01-Aug-22   8,621.4     7,264.5     31     ₹-42,062      -15.7%    92d   

  ENTRIES (4)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  AMBUJACEM   3      CHEMICALS             3.133    0.86   +46.1%    +46.7%    537.3       557    ₹299,278    
  INDHOTEL    9      CONSUMER_DISCRETIONARY  2.622    1.17   +70.0%    +29.7%    338.5       885    ₹299,561    
  EICHERMOT   10     AUTO                  2.580    1.00   +47.6%    +25.2%    3,668.2     81     ₹297,124    
  ZFCVINDIA   11     CAPITAL_GOODS         2.535    0.47   +40.7%    +20.9%    10,220.9    29     ₹296,405    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  ATGL        5      OIL_GAS               01-Sep-21   1,509.2     3,677.4     150    ₹325,241      +143.7%   
  HAL         2      CAPITAL_GOODS         01-Apr-22   723.8       1,207.2     362    ₹174,993      +66.8%    
  SCHAEFFLER  19     AUTO_ANCILLARY        03-Jan-22   1,757.9     2,718.3     140    ₹134,457      +54.6%    
  COALINDIA   21     METALS                01-Apr-22   137.5       184.0       1910   ₹88,740       +33.8%    
  FLUOROCHEM  24     CHEMICALS             01-Apr-22   2,785.9     3,698.0     94     ₹85,741       +32.7%    
  TIINDIA     7      AUTO_ANCILLARY        01-Sep-22   2,251.6     2,785.9     128    ₹68,400       +23.7%    
  TVSMOTOR    1      AUTO                  01-Aug-22   911.5       1,119.4     297    ₹61,744       +22.8%    
  VBL         6      FMCG                  01-Aug-22   183.1       219.4       1479   ₹53,712       +19.8%    
  SOLARINDS   4      CHEMICALS             01-Sep-22   3,315.3     3,942.4     87     ₹54,555       +18.9%    
  CGPOWER     23     CAPITAL_GOODS         01-Aug-22   222.0       258.9       1219   ₹44,976       +16.6%    
  ITC         8      FMCG                  01-Aug-22   260.9       296.5       1038   ₹36,990       +13.7%    
  M&M         17     AUTO                  01-Aug-22   1,206.3     1,319.7     224    ₹25,390       +9.4%     
  BAJAJHLDNG  12     FINANCIAL_SERVICES    03-Oct-22   6,299.2     6,498.9     45     ₹8,985        +3.2%     

  AFTER: Invested ₹5,828,602 | Cash ₹164,687 | Total ₹5,993,290 | Positions 17/20 | Slot ₹299,735

========================================================================
  REBALANCE #61  —  01 Dec 2022
  NAV: ₹5,961,582  |  Slot: ₹298,079  |  Cash: ₹164,687
========================================================================

  EXITS (4)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ATGL        —      OIL_GAS               01-Sep-21   1,509.2     3,608.7     150    ₹314,932      +139.1%   456d  
  FLUOROCHEM  36     CHEMICALS             01-Apr-22   2,785.9     3,474.9     94     ₹64,764       +24.7%    244d  
  ZFCVINDIA   86     CAPITAL_GOODS         01-Nov-22   10,220.9    9,437.4     29     ₹-22,722      -7.7%     30d   
  EICHERMOT   39     AUTO                  01-Nov-22   3,668.2     3,319.6     81     ₹-28,235      -9.5%     30d   

  ENTRIES (5)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  IRFC        1      FINANCIAL_SERVICES    5.377    0.46   +55.2%    +69.4%    32.4        9199   ₹298,068    
  UNIONBANK   2      BANKING               4.582    1.16   +96.9%    +92.4%    74.3        4011   ₹298,010    
  INDIANB     4      BANKING               3.199    1.19   +94.8%    +42.1%    255.3       1167   ₹297,912    
  BANKINDIA   5      FINANCIAL_SERVICES    3.192    1.19   +53.1%    +60.1%    76.3        3905   ₹298,024    
  CUMMINSIND  8      CAPITAL_GOODS         2.897    0.77   +67.1%    +20.6%    1,372.2     217    ₹297,757    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  HAL         3      CAPITAL_GOODS         01-Apr-22   723.8       1,323.9     362    ₹217,246      +82.9%    
  SCHAEFFLER  33     AUTO_ANCILLARY        03-Jan-22   1,757.9     2,684.3     140    ₹129,693      +52.7%    
  VBL         7      FMCG                  01-Aug-22   183.1       250.6       1479   ₹99,800       +36.8%    
  COALINDIA   21     METALS                01-Apr-22   137.5       180.3       1910   ₹81,668       +31.1%    
  TIINDIA     13     AUTO_ANCILLARY        01-Sep-22   2,251.6     2,806.1     128    ₹70,987       +24.6%    
  CGPOWER     9      CAPITAL_GOODS         01-Aug-22   222.0       276.0       1219   ₹65,808       +24.3%    
  SOLARINDS   15     CHEMICALS             01-Sep-22   3,315.3     4,064.7     87     ₹65,198       +22.6%    
  TVSMOTOR    20     AUTO                  01-Aug-22   911.5       1,032.8     297    ₹36,036       +13.3%    
  ITC         10     FMCG                  01-Aug-22   260.9       288.1       1038   ₹28,227       +10.4%    
  AMBUJACEM   6      CHEMICALS             01-Nov-22   537.3       573.7       557    ₹20,298       +6.8%     
  M&M         34     AUTO                  01-Aug-22   1,206.3     1,260.6     224    ₹12,160       +4.5%     
  BAJAJHLDNG  32     FINANCIAL_SERVICES    03-Oct-22   6,299.2     6,113.1     45     ₹-8,377       -3.0%     
  INDHOTEL    14     CONSUMER_DISCRETIONARY  01-Nov-22   338.5       321.2       885    ₹-15,268      -5.1%     

  AFTER: Invested ₹5,876,150 | Cash ₹83,664 | Total ₹5,959,813 | Positions 18/20 | Slot ₹298,079

========================================================================
  REBALANCE #62  —  02 Jan 2023
  NAV: ₹5,905,866  |  Slot: ₹295,293  |  Cash: ₹83,664
========================================================================

  EXITS (6)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SCHAEFFLER  65     AUTO_ANCILLARY        03-Jan-22   1,757.9     2,633.9     140    ₹122,645      +49.8%    364d  
  BANKINDIA   —      FINANCIAL_SERVICES    01-Dec-22   76.3        84.0        3905   ₹30,125       +10.1%    32d   
  INDIANB     —      BANKING               01-Dec-22   255.3       273.5       1167   ₹21,260       +7.1%     32d   
  UNIONBANK   —      BANKING               01-Dec-22   74.3        74.2        4011   ₹-366         -0.1%     32d   
  AMBUJACEM   43     CHEMICALS             01-Nov-22   537.3       520.0       557    ₹-9,654       -3.2%     62d   
  BAJAJHLDNG  121    FINANCIAL_SERVICES    03-Oct-22   6,299.2     5,427.9     45     ₹-39,210      -13.8%    91d   

  ENTRIES (6)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  PFC         2      FINANCIAL_SERVICES    3.883    0.91   +39.1%    +46.6%    102.6       2878   ₹295,243    
  RECLTD      4      FINANCIAL_SERVICES    3.331    0.92   +38.7%    +34.6%    100.3       2944   ₹295,218    
  GICRE       6      FINANCIAL_SERVICES    2.843    1.02   +40.9%    +52.7%    172.6       1711   ₹295,261    
  AXISBANK    8      BANKING               2.785    1.05   +39.6%    +28.4%    939.1       314    ₹294,881    
  YESBANK     9      BANKING               2.665    1.06   +61.0%    +38.3%    21.6        13639  ₹295,284    
  BHARATFORG  14     AUTO_ANCILLARY        2.346    1.07   +27.2%    +27.3%    866.0       340    ₹294,436    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  HAL         5      CAPITAL_GOODS         01-Apr-22   723.8       1,220.7     362    ₹179,867      +68.6%    
  VBL         3      FMCG                  01-Aug-22   183.1       264.3       1479   ₹120,000      +44.3%    
  SOLARINDS   7      CHEMICALS             01-Sep-22   3,315.3     4,419.1     87     ₹96,030       +33.3%    
  COALINDIA   13     METALS                01-Apr-22   137.5       179.2       1910   ₹79,612       +30.3%    
  TIINDIA     39     AUTO_ANCILLARY        01-Sep-22   2,251.6     2,773.5     128    ₹66,806       +23.2%    
  CGPOWER     18     CAPITAL_GOODS         01-Aug-22   222.0       267.3       1219   ₹55,212       +20.4%    
  TVSMOTOR    12     AUTO                  01-Aug-22   911.5       1,055.5     297    ₹42,769       +15.8%    
  ITC         11     FMCG                  01-Aug-22   260.9       282.4       1038   ₹22,326       +8.2%     
  M&M         26     AUTO                  01-Aug-22   1,206.3     1,230.8     224    ₹5,480        +2.0%     
  CUMMINSIND  10     CAPITAL_GOODS         01-Dec-22   1,372.2     1,322.3     217    ₹-10,810      -3.6%     
  IRFC        1      FINANCIAL_SERVICES    01-Dec-22   32.4        31.2        9199   ₹-11,330      -3.8%     
  INDHOTEL    24     CONSUMER_DISCRETIONARY  01-Nov-22   338.5       314.5       885    ₹-21,234      -7.1%     

  AFTER: Invested ₹5,744,931 | Cash ₹158,833 | Total ₹5,903,764 | Positions 18/20 | Slot ₹295,293

========================================================================
  REBALANCE #63  —  01 Feb 2023
  NAV: ₹5,655,774  |  Slot: ₹282,789  |  Cash: ₹158,833
========================================================================

  EXITS (3)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TIINDIA     41     AUTO_ANCILLARY        01-Sep-22   2,251.6     2,605.8     128    ₹45,343       +15.7%    153d  
  BHARATFORG  42     AUTO_ANCILLARY        02-Jan-23   866.0       850.4       340    ₹-5,296       -1.8%     30d   
  AXISBANK    66     BANKING               02-Jan-23   939.1       855.0       314    ₹-26,416      -9.0%     30d   

  ENTRIES (3)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  BRITANNIA   5      FMCG                  2.849    0.55   +27.3%    +17.4%    4,187.2     67     ₹280,541    
  HINDZINC    9      CHEMICALS             2.730    0.68   +24.8%    +25.9%    268.2       1054   ₹282,703    
  AIAENG      12     METALS                2.270    0.51   +53.1%    +2.8%     2,815.6     100    ₹281,563    

  HOLDS (15)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  HAL         14     CAPITAL_GOODS         01-Apr-22   723.8       1,136.0     362    ₹149,228      +57.0%    
  CGPOWER     4      CAPITAL_GOODS         01-Aug-22   222.0       298.7       1219   ₹93,505       +34.5%    
  COALINDIA   27     METALS                01-Apr-22   137.5       175.9       1910   ₹73,369       +27.9%    
  VBL         8      FMCG                  01-Aug-22   183.1       232.5       1479   ₹73,018       +27.0%    
  SOLARINDS   11     CHEMICALS             01-Sep-22   3,315.3     3,931.0     87     ₹53,566       +18.6%    
  ITC         2      FMCG                  01-Aug-22   260.9       306.6       1038   ₹47,470       +17.5%    
  TVSMOTOR    22     AUTO                  01-Aug-22   911.5       1,001.7     297    ₹26,789       +9.9%     
  M&M         10     AUTO                  01-Aug-22   1,206.3     1,317.7     224    ₹24,953       +9.2%     
  CUMMINSIND  6      CAPITAL_GOODS         01-Dec-22   1,372.2     1,361.3     217    ₹-2,350       -0.8%     
  RECLTD      3      FINANCIAL_SERVICES    02-Jan-23   100.3       97.8        2944   ₹-7,233       -2.5%     
  INDHOTEL    19     CONSUMER_DISCRETIONARY  01-Nov-22   338.5       323.9       885    ₹-12,898      -4.3%     
  IRFC        1      FINANCIAL_SERVICES    01-Dec-22   32.4        29.8        9199   ₹-23,532      -7.9%     
  PFC         7      FINANCIAL_SERVICES    02-Jan-23   102.6       93.4        2878   ₹-26,329      -8.9%     
  GICRE       13     FINANCIAL_SERVICES    02-Jan-23   172.6       148.2       1711   ₹-41,664      -14.1%    
  YESBANK     36     BANKING               02-Jan-23   21.6        16.9        13639  ₹-64,785      -21.9%    

  AFTER: Invested ₹5,450,602 | Cash ₹204,169 | Total ₹5,654,771 | Positions 18/20 | Slot ₹282,789

========================================================================
  REBALANCE #64  —  01 Mar 2023
  NAV: ₹5,771,881  |  Slot: ₹288,594  |  Cash: ₹204,169
========================================================================
  [SECTOR CAP≤4] dropped: ABB, ZFCVINDIA

  [REGIME OFF] Nifty 200 14,664.8 < SMA200 14,849.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  HAL         8      CAPITAL_GOODS         01-Apr-22   723.8       1,289.7     362    ₹204,870      +78.2%    
  VBL         4      FMCG                  01-Aug-22   183.1       266.4       1479   ₹123,166      +45.5%    
  CGPOWER     5      CAPITAL_GOODS         01-Aug-22   222.0       303.1       1219   ₹98,803       +36.5%    
  COALINDIA   26     METALS                01-Apr-22   137.5       178.7       1910   ₹78,630       +29.9%    
  ITC         1      FMCG                  01-Aug-22   260.9       326.6       1038   ₹68,172       +25.2%    
  SOLARINDS   25     CHEMICALS             01-Sep-22   3,315.3     3,893.0     87     ₹50,263       +17.4%    
  TVSMOTOR    14     AUTO                  01-Aug-22   911.5       1,050.7     297    ₹41,351       +15.3%    
  CUMMINSIND  2      CAPITAL_GOODS         01-Dec-22   1,372.2     1,540.7     217    ₹36,582       +12.3%    
  M&M         21     AUTO                  01-Aug-22   1,206.3     1,239.8     224    ₹7,510        +2.8%     
  PFC         7      FINANCIAL_SERVICES    02-Jan-23   102.6       104.9       2878   ₹6,669        +2.3%     
  BRITANNIA   39     FMCG                  01-Feb-23   4,187.2     4,195.7     67     ₹568          +0.2%     
  RECLTD      13     FINANCIAL_SERVICES    02-Jan-23   100.3       98.5        2944   ₹-5,322       -1.8%     
  AIAENG      22     METALS                01-Feb-23   2,815.6     2,672.0     100    ₹-14,358      -5.1%     
  HINDZINC    40     CHEMICALS             01-Feb-23   268.2       251.4       1054   ₹-17,733      -6.3%     
  INDHOTEL    29     CONSUMER_DISCRETIONARY  01-Nov-22   338.5       312.3       885    ₹-23,165      -7.7%     
  YESBANK     32     BANKING               02-Jan-23   21.6        18.3        13639  ₹-45,691      -15.5%    
  IRFC        108    FINANCIAL_SERVICES    01-Dec-22   32.4        25.9        9199   ₹-60,137      -20.2% ⚠  
  GICRE       52     FINANCIAL_SERVICES    02-Jan-23   172.6       131.7       1711   ₹-69,963      -23.7%    
  ⚠  WAZ < 0 (momentum below universe mean): IRFC

  AFTER: Invested ₹5,567,712 | Cash ₹204,169 | Total ₹5,771,881 | Positions 18/20 | Slot ₹288,594

========================================================================
  REBALANCE #65  —  03 Apr 2023
  NAV: ₹5,769,037  |  Slot: ₹288,452  |  Cash: ₹204,169
========================================================================
  [SECTOR CAP≤4] dropped: SIEMENS, ZFCVINDIA, GODREJCP

  [REGIME OFF] Nifty 200 14,602.0 < SMA200 14,914.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  HAL         6      CAPITAL_GOODS         01-Apr-22   723.8       1,312.6     362    ₹213,162      +81.4%    
  VBL         5      FMCG                  01-Aug-22   183.1       280.4       1479   ₹143,850      +53.1%    
  CGPOWER     15     CAPITAL_GOODS         01-Aug-22   222.0       294.6       1219   ₹88,479       +32.7%    
  COALINDIA   47     METALS                01-Apr-22   137.5       179.8       1910   ₹80,814       +30.8%    
  ITC         1      FMCG                  01-Aug-22   260.9       326.8       1038   ₹68,351       +25.2%    
  TVSMOTOR    13     AUTO                  01-Aug-22   911.5       1,072.3     297    ₹47,751       +17.6%    
  CUMMINSIND  7      CAPITAL_GOODS         01-Dec-22   1,372.2     1,543.4     217    ₹37,156       +12.5%    
  SOLARINDS   94     CHEMICALS             01-Sep-22   3,315.3     3,724.5     87     ₹35,601       +12.3%    
  PFC         11     FINANCIAL_SERVICES    02-Jan-23   102.6       108.8       2878   ₹17,981       +6.1%     
  AIAENG      3      METALS                01-Feb-23   2,815.6     2,916.9     100    ₹10,123       +3.6%     
  RECLTD      21     FINANCIAL_SERVICES    02-Jan-23   100.3       100.2       2944   ₹-157         -0.1%     
  BRITANNIA   23     FMCG                  01-Feb-23   4,187.2     4,142.3     67     ₹-3,010       -1.1%     
  HINDZINC    43     CHEMICALS             01-Feb-23   268.2       261.3       1054   ₹-7,310       -2.6%     
  M&M         37     AUTO                  01-Aug-22   1,206.3     1,140.2     224    ₹-14,812      -5.5%     
  INDHOTEL    46     CONSUMER_DISCRETIONARY  01-Nov-22   338.5       318.3       885    ₹-17,856      -6.0%     
  IRFC        91     FINANCIAL_SERVICES    01-Dec-22   32.4        26.3        9199   ₹-55,779      -18.7%    
  GICRE       120    FINANCIAL_SERVICES    02-Jan-23   172.6       125.2       1711   ₹-81,047      -27.4% ⚠  
  YESBANK     124    BANKING               02-Jan-23   21.6        15.4        13639  ₹-85,926      -29.1% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): GICRE, YESBANK

  AFTER: Invested ₹5,564,867 | Cash ₹204,169 | Total ₹5,769,037 | Positions 18/20 | Slot ₹288,452

========================================================================
  REBALANCE #66  —  02 May 2023
  NAV: ₹6,144,998  |  Slot: ₹307,250  |  Cash: ₹204,169
========================================================================
  [SECTOR CAP≤4] dropped: ABB

  EXITS (6)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SOLARINDS   87     CHEMICALS             01-Sep-22   3,315.3     3,825.9     87     ₹44,419       +15.4%    243d  
  HINDZINC    78     CHEMICALS             01-Feb-23   268.2       276.2       1054   ₹8,390        +3.0%     90d   
  M&M         81     AUTO                  01-Aug-22   1,206.3     1,205.9     224    ₹-98          -0.0%     274d  
  IRFC        39     FINANCIAL_SERVICES    01-Dec-22   32.4        31.8        9199   ₹-5,665       -1.9%     152d  
  GICRE       118    FINANCIAL_SERVICES    02-Jan-23   172.6       143.4       1711   ₹-49,918      -16.9%    120d  
  YESBANK     133    BANKING               02-Jan-23   21.6        15.9        13639  ₹-77,742      -26.3%    120d  

  ENTRIES (6)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  ZYDUSLIFE   3      PHARMA                2.840    0.59   +52.6%    +18.3%    505.1       608    ₹307,109    
  SIEMENS     6      CAPITAL_GOODS         2.754    0.81   +55.7%    +15.9%    2,039.2     150    ₹305,887    
  INDIANB     8      BANKING               2.709    1.18   +115.1%   +13.4%    304.4       1009   ₹307,124    
  BAJAJ-AUTO  10     AUTO                  2.632    0.65   +26.7%    +22.0%    4,222.3     72     ₹304,005    
  BOSCHLTD    11     AUTO_ANCILLARY        2.617    0.99   +43.2%    +17.4%    19,016.2    16     ₹304,259    
  APOLLOTYRE  13     AUTO_ANCILLARY        2.427    1.03   +83.6%    +7.9%     332.7       923    ₹307,118    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  HAL         9      CAPITAL_GOODS         01-Apr-22   723.8       1,422.0     362    ₹252,737      +96.5%    
  VBL         15     FMCG                  01-Aug-22   183.1       280.7       1479   ₹144,329      +53.3%    
  ITC         1      FMCG                  01-Aug-22   260.9       366.0       1038   ₹109,125      +40.3%    
  COALINDIA   34     METALS                01-Apr-22   137.5       192.8       1910   ₹105,697      +40.2%    
  CGPOWER     31     CAPITAL_GOODS         01-Aug-22   222.0       304.7       1219   ₹100,765      +37.2%    
  TVSMOTOR    2      AUTO                  01-Aug-22   911.5       1,144.6     297    ₹69,241       +25.6%    
  PFC         5      FINANCIAL_SERVICES    02-Jan-23   102.6       121.9       2878   ₹55,455       +18.8%    
  RECLTD      7      FINANCIAL_SERVICES    02-Jan-23   100.3       116.8       2944   ₹48,601       +16.5%    
  CUMMINSIND  4      CAPITAL_GOODS         01-Dec-22   1,372.2     1,538.5     217    ₹36,102       +12.1%    
  BRITANNIA   27     FMCG                  01-Feb-23   4,187.2     4,395.5     67     ₹13,956       +5.0%     
  INDHOTEL    19     CONSUMER_DISCRETIONARY  01-Nov-22   338.5       344.5       885    ₹5,309        +1.8%     
  AIAENG      23     METALS                01-Feb-23   2,815.6     2,741.9     100    ₹-7,369       -2.6%     

  AFTER: Invested ₹6,126,980 | Cash ₹15,838 | Total ₹6,142,818 | Positions 18/20 | Slot ₹307,250

========================================================================
  REBALANCE #67  —  01 Jun 2023
  NAV: ₹6,512,122  |  Slot: ₹325,606  |  Cash: ₹15,838
========================================================================
  [SECTOR CAP≤4] dropped: 3MINDIA

  EXITS (7)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COALINDIA   66     METALS                01-Apr-22   137.5       188.1       1910   ₹96,727       +36.8%    426d  
  BRITANNIA   59     FMCG                  01-Feb-23   4,187.2     4,516.5     67     ₹22,064       +7.9%     120d  
  AIAENG      52     METALS                01-Feb-23   2,815.6     2,991.1     100    ₹17,547       +6.2%     120d  
  SIEMENS     43     CAPITAL_GOODS         02-May-23   2,039.2     2,059.2     150    ₹2,989        +1.0%     30d   
  ZYDUSLIFE   45     PHARMA                02-May-23   505.1       501.6       608    ₹-2,109       -0.7%     30d   
  BOSCHLTD    74     AUTO_ANCILLARY        02-May-23   19,016.2    17,867.1    16     ₹-18,385      -6.0%     30d   
  INDIANB     81     BANKING               02-May-23   304.4       254.2       1009   ₹-50,620      -16.5%    30d   

  ENTRIES (6)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  CHOLAFIN    7      FINANCIAL_SERVICES    2.798    1.11   +63.8%    +36.8%    1,039.4     313    ₹325,320    
  ABB         8      CAPITAL_GOODS         2.610    0.79   +80.3%    +25.7%    3,873.2     84     ₹325,352    
  SYNGENE     10     HEALTHCARE            2.480    0.43   +37.7%    +28.3%    722.7       450    ₹325,206    
  OFSS        12     IT                    2.332    0.80   +24.3%    +22.3%    3,252.2     100    ₹325,219    
  AUROPHARMA  13     PHARMA                2.305    0.72   +24.8%    +40.4%    653.3       498    ₹325,363    
  TORNTPHARM  14     PHARMA                2.273    0.39   +24.5%    +19.9%    1,720.2     189    ₹325,126    

  HOLDS (11)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  HAL         15     CAPITAL_GOODS         01-Apr-22   723.8       1,490.9     362    ₹277,692      +106.0%   
  VBL         4      FMCG                  01-Aug-22   183.1       335.0       1479   ₹224,686      +83.0%    
  CGPOWER     1      CAPITAL_GOODS         01-Aug-22   222.0       385.8       1219   ₹199,652      +73.8%    
  ITC         2      FMCG                  01-Aug-22   260.9       387.7       1038   ₹131,616      +48.6%    
  TVSMOTOR    11     AUTO                  01-Aug-22   911.5       1,257.5     297    ₹102,754      +38.0%    
  PFC         6      FINANCIAL_SERVICES    02-Jan-23   102.6       128.6       2878   ₹74,849       +25.4%    
  CUMMINSIND  19     CAPITAL_GOODS         01-Dec-22   1,372.2     1,683.2     217    ₹67,502       +22.7%    
  RECLTD      3      FINANCIAL_SERVICES    02-Jan-23   100.3       120.3       2944   ₹59,058       +20.0%    
  INDHOTEL    5      CONSUMER_DISCRETIONARY  01-Nov-22   338.5       392.3       885    ₹47,645       +15.9%    
  APOLLOTYRE  9      AUTO_ANCILLARY        02-May-23   332.7       375.8       923    ₹39,743       +12.9%    
  BAJAJ-AUTO  23     AUTO                  02-May-23   4,222.3     4,359.0     72     ₹9,841        +3.2%     

  AFTER: Invested ₹6,330,544 | Cash ₹179,260 | Total ₹6,509,805 | Positions 17/20 | Slot ₹325,606

========================================================================
  REBALANCE #68  —  03 Jul 2023
  NAV: ₹6,985,440  |  Slot: ₹349,272  |  Cash: ₹179,260
========================================================================
  [SECTOR CAP≤4] dropped: INDIGO, ZFCVINDIA, SUPREMEIND, BEL

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VBL         49     FMCG                  01-Aug-22   183.1       324.7       1479   ₹209,342      +77.3%    336d  
  PFC         —      FINANCIAL_SERVICES    02-Jan-23   102.6       157.8       2878   ₹158,857      +53.8%    182d  
  INDHOTEL    28     CONSUMER_DISCRETIONARY  01-Nov-22   338.5       388.1       885    ₹43,897       +14.7%    244d  
  AUROPHARMA  43     PHARMA                01-Jun-23   653.3       706.6       498    ₹26,525       +8.2%     32d   
  BAJAJ-AUTO  35     AUTO                  02-May-23   4,222.3     4,460.4     72     ₹17,147       +5.6%     62d   

  ENTRIES (6)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  ASTRAL      8      INFRASTRUCTURE        2.468    0.81   +57.3%    +51.0%    1,951.5     178    ₹347,366    
  SUNDRMFAST  10     AUTO_ANCILLARY        2.342    0.37   +70.8%    +24.7%    1,183.8     295    ₹349,221    
  BSE         12     FINANCIAL_SERVICES    2.265    0.93   +12.6%    +64.1%    218.0       1601   ₹349,090    
  TRENT       13     CONSUMER_DISCRETIONARY  2.238    1.12   +65.7%    +35.9%    1,748.8     199    ₹348,018    
  TATACOMM    16     TELECOM               2.133    0.94   +74.9%    +32.6%    1,552.7     224    ₹347,805    
  IOC         18     OIL_GAS               2.065    0.55   +36.2%    +23.9%    79.1        4415   ₹349,239    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  HAL         4      CAPITAL_GOODS         01-Apr-22   723.8       1,824.5     362    ₹398,449      +152.1%   
  CGPOWER     7      CAPITAL_GOODS         01-Aug-22   222.0       376.2       1219   ₹187,972      +69.5%    
  ITC         3      FMCG                  01-Aug-22   260.9       408.5       1038   ₹153,170      +56.6%    
  TVSMOTOR    25     AUTO                  01-Aug-22   911.5       1,303.2     297    ₹116,332      +43.0%    
  RECLTD      1      FINANCIAL_SERVICES    02-Jan-23   100.3       142.3       2944   ₹123,689      +41.9%    
  CUMMINSIND  17     CAPITAL_GOODS         01-Dec-22   1,372.2     1,817.1     217    ₹96,555       +32.4%    
  APOLLOTYRE  6      AUTO_ANCILLARY        02-May-23   332.7       381.6       923    ₹45,065       +14.7%    
  CHOLAFIN    2      FINANCIAL_SERVICES    01-Jun-23   1,039.4     1,164.1     313    ₹39,033       +12.0%    
  ABB         9      CAPITAL_GOODS         01-Jun-23   3,873.2     4,331.4     84     ₹38,485       +11.8%    
  TORNTPHARM  24     PHARMA                01-Jun-23   1,720.2     1,840.7     189    ₹22,763       +7.0%     
  OFSS        21     IT                    01-Jun-23   3,252.2     3,432.6     100    ₹18,039       +5.5%     
  SYNGENE     26     HEALTHCARE            01-Jun-23   722.7       756.8       450    ₹15,354       +4.7%     

  AFTER: Invested ₹6,946,145 | Cash ₹36,812 | Total ₹6,982,957 | Positions 18/20 | Slot ₹349,272

========================================================================
  REBALANCE #69  —  01 Aug 2023
  NAV: ₹7,375,827  |  Slot: ₹368,791  |  Cash: ₹36,812
========================================================================
  [SECTOR CAP≤4] dropped: ZFCVINDIA, SUPREMEIND, HDFCAMC

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TVSMOTOR    38     AUTO                  01-Aug-22   911.5       1,353.9     297    ₹131,407      +48.5%    365d  
  SYNGENE     44     HEALTHCARE            01-Jun-23   722.7       803.2       450    ₹36,224       +11.1%    61d   
  OFSS        69     IT                    01-Jun-23   3,252.2     3,480.1     100    ₹22,791       +7.0%     61d   
  IOC         25     OIL_GAS               03-Jul-23   79.1        80.5        4415   ₹6,232        +1.8%     29d   
  TRENT       50     CONSUMER_DISCRETIONARY  03-Jul-23   1,748.8     1,705.0     199    ₹-8,733       -2.5%     29d   

  ENTRIES (4)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  PFC         1      FINANCIAL_SERVICES    4.311    1.20   +150.3%   +55.3%    185.9       1984   ₹368,741    
  ZYDUSLIFE   5      PHARMA                2.742    0.42   +85.5%    +22.9%    625.2       589    ₹368,220    
  NTPC        6      ENERGY                2.721    0.70   +56.9%    +27.7%    207.6       1776   ₹368,739    
  LUPIN       8      PHARMA                2.594    0.61   +56.9%    +39.9%    977.8       377    ₹368,632    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  HAL         10     CAPITAL_GOODS         01-Apr-22   723.8       1,870.9     362    ₹415,241      +158.5%   
  CGPOWER     7      CAPITAL_GOODS         01-Aug-22   222.0       405.1       1219   ₹223,194      +82.5%    
  RECLTD      2      FINANCIAL_SERVICES    02-Jan-23   100.3       175.8       2944   ₹222,350      +75.3%    
  ITC         17     FMCG                  01-Aug-22   260.9       409.8       1038   ₹154,589      +57.1%    
  CUMMINSIND  16     CAPITAL_GOODS         01-Dec-22   1,372.2     1,865.6     217    ₹107,074      +36.0%    
  BSE         9      FINANCIAL_SERVICES    03-Jul-23   218.0       274.3       1601   ₹90,010       +25.8%    
  APOLLOTYRE  13     AUTO_ANCILLARY        02-May-23   332.7       415.0       923    ₹75,961       +24.7%    
  ABB         15     CAPITAL_GOODS         01-Jun-23   3,873.2     4,437.8     84     ₹47,426       +14.6%    
  TORNTPHARM  19     PHARMA                01-Jun-23   1,720.2     1,930.8     189    ₹39,799       +12.2%    
  TATACOMM    4      TELECOM               03-Jul-23   1,552.7     1,721.2     224    ₹37,740       +10.9%    
  CHOLAFIN    18     FINANCIAL_SERVICES    01-Jun-23   1,039.4     1,126.2     313    ₹27,188       +8.4%     
  SUNDRMFAST  20     AUTO_ANCILLARY        03-Jul-23   1,183.8     1,227.6     295    ₹12,934       +3.7%     
  ASTRAL      23     INFRASTRUCTURE        03-Jul-23   1,951.5     1,987.2     178    ₹6,347        +1.8%     

  AFTER: Invested ₹7,007,030 | Cash ₹367,046 | Total ₹7,374,076 | Positions 17/20 | Slot ₹368,791

========================================================================
  REBALANCE #70  —  01 Sep 2023
  NAV: ₹7,563,187  |  Slot: ₹378,159  |  Cash: ₹367,046
========================================================================
  [SECTOR CAP≤4] dropped: ZFCVINDIA, LT

  EXITS (8)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ITC         55     FMCG                  01-Aug-22   260.9       388.9       1038   ₹132,852      +49.1%    396d  
  CUMMINSIND  97     CAPITAL_GOODS         01-Dec-22   1,372.2     1,651.4     217    ₹60,599       +20.4%    274d  
  APOLLOTYRE  80     AUTO_ANCILLARY        02-May-23   332.7       373.8       923    ₹37,855       +12.3%    122d  
  ABB         87     CAPITAL_GOODS         01-Jun-23   3,873.2     4,219.1     84     ₹29,053       +8.9%     92d   
  CHOLAFIN    68     FINANCIAL_SERVICES    01-Jun-23   1,039.4     1,124.6     313    ₹26,689       +8.2%     92d   
  TORNTPHARM  124    PHARMA                01-Jun-23   1,720.2     1,738.4     189    ₹3,426        +1.1%     92d   
  PFC         —      FINANCIAL_SERVICES    01-Aug-23   185.9       184.8       1984   ₹-2,066       -0.6%     31d   
  ASTRAL      131    INFRASTRUCTURE        03-Jul-23   1,951.5     1,900.4     178    ₹-9,096       -2.6%     60d   

  ENTRIES (8)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  SUPREMEIND  3      CAPITAL_GOODS         3.893    0.48   +130.5%   +56.9%    4,325.9     87     ₹376,352    
  IRFC        4      FINANCIAL_SERVICES    3.679    0.87   +178.0%   +74.2%    52.8        7159   ₹378,134    
  LINDEINDIA  5      CHEMICALS             3.576    0.78   +97.7%    +63.8%    6,527.5     57     ₹372,066    
  BHEL        6      CAPITAL_GOODS         3.358    1.11   +137.8%   +64.3%    135.8       2785   ₹378,065    
  GMRINFRA    8      INFRASTRUCTURE        2.825    1.01   +78.8%    +52.4%    63.0        6007   ₹378,141    
  APLAPOLLO   9      METALS                2.804    0.43   +78.3%    +52.8%    1,716.9     220    ₹377,716    
  ESCORTS     10     AUTO                  2.769    0.62   +71.1%    +44.0%    3,077.7     122    ₹375,477    
  BHARATFORG  13     AUTO_ANCILLARY        2.489    0.92   +48.4%    +37.6%    1,063.6     355    ₹377,577    

  HOLDS (9)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  HAL         18     CAPITAL_GOODS         01-Apr-22   723.8       1,914.7     362    ₹431,108      +164.5%   
  RECLTD      1      FINANCIAL_SERVICES    02-Jan-23   100.3       212.8       2944   ₹331,136      +112.2%   
  CGPOWER     20     CAPITAL_GOODS         01-Aug-22   222.0       421.0       1219   ₹242,500      +89.6%    
  BSE         2      FINANCIAL_SERVICES    03-Jul-23   218.0       372.2       1601   ₹246,725      +70.7%    
  TATACOMM    14     TELECOM               03-Jul-23   1,552.7     1,808.7     224    ₹57,348       +16.5%    
  LUPIN       12     PHARMA                01-Aug-23   977.8       1,082.3     377    ₹39,411       +10.7%    
  SUNDRMFAST  28     AUTO_ANCILLARY        03-Jul-23   1,183.8     1,247.0     295    ₹18,635       +5.3%     
  NTPC        7      ENERGY                01-Aug-23   207.6       215.7       1776   ₹14,376       +3.9%     
  ZYDUSLIFE   16     PHARMA                01-Aug-23   625.2       605.5       589    ₹-11,589      -3.1%     

  AFTER: Invested ₹7,362,761 | Cash ₹196,847 | Total ₹7,559,609 | Positions 17/20 | Slot ₹378,159

========================================================================
  REBALANCE #71  —  03 Oct 2023
  NAV: ₹7,897,183  |  Slot: ₹394,859  |  Cash: ₹196,847
========================================================================

  EXITS (4)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HAL         63     CAPITAL_GOODS         01-Apr-22   723.8       1,901.3     362    ₹426,246      +162.7%   550d  
  IRFC        —      FINANCIAL_SERVICES    01-Sep-23   52.8        73.2        7159   ₹145,934      +38.6%    32d   
  SUNDRMFAST  53     AUTO_ANCILLARY        03-Jul-23   1,183.8     1,260.1     295    ₹22,504       +6.4%     92d   
  BHEL        —      CAPITAL_GOODS         01-Sep-23   135.8       130.8       2785   ₹-13,884      -3.7%     32d   

  ENTRIES (5)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  JSL         2      METALS                4.204    0.73   +284.1%   +46.1%    490.4       805    ₹394,805    
  RVNL        4      INFRASTRUCTURE        3.647    1.14   +426.1%   +41.9%    170.4       2317   ₹394,823    
  LT          5      CAPITAL_GOODS         3.006    0.83   +67.9%    +26.7%    2,991.9     131    ₹391,936    
  INDIANB     6      BANKING               2.857    1.10   +144.4%   +48.2%    420.8       938    ₹394,692    
  NMDC        9      METALS                2.569    1.04   +65.9%    +41.0%    44.2        8926   ₹394,821    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  RECLTD      1      FINANCIAL_SERVICES    02-Jan-23   100.3       260.0       2944   ₹470,151      +159.3%   
  CGPOWER     24     CAPITAL_GOODS         01-Aug-22   222.0       441.2       1219   ₹267,192      +98.7%    
  BSE         3      FINANCIAL_SERVICES    03-Jul-23   218.0       428.9       1601   ₹337,566      +96.7%    
  TATACOMM    26     TELECOM               03-Jul-23   1,552.7     1,856.9     224    ₹68,142       +19.6%    
  LUPIN       8      PHARMA                01-Aug-23   977.8       1,161.8     377    ₹69,384       +18.8%    
  NTPC        7      ENERGY                01-Aug-23   207.6       225.5       1776   ₹31,813       +8.6%     
  BHARATFORG  14     AUTO_ANCILLARY        01-Sep-23   1,063.6     1,068.4     355    ₹1,715        +0.5%     
  ESCORTS     11     AUTO                  01-Sep-23   3,077.7     3,068.6     122    ₹-1,108       -0.3%     
  ZYDUSLIFE   28     PHARMA                01-Aug-23   625.2       602.6       589    ₹-13,273      -3.6%     
  APLAPOLLO   35     METALS                01-Sep-23   1,716.9     1,628.4     220    ₹-19,470      -5.2%     
  GMRINFRA    17     INFRASTRUCTURE        01-Sep-23   63.0        59.5        6007   ₹-21,024      -5.6%     
  LINDEINDIA  10     CHEMICALS             01-Sep-23   6,527.5     5,964.1     57     ₹-32,114      -8.6%     
  SUPREMEIND  15     CAPITAL_GOODS         01-Sep-23   4,325.9     3,951.8     87     ₹-32,547      -8.6%     

  AFTER: Invested ₹7,723,175 | Cash ₹171,667 | Total ₹7,894,843 | Positions 18/20 | Slot ₹394,859

========================================================================
  REBALANCE #72  —  01 Nov 2023
  NAV: ₹7,814,377  |  Slot: ₹390,719  |  Cash: ₹171,667
========================================================================

  EXITS (7)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  RECLTD      —      FINANCIAL_SERVICES    02-Jan-23   100.3       252.0       2944   ₹446,676      +151.3%   303d  
  CGPOWER     63     CAPITAL_GOODS         01-Aug-22   222.0       380.3       1219   ₹192,995      +71.3%    457d  
  TATACOMM    —      TELECOM               03-Jul-23   1,552.7     1,632.2     224    ₹17,815       +5.1%     121d  
  ZYDUSLIFE   79     PHARMA                01-Aug-23   625.2       567.5       589    ₹-33,981      -9.2%     92d   
  RVNL        —      INFRASTRUCTURE        03-Oct-23   170.4       151.1       2317   ₹-44,632      -11.3%    29d   
  APLAPOLLO   76     METALS                01-Sep-23   1,716.9     1,515.5     220    ₹-44,313      -11.7%    61d   
  GMRINFRA    48     INFRASTRUCTURE        01-Sep-23   63.0        54.2        6007   ₹-52,261      -13.8%    61d   

  ENTRIES (7)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  COALINDIA   3      METALS                3.197    0.81   +41.8%    +36.1%    254.9       1533   ₹390,688    
  SOLARINDS   4      CHEMICALS             3.152    0.21   +42.4%    +45.3%    5,514.0     70     ₹385,977    
  PERSISTENT  7      IT                    2.714    1.20   +64.0%    +30.9%    3,057.8     127    ₹388,342    
  TRENT       8      CONSUMER_DISCRETIONARY  2.578    0.95   +53.5%    +25.1%    2,194.2     178    ₹390,575    
  SUNDARMFIN  10     FINANCIAL_SERVICES    2.524    0.41   +37.8%    +20.2%    3,127.1     124    ₹387,763    
  PRESTIGE    12     REAL_ESTATE           2.425    0.50   +70.3%    +27.7%    748.3       522    ₹390,597    
  ZFCVINDIA   13     CAPITAL_GOODS         2.389    0.49   +53.8%    +19.3%    15,413.8    25     ₹385,346    

  HOLDS (11)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BSE         1      FINANCIAL_SERVICES    03-Jul-23   218.0       593.1       1601   ₹600,383      +172.0%   
  LUPIN       11     PHARMA                01-Aug-23   977.8       1,120.8     377    ₹53,922       +14.6%    
  NTPC        24     ENERGY                01-Aug-23   207.6       217.5       1776   ₹17,532       +4.8%     
  NMDC        6      METALS                03-Oct-23   44.2        45.9        8926   ₹14,934       +3.8%     
  SUPREMEIND  5      CAPITAL_GOODS         01-Sep-23   4,325.9     4,404.9     87     ₹6,875        +1.8%     
  INDIANB     15     BANKING               03-Oct-23   420.8       411.0       938    ₹-9,212       -2.3%     
  ESCORTS     18     AUTO                  01-Sep-23   3,077.7     3,003.4     122    ₹-9,058       -2.4%     
  BHARATFORG  35     AUTO_ANCILLARY        01-Sep-23   1,063.6     1,012.8     355    ₹-18,042      -4.8%     
  LT          14     CAPITAL_GOODS         03-Oct-23   2,991.9     2,818.6     131    ₹-22,694      -5.8%     
  LINDEINDIA  9      CHEMICALS             01-Sep-23   6,527.5     5,936.2     57     ₹-33,699      -9.1%     
  JSL         2      METALS                03-Oct-23   490.4       444.1       805    ₹-37,280      -9.4%     

  AFTER: Invested ₹7,447,137 | Cash ₹364,011 | Total ₹7,811,148 | Positions 18/20 | Slot ₹390,719

========================================================================
  REBALANCE #73  —  01 Dec 2023
  NAV: ₹8,941,929  |  Slot: ₹447,096  |  Cash: ₹364,011
========================================================================

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BHARATFORG  73     AUTO_ANCILLARY        01-Sep-23   1,063.6     1,140.2     355    ₹27,195       +7.2%     91d   
  ESCORTS     82     AUTO                  01-Sep-23   3,077.7     3,147.7     122    ₹8,543        +2.3%     91d   
  SUPREMEIND  53     CAPITAL_GOODS         01-Sep-23   4,325.9     4,397.9     87     ₹6,265        +1.7%     91d   
  INDIANB     106    BANKING               03-Oct-23   420.8       382.5       938    ₹-35,910      -9.1%     59d   
  LINDEINDIA  58     CHEMICALS             01-Sep-23   6,527.5     5,929.7     57     ₹-34,072      -9.2%     91d   

  ENTRIES (5)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  TVSMOTOR    4      AUTO                  3.051    0.63   +82.3%    +38.3%    1,887.8     236    ₹445,527    
  AUROPHARMA  6      PHARMA                2.912    0.39   +128.7%   +25.1%    1,028.4     434    ₹446,326    
  BAJAJ-AUTO  7      AUTO                  2.818    0.63   +72.1%    +29.6%    5,849.1     76     ₹444,529    
  IOC         10     OIL_GAS               2.631    0.73   +67.4%    +27.2%    100.2       4464   ₹447,087    
  ALKEM       15     PHARMA                2.224    0.55   +52.1%    +25.5%    4,533.0     98     ₹444,232    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BSE         1      FINANCIAL_SERVICES    03-Jul-23   218.0       825.7       1601   ₹972,877      +278.7%   
  PRESTIGE    2      REAL_ESTATE           01-Nov-23   748.3       1,035.6     522    ₹149,993      +38.4%    
  LUPIN       17     PHARMA                01-Aug-23   977.8       1,283.0     377    ₹115,062      +31.2%    
  TRENT       5      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     2,803.3     178    ₹108,421      +27.8%    
  NMDC        9      METALS                03-Oct-23   44.2        54.4        8926   ₹90,672       +23.0%    
  NTPC        11     ENERGY                01-Aug-23   207.6       253.9       1776   ₹82,210       +22.3%    
  COALINDIA   3      METALS                01-Nov-23   254.9       301.3       1533   ₹71,218       +18.2%    
  SOLARINDS   31     CHEMICALS             01-Nov-23   5,514.0     6,180.5     70     ₹46,655       +12.1%    
  SUNDARMFIN  13     FINANCIAL_SERVICES    01-Nov-23   3,127.1     3,297.0     124    ₹21,067       +5.4%     
  JSL         8      METALS                03-Oct-23   490.4       512.9       805    ₹18,096       +4.6%     
  ZFCVINDIA   16     CAPITAL_GOODS         01-Nov-23   15,413.8    16,105.0    25     ₹17,278       +4.5%     
  LT          14     CAPITAL_GOODS         03-Oct-23   2,991.9     3,106.2     131    ₹14,972       +3.8%     
  PERSISTENT  23     IT                    01-Nov-23   3,057.8     3,167.3     127    ₹13,899       +3.6%     

  AFTER: Invested ₹8,937,433 | Cash ₹1,851 | Total ₹8,939,284 | Positions 18/20 | Slot ₹447,096

========================================================================
  REBALANCE #74  —  01 Jan 2024
  NAV: ₹9,516,565  |  Slot: ₹475,828  |  Cash: ₹1,851
========================================================================

  EXITS (3)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LUPIN       43     PHARMA                01-Aug-23   977.8       1,299.0     377    ₹121,106      +32.9%    153d  
  SUNDARMFIN  66     FINANCIAL_SERVICES    01-Nov-23   3,127.1     3,435.7     124    ₹38,260       +9.9%     61d   
  ZFCVINDIA   54     CAPITAL_GOODS         01-Nov-23   15,413.8    16,266.4    25     ₹21,314       +5.5%     61d   

  ENTRIES (2)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  HAL         5      CAPITAL_GOODS         3.090    0.99   +126.4%   +47.3%    2,745.8     173    ₹475,025    
  BDL         13     CAPITAL_GOODS         2.566    0.82   +108.5%   +68.5%    852.1       558    ₹475,450    

  HOLDS (15)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BSE         1      FINANCIAL_SERVICES    03-Jul-23   218.0       728.0       1601   ₹816,483      +233.9%   
  PRESTIGE    2      REAL_ESTATE           01-Nov-23   748.3       1,186.2     522    ₹228,597      +58.5%    
  NMDC        12     METALS                03-Oct-23   44.2        63.2        8926   ₹169,076      +42.8%    
  NTPC        6      ENERGY                01-Aug-23   207.6       292.4       1776   ₹150,619      +40.8%    
  TRENT       4      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     2,997.4     178    ₹142,970      +36.6%    
  COALINDIA   11     METALS                01-Nov-23   254.9       331.9       1533   ₹118,055      +30.2%    
  SOLARINDS   24     CHEMICALS             01-Nov-23   5,514.0     6,769.0     70     ₹87,851       +22.8%    
  PERSISTENT  26     IT                    01-Nov-23   3,057.8     3,602.8     127    ₹69,209       +17.8%    
  IOC         3      OIL_GAS               01-Dec-23   100.2       117.3       4464   ₹76,557       +17.1%    
  JSL         15     METALS                03-Oct-23   490.4       568.9       805    ₹63,131       +16.0%    
  LT          20     CAPITAL_GOODS         03-Oct-23   2,991.9     3,432.1     131    ₹57,670       +14.7%    
  BAJAJ-AUTO  7      AUTO                  01-Dec-23   5,849.1     6,482.8     76     ₹48,162       +10.8%    
  ALKEM       8      PHARMA                01-Dec-23   4,533.0     4,993.4     98     ₹45,122       +10.2%    
  TVSMOTOR    9      AUTO                  01-Dec-23   1,887.8     1,995.8     236    ₹25,486       +5.7%     
  AUROPHARMA  10     PHARMA                01-Dec-23   1,028.4     1,074.6     434    ₹20,033       +4.5%     

  AFTER: Invested ₹9,142,766 | Cash ₹372,670 | Total ₹9,515,436 | Positions 17/20 | Slot ₹475,828

========================================================================
  REBALANCE #75  —  01 Feb 2024
  NAV: ₹9,950,686  |  Slot: ₹497,534  |  Cash: ₹372,670
========================================================================

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  NMDC        31     METALS                03-Oct-23   44.2        67.3        8926   ₹206,278      +52.2%    121d  
  SOLARINDS   93     CHEMICALS             01-Nov-23   5,514.0     6,333.4     70     ₹57,362       +14.9%    92d   
  JSL         61     METALS                03-Oct-23   490.4       560.0       805    ₹56,024       +14.2%    121d  
  LT          67     CAPITAL_GOODS         03-Oct-23   2,991.9     3,308.0     131    ₹41,416       +10.6%    121d  
  ALKEM       42     PHARMA                01-Dec-23   4,533.0     4,734.7     98     ₹19,770       +4.5%     62d   

  ENTRIES (5)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  IRFC        1      FINANCIAL_SERVICES    4.292    1.11   +423.4%   +136.1%   164.1       3031   ₹497,472    
  NHPC        7      ENERGY                2.950    0.83   +124.2%   +79.2%    85.8        5799   ₹497,497    
  GLAXO       8      PHARMA                2.819    0.56   +76.8%    +55.8%    2,162.3     230    ₹497,337    
  RVNL        9      INFRASTRUCTURE        2.783    1.17   +297.9%   +90.3%    293.9       1692   ₹497,333    
  TATAPOWER   10     ENERGY                2.682    1.13   +89.4%    +62.6%    384.8       1292   ₹497,152    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BSE         4      FINANCIAL_SERVICES    03-Jul-23   218.0       824.0       1601   ₹970,125      +277.9%   
  PRESTIGE    13     REAL_ESTATE           01-Nov-23   748.3       1,239.1     522    ₹256,205      +65.6%    
  NTPC        11     ENERGY                01-Aug-23   207.6       304.0       1776   ₹171,158      +46.4%    
  TRENT       3      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     3,098.0     178    ₹160,867      +41.2%    
  COALINDIA   17     METALS                01-Nov-23   254.9       353.5       1533   ₹151,234      +38.7%    
  IOC         2      OIL_GAS               01-Dec-23   100.2       134.8       4464   ₹154,722      +34.6%    
  PERSISTENT  29     IT                    01-Nov-23   3,057.8     4,099.4     127    ₹132,277      +34.1%    
  BAJAJ-AUTO  5      AUTO                  01-Dec-23   5,849.1     7,406.3     76     ₹118,351      +26.6%    
  HAL         6      CAPITAL_GOODS         01-Jan-24   2,745.8     2,912.1     173    ₹28,774       +6.1%     
  TVSMOTOR    19     AUTO                  01-Dec-23   1,887.8     1,972.3     236    ₹19,934       +4.5%     
  AUROPHARMA  22     PHARMA                01-Dec-23   1,028.4     1,065.3     434    ₹15,997       +3.6%     
  BDL         28     CAPITAL_GOODS         01-Jan-24   852.1       840.9       558    ₹-6,223       -1.3%     

  AFTER: Invested ₹9,672,188 | Cash ₹275,546 | Total ₹9,947,734 | Positions 17/20 | Slot ₹497,534

========================================================================
  REBALANCE #76  —  01 Mar 2024
  NAV: ₹10,117,205  |  Slot: ₹505,860  |  Cash: ₹275,546
========================================================================

  EXITS (6)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PRESTIGE    43     REAL_ESTATE           01-Nov-23   748.3       1,165.0     522    ₹217,528      +55.7%    121d  
  BDL         36     CAPITAL_GOODS         01-Jan-24   852.1       908.4       558    ₹31,419       +6.6%     60d   
  AUROPHARMA  61     PHARMA                01-Dec-23   1,028.4     1,022.6     434    ₹-2,534       -0.6%     91d   
  GLAXO       41     PHARMA                01-Feb-24   2,162.3     2,083.8     230    ₹-18,069      -3.6%     29d   
  IRFC        —      FINANCIAL_SERVICES    01-Feb-24   164.1       142.4       3031   ₹-65,881      -13.2%    29d   
  RVNL        —      INFRASTRUCTURE        01-Feb-24   293.9       243.2       1692   ₹-85,828      -17.3%    29d   

  ENTRIES (6)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  OIL         1      OIL_GAS               3.259    0.95   +148.2%   +93.0%    366.3       1381   ₹505,806    
  OFSS        2      IT                    3.138    0.96   +156.5%   +92.2%    6,933.3     72     ₹499,195    
  ZYDUSLIFE   3      PHARMA                3.011    0.39   +99.4%    +45.1%    912.6       554    ₹505,577    
  ADANIGREEN  5      ENERGY                2.915    1.03   +247.1%   +91.4%    1,969.6     256    ₹504,205    
  LUPIN       6      PHARMA                2.858    0.42   +143.4%   +26.8%    1,607.5     314    ₹504,743    
  ZOMATO      8      CONSUMER_DISCRETIONARY  2.819    0.12   +251.3%   +42.8%    166.5       3038   ₹505,827    

  HOLDS (11)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BSE         7      FINANCIAL_SERVICES    03-Jul-23   218.0       772.1       1601   ₹887,061      +254.1%   
  TRENT       4      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     3,895.2     178    ₹302,771      +77.5%    
  NTPC        12     ENERGY                01-Aug-23   207.6       327.5       1776   ₹212,831      +57.7%    
  COALINDIA   20     METALS                01-Nov-23   254.9       392.4       1533   ₹210,919      +54.0%    
  IOC         9      OIL_GAS               01-Dec-23   100.2       152.5       4464   ₹233,691      +52.3%    
  PERSISTENT  35     IT                    01-Nov-23   3,057.8     4,249.2     127    ₹151,309      +39.0%    
  BAJAJ-AUTO  10     AUTO                  01-Dec-23   5,849.1     7,778.1     76     ₹146,607      +33.0%    
  TVSMOTOR    30     AUTO                  01-Dec-23   1,887.8     2,216.8     236    ₹77,637       +17.4%    
  HAL         14     CAPITAL_GOODS         01-Jan-24   2,745.8     3,086.9     173    ₹59,007       +12.4%    
  NHPC        19     ENERGY                01-Feb-24   85.8        85.6        5799   ₹-933         -0.2%     
  TATAPOWER   31     ENERGY                01-Feb-24   384.8       373.5       1292   ₹-14,639      -2.9%     

  AFTER: Invested ₹9,985,862 | Cash ₹127,751 | Total ₹10,113,613 | Positions 17/20 | Slot ₹505,860

========================================================================
  REBALANCE #77  —  01 Apr 2024
  NAV: ₹10,561,355  |  Slot: ₹528,068  |  Cash: ₹127,751
========================================================================

  EXITS (7)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  NTPC        —      ENERGY                01-Aug-23   207.6       326.4       1776   ₹210,884      +57.2%    244d  
  COALINDIA   —      METALS                01-Nov-23   254.9       388.7       1533   ₹205,123      +52.5%    152d  
  IOC         —      OIL_GAS               01-Dec-23   100.2       152.4       4464   ₹233,088      +52.1%    122d  
  PERSISTENT  62     IT                    01-Nov-23   3,057.8     3,949.9     127    ₹113,297      +29.2%    152d  
  TATAPOWER   —      ENERGY                01-Feb-24   384.8       402.7       1292   ₹23,141       +4.7%     60d   
  NHPC        —      ENERGY                01-Feb-24   85.8        86.2        5799   ₹2,659        +0.5%     60d   
  ADANIGREEN  —      ENERGY                01-Mar-24   1,969.6     1,888.3     256    ₹-20,800      -4.1%     31d   

  ENTRIES (7)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  SUNPHARMA   7      PHARMA                2.799    0.47   +71.1%    +30.8%    1,598.7     330    ₹527,567    
  BOSCHLTD    8      AUTO_ANCILLARY        2.749    0.68   +70.7%    +38.6%    29,732.3    17     ₹505,449    
  CUMMINSIND  9      CAPITAL_GOODS         2.742    0.77   +84.2%    +52.5%    2,927.3     180    ₹526,920    
  TORNTPOWER  11     ENERGY                2.708    0.67   +172.7%   +59.9%    1,384.1     381    ₹527,354    
  SUZLON      12     ENERGY                2.680    1.18   +433.5%   +11.6%    41.3        12770  ₹528,039    
  INDHOTEL    14     CONSUMER_DISCRETIONARY  2.432    1.02   +91.4%    +37.7%    597.8       883    ₹527,891    
  KALYANKJIL  16     CONSUMER_DISCRETIONARY  2.210    0.53   +260.5%   +20.9%    423.7       1246   ₹527,940    

  HOLDS (10)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BSE         2      FINANCIAL_SERVICES    03-Jul-23   218.0       895.5       1601   ₹1,084,643    +310.7%   
  TRENT       10     CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     3,880.8     178    ₹300,203      +76.9%    
  BAJAJ-AUTO  4      AUTO                  01-Dec-23   5,849.1     8,747.6     76     ₹220,285      +49.6%    
  HAL         15     CAPITAL_GOODS         01-Jan-24   2,745.8     3,330.6     173    ₹101,171      +21.3%    
  OFSS        1      IT                    01-Mar-24   6,933.3     8,048.0     72     ₹80,259       +16.1%    
  TVSMOTOR    36     AUTO                  01-Dec-23   1,887.8     2,122.8     236    ₹55,465       +12.4%    
  ZOMATO      5      CONSUMER_DISCRETIONARY  01-Mar-24   166.5       184.5       3038   ₹54,684       +10.8%    
  ZYDUSLIFE   3      PHARMA                01-Mar-24   912.6       986.0       554    ₹40,677       +8.0%     
  OIL         13     OIL_GAS               01-Mar-24   366.3       373.1       1381   ₹9,471        +1.9%     
  LUPIN       6      PHARMA                01-Mar-24   1,607.5     1,606.7     314    ₹-249         -0.0%     

  AFTER: Invested ₹10,243,663 | Cash ₹313,333 | Total ₹10,556,996 | Positions 17/20 | Slot ₹528,068

========================================================================
  REBALANCE #78  —  02 May 2024
  NAV: ₹10,863,403  |  Slot: ₹543,170  |  Cash: ₹313,333
========================================================================
  [SECTOR CAP≤4] dropped: ABB

  EXITS (4)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TVSMOTOR    51     AUTO                  01-Dec-23   1,887.8     2,057.2     236    ₹39,972       +9.0%     153d  
  OFSS        48     IT                    01-Mar-24   6,933.3     6,978.4     72     ₹3,250        +0.7%     62d   
  INDHOTEL    44     CONSUMER_DISCRETIONARY  01-Apr-24   597.8       572.7       883    ₹-22,164      -4.2%     31d   
  SUNPHARMA   49     PHARMA                01-Apr-24   1,598.7     1,490.5     330    ₹-35,700      -6.8%     31d   

  ENTRIES (4)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  DIXON       6      CONSUMER_DURABLES     2.871    0.78   +187.2%   +43.0%    8,403.6     64     ₹537,827    
  INDIGO      8      CAPITAL_GOODS         2.652    0.83   +106.7%   +43.5%    4,101.3     132    ₹541,374    
  SUNDARMFIN  10     FINANCIAL_SERVICES    2.512    0.46   +109.9%   +37.3%    4,786.9     113    ₹540,916    
  VOLTAS      13     CONSUMER_DURABLES     2.380    0.81   +73.9%    +47.4%    1,468.4     369    ₹541,827    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BSE         1      FINANCIAL_SERVICES    03-Jul-23   218.0       944.7       1601   ₹1,163,396    +333.3%   
  TRENT       2      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     4,639.5     178    ₹435,255      +111.4%   
  BAJAJ-AUTO  18     AUTO                  01-Dec-23   5,849.1     8,813.8     76     ₹225,321      +50.7%    
  HAL         5      CAPITAL_GOODS         01-Jan-24   2,745.8     3,862.8     173    ₹193,247      +40.7%    
  ZOMATO      3      CONSUMER_DISCRETIONARY  01-Mar-24   166.5       195.4       3038   ₹87,950       +17.4%    
  CUMMINSIND  4      CAPITAL_GOODS         01-Apr-24   2,927.3     3,220.5     180    ₹52,773       +10.0%    
  OIL         7      OIL_GAS               01-Mar-24   366.3       398.0       1381   ₹43,866       +8.7%     
  ZYDUSLIFE   12     PHARMA                01-Mar-24   912.6       974.4       554    ₹34,230       +6.8%     
  TORNTPOWER  9      ENERGY                01-Apr-24   1,384.1     1,465.4     381    ₹30,974       +5.9%     
  LUPIN       17     PHARMA                01-Mar-24   1,607.5     1,630.3     314    ₹7,177        +1.4%     
  BOSCHLTD    11     AUTO_ANCILLARY        01-Apr-24   29,732.3    30,113.4    17     ₹6,479        +1.3%     
  SUZLON      15     ENERGY                01-Apr-24   41.3        41.7        12770  ₹4,470        +0.8%     
  KALYANKJIL  16     CONSUMER_DISCRETIONARY  01-Apr-24   423.7       409.7       1246   ₹-17,478      -3.3%     

  AFTER: Invested ₹10,726,475 | Cash ₹134,361 | Total ₹10,860,836 | Positions 17/20 | Slot ₹543,170

========================================================================
  REBALANCE #79  —  03 Jun 2024
  NAV: ₹11,192,167  |  Slot: ₹559,608  |  Cash: ₹134,361
========================================================================
  [SECTOR CAP≤4] dropped: ABB, BDL, THERMAX

  EXITS (7)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HAL         —      CAPITAL_GOODS         01-Jan-24   2,745.8     5,160.9     173    ₹417,818      +88.0%    154d  
  SUZLON      —      ENERGY                01-Apr-24   41.3        50.0        12770  ₹110,461      +20.9%    63d   
  ZOMATO      44     CONSUMER_DISCRETIONARY  01-Mar-24   166.5       175.4       3038   ₹27,190       +5.4%     94d   
  BOSCHLTD    61     AUTO_ANCILLARY        01-Apr-24   29,732.3    29,444.7    17     ₹-4,890       -1.0%     63d   
  LUPIN       52     PHARMA                01-Mar-24   1,607.5     1,567.3     314    ₹-12,614      -2.5%     94d   
  VOLTAS      38     CONSUMER_DURABLES     02-May-24   1,468.4     1,393.4     369    ₹-27,659      -5.1%     32d   
  SUNDARMFIN  62     FINANCIAL_SERVICES    02-May-24   4,786.9     4,377.0     113    ₹-46,321      -8.6%     32d   

  ENTRIES (7)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  HINDZINC    1      CHEMICALS             3.454    0.74   +140.0%   +128.5%   646.6       865    ₹559,342    
  SIEMENS     2      CAPITAL_GOODS         3.286    0.93   +114.1%   +59.3%    4,254.8     131    ₹557,385    
  MAZDOCK     5      CAPITAL_GOODS         2.820    1.11   +327.5%   +55.7%    1,602.9     349    ₹559,411    
  VEDL        9      METALS                2.618    1.09   +80.8%    +76.7%    398.1       1405   ₹559,271    
  PRESTIGE    11     REAL_ESTATE           2.590    0.94   +250.6%   +41.9%    1,731.5     323    ₹559,289    
  BHARTIARTL  13     TELECOM               2.405    0.60   +74.7%    +25.3%    1,371.9     407    ₹558,372    
  SOLARINDS   14     CHEMICALS             2.350    0.51   +158.6%   +44.8%    9,847.1     56     ₹551,438    

  HOLDS (10)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BSE         4      FINANCIAL_SERVICES    03-Jul-23   218.0       894.9       1601   ₹1,083,585    +310.4%   
  TRENT       7      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     4,658.2     178    ₹438,578      +112.3%   
  BAJAJ-AUTO  24     AUTO                  01-Dec-23   5,849.1     9,031.3     76     ₹241,850      +54.4%    
  CUMMINSIND  12     CAPITAL_GOODS         01-Apr-24   2,927.3     3,618.2     180    ₹124,361      +23.6%    
  DIXON       3      CONSUMER_DURABLES     02-May-24   8,403.6     9,878.1     64     ₹94,369       +17.5%    
  OIL         29     OIL_GAS               01-Mar-24   366.3       422.6       1381   ₹77,782       +15.4%    
  ZYDUSLIFE   33     PHARMA                01-Mar-24   912.6       1,018.0     554    ₹58,406       +11.6%    
  TORNTPOWER  15     ENERGY                01-Apr-24   1,384.1     1,474.4     381    ₹34,385       +6.5%     
  INDIGO      21     CAPITAL_GOODS         02-May-24   4,101.3     4,290.8     132    ₹25,011       +4.6%     
  KALYANKJIL  23     CONSUMER_DISCRETIONARY  01-Apr-24   423.7       388.9       1246   ₹-43,324      -8.2%     

  AFTER: Invested ₹10,896,504 | Cash ₹291,027 | Total ₹11,187,531 | Positions 17/20 | Slot ₹559,608

========================================================================
  REBALANCE #80  —  01 Jul 2024
  NAV: ₹11,957,806  |  Slot: ₹597,890  |  Cash: ₹291,027
========================================================================

  EXITS (9)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DIXON       —      CONSUMER_DURABLES     02-May-24   8,403.6     12,436.5    64     ₹258,106      +48.0%    60d   
  MAZDOCK     —      CAPITAL_GOODS         03-Jun-24   1,602.9     2,162.0     349    ₹195,131      +34.9%    28d   
  ZYDUSLIFE   48     PHARMA                01-Mar-24   912.6       1,052.6     554    ₹77,556       +15.3%    122d  
  SIEMENS     —      CAPITAL_GOODS         03-Jun-24   4,254.8     4,604.4     131    ₹45,788       +8.2%     28d   
  PRESTIGE    —      REAL_ESTATE           03-Jun-24   1,731.5     1,836.5     323    ₹33,908       +6.1%     28d   
  INDIGO      47     CAPITAL_GOODS         02-May-24   4,101.3     4,215.0     132    ₹15,009       +2.8%     60d   
  TORNTPOWER  68     ENERGY                01-Apr-24   1,384.1     1,420.7     381    ₹13,941       +2.6%     91d   
  SOLARINDS   43     CHEMICALS             03-Jun-24   9,847.1     10,085.1    56     ₹13,327       +2.4%     28d   
  VEDL        —      METALS                03-Jun-24   398.1       404.3       1405   ₹8,734        +1.6%     28d   

  ENTRIES (9)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  MOTHERSON   1      AUTO_ANCILLARY        3.489    0.88   +144.0%   +67.8%    129.4       4620   ₹597,765    
  UNOMINDA    4      AUTO_ANCILLARY        2.835    0.66   +95.8%    +69.8%    1,138.9     524    ₹596,783    
  M&M         5      AUTO                  2.796    1.11   +107.2%   +54.3%    2,832.2     211    ₹597,593    
  BDL         6      CAPITAL_GOODS         2.674    1.12   +171.7%   +87.7%    1,602.2     373    ₹597,627    
  ESCORTS     7      AUTO                  2.631    0.61   +89.5%    +50.5%    4,067.5     146    ₹593,848    
  SUZLON      8      ENERGY                2.537    1.01   +263.8%   +42.7%    52.9        11295  ₹597,844    
  COROMANDEL  11     CHEMICALS             2.438    0.87   +71.0%    +50.3%    1,574.0     379    ₹596,553    
  EMAMILTD    12     FMCG                  2.337    0.52   +76.2%    +65.7%    693.5       862    ₹597,827    
  ENDURANCE   14     AUTO_ANCILLARY        2.096    0.59   +71.3%    +53.5%    2,685.6     222    ₹596,213    

  HOLDS (8)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  BSE         10     FINANCIAL_SERVICES    03-Jul-23   218.0       855.3       1601   ₹1,020,302    +292.3%   
  TRENT       2      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     5,510.9     178    ₹590,361      +151.2%   
  BAJAJ-AUTO  24     AUTO                  01-Dec-23   5,849.1     9,296.8     76     ₹262,026      +58.9%    
  CUMMINSIND  13     CAPITAL_GOODS         01-Apr-24   2,927.3     3,884.9     180    ₹172,353      +32.7%    
  OIL         21     OIL_GAS               01-Mar-24   366.3       450.7       1381   ₹116,580      +23.0%    
  KALYANKJIL  9      CONSUMER_DISCRETIONARY  01-Apr-24   423.7       494.9       1246   ₹88,754       +16.8%    
  BHARTIARTL  17     TELECOM               03-Jun-24   1,371.9     1,434.0     407    ₹25,247       +4.5%     
  HINDZINC    3      CHEMICALS             03-Jun-24   646.6       609.8       865    ₹-31,847      -5.7%     

  AFTER: Invested ₹11,478,406 | Cash ₹473,022 | Total ₹11,951,428 | Positions 17/20 | Slot ₹597,890

========================================================================
  REBALANCE #81  —  01 Aug 2024
  NAV: ₹12,402,648  |  Slot: ₹620,132  |  Cash: ₹473,022
========================================================================

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BSE         40     FINANCIAL_SERVICES    03-Jul-23   218.0       878.3       1601   ₹1,057,089    +302.8%   395d  
  OIL         —      OIL_GAS               01-Mar-24   366.3       567.3       1381   ₹277,637      +54.9%    153d  
  CUMMINSIND  37     CAPITAL_GOODS         01-Apr-24   2,927.3     3,738.1     180    ₹145,932      +27.7%    122d  
  BHARTIARTL  32     TELECOM               03-Jun-24   1,371.9     1,484.7     407    ₹45,919       +8.2%     59d   
  ENDURANCE   47     AUTO_ANCILLARY        01-Jul-24   2,685.6     2,550.7     222    ₹-29,953      -5.0%     31d   

  ENTRIES (7)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  OFSS        6      IT                    2.685    0.95   +186.5%   +48.1%    10,120.6    61     ₹617,359    
  ZYDUSLIFE   7      PHARMA                2.522    0.63   +103.4%   +30.5%    1,227.3     505    ₹619,807    
  PERSISTENT  8      IT                    2.389    0.84   +91.3%    +42.7%    4,750.7     130    ₹617,593    
  ZOMATO      9      CONSUMER_DISCRETIONARY  2.363    0.53   +211.9%   +21.2%    234.1       2649   ₹620,104    
  TVSMOTOR    10     AUTO                  2.270    0.70   +92.9%    +25.4%    2,564.6     241    ₹618,060    
  TORNTPOWER  12     ENERGY                2.143    0.91   +201.1%   +22.0%    1,782.5     347    ₹618,518    
  LUPIN       13     PHARMA                2.141    0.39   +107.4%   +19.2%    1,941.3     319    ₹619,267    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TRENT       1      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     5,765.8     178    ₹635,738      +162.8%   
  BAJAJ-AUTO  26     AUTO                  01-Dec-23   5,849.1     9,490.0     76     ₹276,710      +62.2%    
  KALYANKJIL  5      CONSUMER_DISCRETIONARY  01-Apr-24   423.7       561.8       1246   ₹172,055      +32.6%    
  SUZLON      2      ENERGY                01-Jul-24   52.9        68.0        11295  ₹169,990      +28.4%    
  EMAMILTD    3      FMCG                  01-Jul-24   693.5       792.6       862    ₹85,392       +14.3%    
  COROMANDEL  11     CHEMICALS             01-Jul-24   1,574.0     1,615.9     379    ₹15,892       +2.7%     
  ESCORTS     27     AUTO                  01-Jul-24   4,067.5     4,093.5     146    ₹3,803        +0.6%     
  MOTHERSON   4      AUTO_ANCILLARY        01-Jul-24   129.4       129.1       4620   ₹-1,367       -0.2%     
  M&M         18     AUTO                  01-Jul-24   2,832.2     2,805.9     211    ₹-5,557       -0.9%     
  HINDZINC    17     CHEMICALS             03-Jun-24   646.6       601.3       865    ₹-39,196      -7.0%     
  UNOMINDA    15     AUTO_ANCILLARY        01-Jul-24   1,138.9     1,035.8     524    ₹-54,020      -9.1%     
  BDL         20     CAPITAL_GOODS         01-Jul-24   1,602.2     1,438.5     373    ₹-61,065      -10.2%    

  AFTER: Invested ₹12,227,310 | Cash ₹170,196 | Total ₹12,397,506 | Positions 19/20 | Slot ₹620,132

========================================================================
  REBALANCE #82  —  02 Sep 2024
  NAV: ₹12,905,766  |  Slot: ₹645,288  |  Cash: ₹170,196
========================================================================

  EXITS (6)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  EMAMILTD    57     FMCG                  01-Jul-24   693.5       780.2       862    ₹74,697       +12.5%    63d   
  M&M         58     AUTO                  01-Jul-24   2,832.2     2,754.9     211    ₹-16,316      -2.7%     63d   
  ESCORTS     129    AUTO                  01-Jul-24   4,067.5     3,732.5     146    ₹-48,898      -8.2%     63d   
  ZYDUSLIFE   65     PHARMA                01-Aug-24   1,227.3     1,099.0     505    ₹-64,832      -10.5%    32d   
  BDL         105    CAPITAL_GOODS         01-Jul-24   1,602.2     1,293.3     373    ₹-115,230     -19.3%    63d   
  HINDZINC    153    CHEMICALS             03-Jun-24   646.6       475.5       865    ₹-148,005     -26.5%    91d   

  ENTRIES (5)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  COLPAL      7      FMCG                  2.741    0.37   +86.5%    +36.6%    3,524.3     183    ₹644,953    
  ICICIGI     8      FINANCIAL_SERVICES    2.517    0.63   +65.6%    +38.8%    2,164.3     298    ₹644,950    
  HCLTECH     11     IT                    2.385    0.72   +59.2%    +37.5%    1,684.5     383    ₹645,146    
  INFY        13     IT                    2.361    0.77   +44.4%    +39.6%    1,886.6     342    ₹645,201    
  SUNPHARMA   14     PHARMA                2.343    0.47   +61.1%    +24.8%    1,787.5     360    ₹643,513    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TRENT       1      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     7,142.0     178    ₹880,703      +225.5%   
  BAJAJ-AUTO  5      AUTO                  01-Dec-23   5,849.1     10,851.1    76     ₹380,154      +85.5%    
  KALYANKJIL  3      CONSUMER_DISCRETIONARY  01-Apr-24   423.7       638.0       1246   ₹266,987      +50.6%    
  SUZLON      2      ENERGY                01-Jul-24   52.9        73.8        11295  ₹235,840      +39.4%    
  LUPIN       4      PHARMA                01-Aug-24   1,941.3     2,218.9     319    ₹88,578       +14.3%    
  COROMANDEL  22     CHEMICALS             01-Jul-24   1,574.0     1,724.4     379    ₹56,998       +9.6%     
  PERSISTENT  6      IT                    01-Aug-24   4,750.7     5,157.2     130    ₹52,837       +8.6%     
  TVSMOTOR    9      AUTO                  01-Aug-24   2,564.6     2,769.5     241    ₹49,388       +8.0%     
  ZOMATO      12     CONSUMER_DISCRETIONARY  01-Aug-24   234.1       244.4       2649   ₹27,444       +4.4%     
  UNOMINDA    27     AUTO_ANCILLARY        01-Jul-24   1,138.9     1,164.8     524    ₹13,583       +2.3%     
  OFSS        10     IT                    01-Aug-24   10,120.6    10,145.5    61     ₹1,519        +0.2%     
  MOTHERSON   25     AUTO_ANCILLARY        01-Jul-24   129.4       127.6       4620   ₹-8,114       -1.4%     
  TORNTPOWER  29     ENERGY                01-Aug-24   1,782.5     1,722.8     347    ₹-20,698      -3.3%     

  AFTER: Invested ₹12,711,872 | Cash ₹190,066 | Total ₹12,901,938 | Positions 18/20 | Slot ₹645,288

========================================================================
  REBALANCE #83  —  01 Oct 2024
  NAV: ₹13,488,082  |  Slot: ₹674,404  |  Cash: ₹190,066
========================================================================

  EXITS (4)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COROMANDEL  48     CHEMICALS             01-Jul-24   1,574.0     1,709.2     379    ₹51,243       +8.6%     92d   
  MOTHERSON   38     AUTO_ANCILLARY        01-Jul-24   129.4       139.2       4620   ₹45,382       +7.6%     92d   
  INFY        44     IT                    02-Sep-24   1,886.6     1,828.8     342    ₹-19,755      -3.1%     29d   
  UNOMINDA    99     AUTO_ANCILLARY        01-Jul-24   1,138.9     1,079.2     524    ₹-31,268      -5.2%     92d   

  ENTRIES (3)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  AJANTPHARM  8      PHARMA                2.707    0.19   +93.5%    +45.3%    3,188.6     211    ₹672,786    
  BSE         10     FINANCIAL_SERVICES    2.543    0.76   +222.9%   +55.2%    1,282.3     525    ₹673,184    
  BHARTIARTL  11     TELECOM               2.489    0.88   +87.6%    +20.4%    1,684.6     400    ₹673,847    

  HOLDS (14)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TRENT       1      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     7,606.1     178    ₹963,313      +246.6%   
  BAJAJ-AUTO  3      AUTO                  01-Dec-23   5,849.1     11,856.9    76     ₹456,599      +102.7%   
  KALYANKJIL  2      CONSUMER_DISCRETIONARY  01-Apr-24   423.7       748.0       1246   ₹404,020      +76.5%    
  SUZLON      5      ENERGY                01-Jul-24   52.9        79.7        11295  ₹302,819      +50.7%    
  ZOMATO      9      CONSUMER_DISCRETIONARY  01-Aug-24   234.1       274.1       2649   ₹106,119      +17.1%    
  PERSISTENT  24     IT                    01-Aug-24   4,750.7     5,435.4     130    ₹89,012       +14.4%    
  LUPIN       7      PHARMA                01-Aug-24   1,941.3     2,180.8     319    ₹76,420       +12.3%    
  TVSMOTOR    15     AUTO                  01-Aug-24   2,564.6     2,817.1     241    ₹60,856       +9.8%     
  SUNPHARMA   6      PHARMA                02-Sep-24   1,787.5     1,889.9     360    ₹36,854       +5.7%     
  COLPAL      4      FMCG                  02-Sep-24   3,524.3     3,709.4     183    ₹33,869       +5.3%     
  OFSS        27     IT                    01-Aug-24   10,120.6    10,613.9    61     ₹30,087       +4.9%     
  TORNTPOWER  18     ENERGY                01-Aug-24   1,782.5     1,816.0     347    ₹11,644       +1.9%     
  HCLTECH     25     IT                    02-Sep-24   1,684.5     1,693.6     383    ₹3,517        +0.5%     
  ICICIGI     28     FINANCIAL_SERVICES    02-Sep-24   2,164.3     2,132.8     298    ₹-9,369       -1.5%     

  AFTER: Invested ₹12,835,929 | Cash ₹649,755 | Total ₹13,485,684 | Positions 17/20 | Slot ₹674,404

========================================================================
  REBALANCE #84  —  01 Nov 2024
  NAV: ₹12,597,082  |  Slot: ₹629,854  |  Cash: ₹649,755
========================================================================
  [SECTOR CAP≤4] dropped: IPCALAB

  EXITS (4)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SUZLON      44     ENERGY                01-Jul-24   52.9        68.1        11295  ₹171,797      +28.7%    123d  
  TVSMOTOR    52     AUTO                  01-Aug-24   2,564.6     2,490.6     241    ₹-17,818      -2.9%     92d   
  ICICIGI     74     FINANCIAL_SERVICES    02-Sep-24   2,164.3     1,903.0     298    ₹-77,864      -12.1%    60d   
  COLPAL      83     FMCG                  02-Sep-24   3,524.3     2,977.1     183    ₹-100,140     -15.5%    60d   

  ENTRIES (4)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  FORTIS      3      HEALTHCARE            3.041    0.52   +95.9%    +25.6%    634.6       992    ₹629,523    
  POWERINDIA  4      CAPITAL_GOODS         2.788    1.19   +222.8%   +15.7%    13,919.7    45     ₹626,387    
  DIVISLAB    5      PHARMA                2.676    0.54   +69.7%    +18.3%    5,876.8     107    ₹628,817    
  LLOYDSME    11     CAPITAL_GOODS         2.407    1.16   +88.9%    +25.3%    977.4       644    ₹629,426    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TRENT       1      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     7,142.8     178    ₹880,845      +225.5%   
  BAJAJ-AUTO  26     AUTO                  01-Dec-23   5,849.1     9,631.8     76     ₹287,491      +64.7%    
  KALYANKJIL  6      CONSUMER_DISCRETIONARY  01-Apr-24   423.7       668.8       1246   ₹305,438      +57.9%    
  BSE         2      FINANCIAL_SERVICES    01-Oct-24   1,282.3     1,483.2     525    ₹105,498      +15.7%    
  LUPIN       9      PHARMA                01-Aug-24   1,941.3     2,184.1     319    ₹77,466       +12.5%    
  PERSISTENT  15     IT                    01-Aug-24   4,750.7     5,337.6     130    ₹76,297       +12.4%    
  ZOMATO      19     CONSUMER_DISCRETIONARY  01-Aug-24   234.1       249.0       2649   ₹39,470       +6.4%     
  SUNPHARMA   8      PHARMA                02-Sep-24   1,787.5     1,829.3     360    ₹15,043       +2.3%     
  TORNTPOWER  23     ENERGY                01-Aug-24   1,782.5     1,780.7     347    ₹-626         -0.1%     
  OFSS        13     IT                    01-Aug-24   10,120.6    10,038.5    61     ₹-5,010       -0.8%     
  HCLTECH     24     IT                    02-Sep-24   1,684.5     1,649.3     383    ₹-13,475      -2.1%     
  BHARTIARTL  10     TELECOM               01-Oct-24   1,684.6     1,603.0     400    ₹-32,627      -4.8%     
  AJANTPHARM  22     PHARMA                01-Oct-24   3,188.6     3,024.5     211    ₹-34,626      -5.1%     

  AFTER: Invested ₹11,979,699 | Cash ₹614,398 | Total ₹12,594,097 | Positions 17/20 | Slot ₹629,854

========================================================================
  REBALANCE #85  —  02 Dec 2024
  NAV: ₹12,755,692  |  Slot: ₹637,785  |  Cash: ₹614,398
========================================================================

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJAJ-AUTO  72     AUTO                  01-Dec-23   5,849.1     8,904.7     76     ₹232,226      +52.2%    367d  
  LLOYDSME    —      CAPITAL_GOODS         01-Nov-24   977.4       1,049.5     644    ₹46,430       +7.4%     31d   
  AJANTPHARM  53     PHARMA                01-Oct-24   3,188.6     2,992.8     211    ₹-41,309      -6.1%     62d   
  POWERINDIA  —      CAPITAL_GOODS         01-Nov-24   13,919.7    12,260.7    45     ₹-74,654      -11.9%    31d   
  TORNTPOWER  44     ENERGY                01-Aug-24   1,782.5     1,547.4     347    ₹-81,575      -13.2%    123d  

  ENTRIES (5)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  INDHOTEL    4      CONSUMER_DISCRETIONARY  3.004    1.13   +90.9%    +23.7%    798.7       798    ₹637,364    
  COFORGE     6      IT                    2.954    0.76   +56.8%    +37.7%    1,722.2     370    ₹637,210    
  POLICYBZR   10     FINANCIAL_SERVICES    2.547    0.75   +138.1%   +9.8%     1,945.1     327    ₹636,048    
  BOSCHLTD    12     AUTO_ANCILLARY        2.433    0.74   +70.6%    +7.8%     34,460.4    18     ₹620,288    
  NAUKRI      14     TELECOM               2.322    1.10   +78.8%    +10.2%    1,678.6     379    ₹636,195    

  HOLDS (12)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TRENT       9      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     6,799.4     178    ₹819,720      +209.9%   
  KALYANKJIL  8      CONSUMER_DISCRETIONARY  01-Apr-24   423.7       720.0       1246   ₹369,172      +69.9%    
  PERSISTENT  11     IT                    01-Aug-24   4,750.7     5,875.8     130    ₹146,265      +23.7%    
  ZOMATO      7      CONSUMER_DISCRETIONARY  01-Aug-24   234.1       282.5       2649   ₹128,238      +20.7%    
  BSE         3      FINANCIAL_SERVICES    01-Oct-24   1,282.3     1,516.5     525    ₹122,978      +18.3%    
  OFSS        2      IT                    01-Aug-24   10,120.6    11,378.1    61     ₹76,702       +12.4%    
  FORTIS      5      HEALTHCARE            01-Nov-24   634.6       676.2       992    ₹41,218       +6.5%     
  DIVISLAB    1      PHARMA                01-Nov-24   5,876.8     6,227.0     107    ₹37,472       +6.0%     
  LUPIN       38     PHARMA                01-Aug-24   1,941.3     2,056.8     319    ₹36,839       +5.9%     
  HCLTECH     19     IT                    02-Sep-24   1,684.5     1,756.4     383    ₹27,536       +4.3%     
  SUNPHARMA   23     PHARMA                02-Sep-24   1,787.5     1,780.3     360    ₹-2,622       -0.4%     
  BHARTIARTL  13     TELECOM               01-Oct-24   1,684.6     1,630.0     400    ₹-21,857      -3.2%     

  AFTER: Invested ₹12,235,633 | Cash ₹516,298 | Total ₹12,751,931 | Positions 17/20 | Slot ₹637,785

========================================================================
  REBALANCE #86  —  01 Jan 2025
  NAV: ₹13,402,561  |  Slot: ₹670,128  |  Cash: ₹516,298
========================================================================

  EXITS (1)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BOSCHLTD    40     AUTO_ANCILLARY        02-Dec-24   34,460.4    33,576.8    18     ₹-15,905      -2.6%     30d   

  ENTRIES (1)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  LLOYDSME    4      CAPITAL_GOODS         3.134    1.09   +108.4%   +33.2%    1,262.1     530    ₹668,890    

  HOLDS (16)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TRENT       14     CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     7,061.9     178    ₹866,449      +221.8%   
  KALYANKJIL  9      CONSUMER_DISCRETIONARY  01-Apr-24   423.7       773.6       1246   ₹435,949      +82.6%    
  BSE         6      FINANCIAL_SERVICES    01-Oct-24   1,282.3     1,803.0     525    ₹273,380      +40.6%    
  PERSISTENT  11     IT                    01-Aug-24   4,750.7     6,375.4     130    ₹211,212      +34.2%    
  LUPIN       8      PHARMA                01-Aug-24   1,941.3     2,350.3     319    ₹130,473      +21.1%    
  ZOMATO      18     CONSUMER_DISCRETIONARY  01-Aug-24   234.1       276.5       2649   ₹112,344      +18.1%    
  OFSS        3      IT                    01-Aug-24   10,120.6    11,715.6    61     ₹97,293       +15.8%    
  FORTIS      12     HEALTHCARE            01-Nov-24   634.6       707.7       992    ₹72,479       +11.5%    
  COFORGE     5      IT                    02-Dec-24   1,722.2     1,903.7     370    ₹67,143       +10.5%    
  INDHOTEL    2      CONSUMER_DISCRETIONARY  02-Dec-24   798.7       871.0       798    ₹57,725       +9.1%     
  POLICYBZR   1      FINANCIAL_SERVICES    02-Dec-24   1,945.1     2,119.2     327    ₹56,931       +9.0%     
  HCLTECH     24     IT                    02-Sep-24   1,684.5     1,794.3     383    ₹42,075       +6.5%     
  SUNPHARMA   23     PHARMA                02-Sep-24   1,787.5     1,860.4     360    ₹26,223       +4.1%     
  NAUKRI      17     TELECOM               02-Dec-24   1,678.6     1,734.9     379    ₹21,329       +3.4%     
  DIVISLAB    10     PHARMA                01-Nov-24   5,876.8     6,045.5     107    ₹18,049       +2.9%     
  BHARTIARTL  31     TELECOM               01-Oct-24   1,684.6     1,582.5     400    ₹-40,858      -6.1%     

  AFTER: Invested ₹12,950,770 | Cash ₹450,996 | Total ₹13,401,766 | Positions 17/20 | Slot ₹670,128

========================================================================
  REBALANCE #87  —  01 Feb 2025
  NAV: ₹11,793,573  |  Slot: ₹589,679  |  Cash: ₹450,996
========================================================================
  [SECTOR CAP≤4] dropped: SBICARD

  [REGIME OFF] Nifty 200 21,580.9 < SMA200 22,498.0 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TRENT       26     CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     6,184.1     178    ₹710,202      +181.8%   
  BSE         1      FINANCIAL_SERVICES    01-Oct-24   1,282.3     1,795.7     525    ₹269,551      +40.0%    
  PERSISTENT  23     IT                    01-Aug-24   4,750.7     5,896.4     130    ₹148,945      +24.1%    
  KALYANKJIL  97     CONSUMER_DISCRETIONARY  01-Apr-24   423.7       504.0       1246   ₹100,013      +18.9% ⚠  
  LUPIN       39     PHARMA                01-Aug-24   1,941.3     2,043.5     319    ₹32,622       +5.3%     
  ZOMATO      —      CONSUMER_DISCRETIONARY  01-Aug-24   234.1       236.3       2649   ₹5,907        +1.0%     
  INDHOTEL    3      CONSUMER_DISCRETIONARY  02-Dec-24   798.7       799.1       798    ₹318          +0.0%     
  FORTIS      31     HEALTHCARE            01-Nov-24   634.6       626.1       992    ₹-8,422       -1.3%     
  LLOYDSME    —      CAPITAL_GOODS         01-Jan-25   1,262.1     1,227.0     530    ₹-18,563      -2.8%     
  SUNPHARMA   47     PHARMA                02-Sep-24   1,787.5     1,714.9     360    ₹-26,135      -4.1%     
  BHARTIARTL  22     TELECOM               01-Oct-24   1,684.6     1,609.8     400    ₹-29,930      -4.4%     
  HCLTECH     60     IT                    02-Sep-24   1,684.5     1,605.9     383    ₹-30,086      -4.7%     
  DIVISLAB    19     PHARMA                01-Nov-24   5,876.8     5,593.1     107    ₹-30,357      -4.8%     
  COFORGE     30     IT                    02-Dec-24   1,722.2     1,600.2     370    ₹-45,118      -7.1%     
  NAUKRI      16     TELECOM               02-Dec-24   1,678.6     1,546.9     379    ₹-49,918      -7.8%     
  POLICYBZR   11     FINANCIAL_SERVICES    02-Dec-24   1,945.1     1,716.2     327    ₹-74,867      -11.8%    
  OFSS        84     IT                    01-Aug-24   10,120.6    8,249.1     61     ₹-114,162     -18.5% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): OFSS, KALYANKJIL

  AFTER: Invested ₹11,342,576 | Cash ₹450,996 | Total ₹11,793,573 | Positions 17/20 | Slot ₹589,679

========================================================================
  REBALANCE #88  —  03 Mar 2025
  NAV: ₹10,593,072  |  Slot: ₹529,654  |  Cash: ₹450,996
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJFINSV, CHOLAFIN, BAJAJHLDNG

  [REGIME OFF] Nifty 200 19,896.9 < SMA200 22,517.6 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TRENT       67     CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     4,942.6     178    ₹489,211      +125.3%   
  BSE         —      FINANCIAL_SERVICES    01-Oct-24   1,282.3     1,448.6     525    ₹87,321       +13.0%    
  PERSISTENT  33     IT                    01-Aug-24   4,750.7     5,259.5     130    ₹66,136       +10.7%    
  KALYANKJIL  93     CONSUMER_DISCRETIONARY  01-Apr-24   423.7       439.2       1246   ₹19,260       +3.6% ⚠   
  LUPIN       26     PHARMA                01-Aug-24   1,941.3     1,940.9     319    ₹-127         -0.0%     
  FORTIS      18     HEALTHCARE            01-Nov-24   634.6       628.4       992    ₹-6,193       -1.0%     
  ZOMATO      —      CONSUMER_DISCRETIONARY  01-Aug-24   234.1       222.1       2649   ₹-31,656      -5.1%     
  BHARTIARTL  5      TELECOM               01-Oct-24   1,684.6     1,582.6     400    ₹-40,819      -6.1%     
  DIVISLAB    12     PHARMA                01-Nov-24   5,876.8     5,515.1     107    ₹-38,697      -6.2%     
  INDHOTEL    —      CONSUMER_DISCRETIONARY  02-Dec-24   798.7       724.6       798    ₹-59,118      -9.3%     
  HCLTECH     99     IT                    02-Sep-24   1,684.5     1,490.6     383    ₹-74,237      -11.5% ⚠  
  SUNPHARMA   84     PHARMA                02-Sep-24   1,787.5     1,569.7     360    ₹-78,420      -12.2% ⚠  
  COFORGE     56     IT                    02-Dec-24   1,722.2     1,457.7     370    ₹-97,851      -15.4%    
  NAUKRI      —      TELECOM               02-Dec-24   1,678.6     1,383.5     379    ₹-111,842     -17.6%    
  LLOYDSME    —      CAPITAL_GOODS         01-Jan-25   1,262.1     982.4       530    ₹-148,236     -22.2%    
  POLICYBZR   36     FINANCIAL_SERVICES    02-Dec-24   1,945.1     1,451.8     327    ₹-161,293     -25.4%    
  OFSS        114    IT                    01-Aug-24   10,120.6    7,269.2     61     ₹-173,938     -28.2% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): SUNPHARMA, KALYANKJIL, HCLTECH, OFSS

  AFTER: Invested ₹10,142,076 | Cash ₹450,996 | Total ₹10,593,072 | Positions 17/20 | Slot ₹529,654

========================================================================
  REBALANCE #89  —  01 Apr 2025
  NAV: ₹11,236,950  |  Slot: ₹561,848  |  Cash: ₹450,996
========================================================================
  [SECTOR CAP≤4] dropped: BAJAJFINSV, CHOLAFIN

  [REGIME OFF] Nifty 200 21,070.8 < SMA200 22,473.0 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TRENT       73     CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     5,571.9     178    ₹601,227      +153.9%   
  BSE         7      FINANCIAL_SERVICES    01-Oct-24   1,282.3     1,816.3     525    ₹280,358      +41.6%    
  PERSISTENT  77     IT                    01-Aug-24   4,750.7     5,179.0     130    ₹55,672       +9.0% ⚠   
  FORTIS      17     HEALTHCARE            01-Nov-24   634.6       687.8       992    ₹52,811       +8.4%     
  KALYANKJIL  124    CONSUMER_DISCRETIONARY  01-Apr-24   423.7       456.7       1246   ₹41,125       +7.8% ⚠   
  BHARTIARTL  8      TELECOM               01-Oct-24   1,684.6     1,709.9     400    ₹10,096       +1.5%     
  LLOYDSME    —      CAPITAL_GOODS         01-Jan-25   1,262.1     1,275.9     530    ₹7,335        +1.1%     
  INDHOTEL    37     CONSUMER_DISCRETIONARY  02-Dec-24   798.7       803.4       798    ₹3,740        +0.6%     
  LUPIN       85     PHARMA                01-Aug-24   1,941.3     1,943.3     319    ₹650          +0.1% ⚠   
  SUNPHARMA   103    PHARMA                02-Sep-24   1,787.5     1,681.9     360    ₹-38,046      -5.9% ⚠   
  DIVISLAB    18     PHARMA                01-Nov-24   5,876.8     5,524.5     107    ₹-37,690      -6.0%     
  COFORGE     66     IT                    02-Dec-24   1,722.2     1,541.7     370    ₹-66,785      -10.5%    
  ZOMATO      —      CONSUMER_DISCRETIONARY  01-Aug-24   234.1       202.0       2649   ₹-84,980      -13.7%    
  HCLTECH     133    IT                    02-Sep-24   1,684.5     1,450.8     383    ₹-89,505      -13.9% ⚠  
  NAUKRI      —      TELECOM               02-Dec-24   1,678.6     1,352.3     379    ₹-123,689     -19.4%    
  POLICYBZR   93     FINANCIAL_SERVICES    02-Dec-24   1,945.1     1,514.6     327    ₹-140,774     -22.1% ⚠  
  OFSS        148    IT                    01-Aug-24   10,120.6    7,036.0     61     ₹-188,166     -30.5% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): PERSISTENT, LUPIN, POLICYBZR, SUNPHARMA, KALYANKJIL, HCLTECH, OFSS

  AFTER: Invested ₹10,785,954 | Cash ₹450,996 | Total ₹11,236,950 | Positions 17/20 | Slot ₹561,848

========================================================================
  REBALANCE #90  —  02 May 2025
  NAV: ₹11,717,429  |  Slot: ₹585,871  |  Cash: ₹450,996
========================================================================
  [SECTOR CAP≤4] dropped: CHOLAFIN

  [REGIME OFF] Nifty 200 22,006.0 < SMA200 22,380.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (17)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  TRENT       —      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     5,149.5     178    ₹526,043      +134.7%   
  BSE         —      FINANCIAL_SERVICES    01-Oct-24   1,282.3     2,099.0     525    ₹428,815      +63.7%    
  KALYANKJIL  45     CONSUMER_DISCRETIONARY  01-Apr-24   423.7       505.8       1246   ₹102,250      +19.4%    
  PERSISTENT  50     IT                    01-Aug-24   4,750.7     5,372.7     130    ₹80,857       +13.1%    
  BHARTIARTL  5      TELECOM               01-Oct-24   1,684.6     1,832.9     400    ₹59,304       +8.8%     
  FORTIS      12     HEALTHCARE            01-Nov-24   634.6       682.6       992    ₹47,659       +7.6%     
  LUPIN       51     PHARMA                01-Aug-24   1,941.3     2,046.3     319    ₹33,494       +5.4%     
  DIVISLAB    8      PHARMA                01-Nov-24   5,876.8     6,051.9     107    ₹18,741       +3.0%     
  SUNPHARMA   32     PHARMA                02-Sep-24   1,787.5     1,806.9     360    ₹6,962        +1.1%     
  ZOMATO      —      CONSUMER_DISCRETIONARY  01-Aug-24   234.1       234.3       2649   ₹530          +0.1%     
  INDHOTEL    —      CONSUMER_DISCRETIONARY  02-Dec-24   798.7       799.1       798    ₹279          +0.0%     
  LLOYDSME    —      CAPITAL_GOODS         01-Jan-25   1,262.1     1,196.3     530    ₹-34,848      -5.2%     
  HCLTECH     99     IT                    02-Sep-24   1,684.5     1,507.8     383    ₹-67,651      -10.5% ⚠  
  COFORGE     78     IT                    02-Dec-24   1,722.2     1,461.4     370    ₹-96,499      -15.1% ⚠  
  NAUKRI      —      TELECOM               02-Dec-24   1,678.6     1,416.1     379    ₹-99,505      -15.6%    
  POLICYBZR   77     FINANCIAL_SERVICES    02-Dec-24   1,945.1     1,590.0     327    ₹-116,118     -18.3% ⚠  
  OFSS        84     IT                    01-Aug-24   10,120.6    8,047.6     61     ₹-126,455     -20.5% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): POLICYBZR, COFORGE, OFSS, HCLTECH

  AFTER: Invested ₹11,266,433 | Cash ₹450,996 | Total ₹11,717,429 | Positions 17/20 | Slot ₹585,871

========================================================================
  REBALANCE #91  —  02 Jun 2025
  NAV: ₹12,449,552  |  Slot: ₹622,478  |  Cash: ₹450,996
========================================================================
  [SECTOR CAP≤4] dropped: AUBANK

  EXITS (13)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TRENT       —      CONSUMER_DISCRETIONARY  01-Nov-23   2,194.2     5,616.6     178    ₹609,186      +156.0%   579d  
  BSE         —      FINANCIAL_SERVICES    01-Oct-24   1,282.3     2,693.3     525    ₹740,798      +110.0%   244d  
  KALYANKJIL  46     CONSUMER_DISCRETIONARY  01-Apr-24   423.7       555.0       1246   ₹163,623      +31.0%    427d  
  PERSISTENT  55     IT                    01-Aug-24   4,750.7     5,484.5     130    ₹95,390       +15.4%    305d  
  LLOYDSME    —      CAPITAL_GOODS         01-Jan-25   1,262.1     1,353.5     530    ₹48,465       +7.2%     152d  
  ZOMATO      —      CONSUMER_DISCRETIONARY  01-Aug-24   234.1       241.2       2649   ₹18,834       +3.0%     305d  
  LUPIN       75     PHARMA                01-Aug-24   1,941.3     1,949.0     319    ₹2,457        +0.4%     305d  
  INDHOTEL    —      CONSUMER_DISCRETIONARY  02-Dec-24   798.7       781.3       798    ₹-13,884      -2.2%     182d  
  HCLTECH     71     IT                    02-Sep-24   1,684.5     1,564.5     383    ₹-45,945      -7.1%     273d  
  SUNPHARMA   77     PHARMA                02-Sep-24   1,787.5     1,658.3     360    ₹-46,513      -7.2%     273d  
  POLICYBZR   47     FINANCIAL_SERVICES    02-Dec-24   1,945.1     1,757.7     327    ₹-61,280      -9.6%     182d  
  NAUKRI      —      TELECOM               02-Dec-24   1,678.6     1,421.6     379    ₹-97,396      -15.3%    182d  
  OFSS        —      IT                    01-Aug-24   10,120.6    8,038.0     61     ₹-127,040     -20.6%    305d  

  ENTRIES (15)
  [52w filter blocked 1: ENDURANCE(-20.6%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  SOLARINDS   1      CHEMICALS             4.006    1.12   +68.3%    +83.8%    16,284.3    38     ₹618,802    
  MFSL        2      FINANCIAL_SERVICES    3.679    0.67   +58.9%    +46.9%    1,522.3     408    ₹621,098    
  COROMANDEL  3      CHEMICALS             3.551    0.83   +84.5%    +36.7%    2,265.8     274    ₹620,837    
  BDL         4      CAPITAL_GOODS         2.940    1.08   +28.4%    +95.2%    1,966.8     316    ₹621,509    
  GVT&D       5      CAPITAL_GOODS         2.824    0.92   +70.4%    +59.2%    2,295.2     271    ₹621,998    
  PAYTM       6      IT                    2.824    1.12   +159.1%   +22.4%    924.3       673    ₹622,088    
  BHARTIHEXA  7      TELECOM               2.786    1.13   +80.6%    +46.6%    1,840.5     338    ₹622,076    
  HDFCLIFE    8      FINANCIAL_SERVICES    2.473    0.72   +36.3%    +24.3%    764.6       814    ₹622,373    
  ICICIBANK   9      BANKING               2.338    0.97   +29.5%    +19.1%    1,439.4     432    ₹621,818    
  MRF         12     AUTO_ANCILLARY        2.166    0.75   +7.6%     +29.3%    140,530.3   4      ₹562,121    
  KPRMILL     14     TEXTILES              2.077    0.41   +41.0%    +38.2%    1,120.4     555    ₹621,838    
  JKCEMENT    15     CEMENT                2.054    0.81   +37.9%    +19.4%    5,464.8     113    ₹617,527    
  SBILIFE     16     FINANCIAL_SERVICES    2.054    0.79   +28.1%    +21.5%    1,800.0     345    ₹620,999    
  HDFCBANK    17     BANKING               2.030    0.88   +26.5%    +15.2%    953.2       653    ₹622,417    
  BAJAJHLDNG  18     FINANCIAL_SERVICES    1.962    0.84   +69.4%    +11.8%    13,325.0    46     ₹612,950    

  HOLDS (4)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  FORTIS      10     HEALTHCARE            01-Nov-24   634.6       721.5       992    ₹86,202       +13.7%    
  DIVISLAB    11     PHARMA                01-Nov-24   5,876.8     6,509.4     107    ₹67,685       +10.8%    
  BHARTIARTL  13     TELECOM               01-Oct-24   1,684.6     1,838.7     400    ₹61,645       +9.1%     
  COFORGE     19     IT                    02-Dec-24   1,722.2     1,705.3     370    ₹-6,245       -1.0%     

  AFTER: Invested ₹12,029,133 | Cash ₹409,435 | Total ₹12,438,568 | Positions 19/20 | Slot ₹622,478

========================================================================
  REBALANCE #92  —  01 Jul 2025
  NAV: ₹13,026,403  |  Slot: ₹651,320  |  Cash: ₹409,435
========================================================================
  [SECTOR CAP≤4] dropped: CRISIL, AUBANK

  EXITS (3)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PAYTM       —      IT                    02-Jun-25   924.3       930.2       673    ₹3,971        +0.6%     29d   
  ICICIBANK   44     BANKING               02-Jun-25   1,439.4     1,421.0     432    ₹-7,931       -1.3%     29d   
  KPRMILL     53     TEXTILES              02-Jun-25   1,120.4     1,102.4     555    ₹-9,997       -1.6%     29d   

  ENTRIES (3)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  ENDURANCE   9      AUTO_ANCILLARY        2.418    0.62   +9.6%     +46.2%    2,873.2     226    ₹649,342    
  POWERINDIA  17     CAPITAL_GOODS         2.143    1.07   +52.8%    +49.0%    19,284.4    33     ₹636,385    
  INDIGO      21     CAPITAL_GOODS         2.020    0.97   +41.2%    +16.3%    5,954.9     109    ₹649,088    

  HOLDS (16)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  FORTIS      16     HEALTHCARE            01-Nov-24   634.6       773.8       992    ₹138,121      +21.9%    
  BHARTIARTL  5      TELECOM               01-Oct-24   1,684.6     2,002.7     400    ₹127,217      +18.9%    
  DIVISLAB    12     PHARMA                01-Nov-24   5,876.8     6,825.4     107    ₹101,504      +16.1%    
  JKCEMENT    7      CEMENT                02-Jun-25   5,464.8     6,121.3     113    ₹74,175       +12.0%    
  COFORGE     6      IT                    02-Dec-24   1,722.2     1,909.3     370    ₹69,231       +10.9%    
  MFSL        1      FINANCIAL_SERVICES    02-Jun-25   1,522.3     1,653.9     408    ₹53,693       +8.6%     
  BHARTIHEXA  11     TELECOM               02-Jun-25   1,840.5     1,973.7     338    ₹45,044       +7.2%     
  BAJAJHLDNG  18     FINANCIAL_SERVICES    02-Jun-25   13,325.0    14,270.4    46     ₹43,489       +7.1%     
  HDFCLIFE    10     FINANCIAL_SERVICES    02-Jun-25   764.6       809.9       814    ₹36,885       +5.9%     
  SOLARINDS   2      CHEMICALS             02-Jun-25   16,284.3    17,185.7    38     ₹34,255       +5.5%     
  HDFCBANK    29     BANKING               02-Jun-25   953.2       1,003.6     653    ₹32,902       +5.3%     
  SBILIFE     14     FINANCIAL_SERVICES    02-Jun-25   1,800.0     1,859.9     345    ₹20,671       +3.3%     
  GVT&D       8      CAPITAL_GOODS         02-Jun-25   2,295.2     2,345.5     271    ₹13,633       +2.2%     
  COROMANDEL  19     CHEMICALS             02-Jun-25   2,265.8     2,307.7     274    ₹11,474       +1.8%     
  MRF         13     AUTO_ANCILLARY        02-Jun-25   140,530.3   142,277.6   4      ₹6,989        +1.2%     
  BDL         15     CAPITAL_GOODS         02-Jun-25   1,966.8     1,972.5     316    ₹1,794        +0.3%     

  AFTER: Invested ₹12,699,998 | Cash ₹324,108 | Total ₹13,024,105 | Positions 19/20 | Slot ₹651,320

========================================================================
  REBALANCE #93  —  01 Aug 2025
  NAV: ₹12,681,973  |  Slot: ₹634,099  |  Cash: ₹324,108
========================================================================

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MRF         34     AUTO_ANCILLARY        02-Jun-25   140,530.3   145,899.2   4      ₹21,476       +3.8%     60d   
  SBILIFE     58     FINANCIAL_SERVICES    02-Jun-25   1,800.0     1,793.7     345    ₹-2,171       -0.3%     60d   
  HDFCLIFE    57     FINANCIAL_SERVICES    02-Jun-25   764.6       741.7       814    ₹-18,630      -3.0%     60d   
  SOLARINDS   30     CHEMICALS             02-Jun-25   16,284.3    13,807.0    38     ₹-94,136      -15.2%    60d   
  BDL         73     CAPITAL_GOODS         02-Jun-25   1,966.8     1,559.0     316    ₹-128,859     -20.7%    60d   

  ENTRIES (5)
  [52w filter blocked 1: APARINDS(-25.0%)]
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  GLENMARK    2      PHARMA                3.587    0.73   +44.5%    +47.1%    2,062.7     307    ₹633,261    
  BOSCHLTD    5      AUTO_ANCILLARY        3.168    0.96   +17.2%    +36.8%    40,390.0    15     ₹605,850    
  MUTHOOTFIN  8      FINANCIAL_SERVICES    2.510    0.69   +45.1%    +15.3%    2,571.0     246    ₹632,456    
  GLAND       10     PHARMA                2.428    0.73   -2.9%     +40.0%    1,960.2     323    ₹633,157    
  ETERNAL     11     CONSUMER_DISCRETIONARY  2.380    1.10   +39.0%    +31.0%    304.8       2080   ₹633,880    

  HOLDS (14)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  FORTIS      4      HEALTHCARE            01-Nov-24   634.6       859.2       992    ₹222,853      +35.4%    
  GVT&D       1      CAPITAL_GOODS         02-Jun-25   2,295.2     2,857.9     271    ₹152,483      +24.5%    
  JKCEMENT    3      CEMENT                02-Jun-25   5,464.8     6,681.5     113    ₹137,482      +22.3%    
  COROMANDEL  7      CHEMICALS             02-Jun-25   2,265.8     2,580.2     274    ₹86,146       +13.9%    
  BHARTIARTL  25     TELECOM               01-Oct-24   1,684.6     1,884.4     400    ₹79,913       +11.9%    
  DIVISLAB    24     PHARMA                01-Nov-24   5,876.8     6,361.5     107    ₹51,864       +8.2%     
  POWERINDIA  6      CAPITAL_GOODS         01-Jul-25   19,284.4    20,544.0    33     ₹41,568       +6.5%     
  HDFCBANK    14     BANKING               02-Jun-25   953.2       1,006.1     653    ₹34,566       +5.6%     
  BAJAJHLDNG  12     FINANCIAL_SERVICES    02-Jun-25   13,325.0    13,779.8    46     ₹20,923       +3.4%     
  BHARTIHEXA  15     TELECOM               02-Jun-25   1,840.5     1,844.6     338    ₹1,399        +0.2%     
  COFORGE     19     IT                    02-Dec-24   1,722.2     1,698.5     370    ₹-8,777       -1.4%     
  INDIGO      21     CAPITAL_GOODS         01-Jul-25   5,954.9     5,778.7     109    ₹-19,206      -3.0%     
  MFSL        9      FINANCIAL_SERVICES    02-Jun-25   1,522.3     1,472.5     408    ₹-20,318      -3.3%     
  ENDURANCE   17     AUTO_ANCILLARY        01-Jul-25   2,873.2     2,476.6     226    ₹-89,631      -13.8%    

  AFTER: Invested ₹12,672,985 | Cash ₹5,261 | Total ₹12,678,246 | Positions 19/20 | Slot ₹634,099

========================================================================
  REBALANCE #94  —  01 Sep 2025
  NAV: ₹12,680,179  |  Slot: ₹634,009  |  Cash: ₹5,261
========================================================================

  EXITS (6)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DIVISLAB    52     PHARMA                01-Nov-24   5,876.8     6,093.0     107    ₹23,134       +3.7%     304d  
  HDFCBANK    38     BANKING               02-Jun-25   953.2       950.6       653    ₹-1,676       -0.3%     91d   
  POWERINDIA  35     CAPITAL_GOODS         01-Jul-25   19,284.4    18,942.0    33     ₹-11,299      -1.8%     62d   
  BHARTIHEXA  33     TELECOM               02-Jun-25   1,840.5     1,770.0     338    ₹-23,816      -3.8%     91d   
  GLAND       31     PHARMA                01-Aug-25   1,960.2     1,871.1     323    ₹-28,792      -4.5%     31d   
  BAJAJHLDNG  53     FINANCIAL_SERVICES    02-Jun-25   13,325.0    12,482.3    46     ₹-38,766      -6.3%     91d   

  ENTRIES (5)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  MARUTI      4      AUTO                  3.040    0.86   +20.4%    +22.4%    14,887.0    42     ₹625,254    
  EICHERMOT   5      AUTO                  2.866    1.11   +30.7%    +18.7%    6,280.0     100    ₹628,000    
  DALBHARAT   7      CHEMICALS             2.735    0.81   +32.5%    +16.1%    2,398.8     264    ₹633,282    
  ULTRACEMCO  12     CEMENT                2.369    0.98   +14.2%    +15.3%    12,826.0    49     ₹628,474    
  UNOMINDA    13     AUTO_ANCILLARY        2.358    1.09   +18.1%    +29.1%    1,313.6     482    ₹633,174    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  FORTIS      2      HEALTHCARE            01-Nov-24   634.6       924.7       992    ₹287,730      +45.7%    
  JKCEMENT    1      CEMENT                02-Jun-25   5,464.8     7,067.5     113    ₹181,100      +29.3%    
  GVT&D       8      CAPITAL_GOODS         02-Jun-25   2,295.2     2,796.5     271    ₹135,854      +21.8%    
  BHARTIARTL  22     TELECOM               01-Oct-24   1,684.6     1,900.6     400    ₹86,393       +12.8%    
  MFSL        9      FINANCIAL_SERVICES    02-Jun-25   1,522.3     1,629.2     408    ₹43,615       +7.0%     
  ETERNAL     10     CONSUMER_DISCRETIONARY  01-Aug-25   304.8       321.1       2080   ₹34,008       +5.4%     
  MUTHOOTFIN  6      FINANCIAL_SERVICES    01-Aug-25   2,571.0     2,687.3     246    ₹28,614       +4.5%     
  COROMANDEL  27     CHEMICALS             02-Jun-25   2,265.8     2,330.2     274    ₹17,642       +2.8%     
  ENDURANCE   17     AUTO_ANCILLARY        01-Jul-25   2,873.2     2,939.9     226    ₹15,075       +2.3%     
  COFORGE     23     IT                    02-Dec-24   1,722.2     1,756.5     370    ₹12,695       +2.0%     
  BOSCHLTD    3      AUTO_ANCILLARY        01-Aug-25   40,390.0    40,785.0    15     ₹5,925        +1.0%     
  INDIGO      30     CAPITAL_GOODS         01-Jul-25   5,954.9     5,670.0     109    ₹-31,058      -4.8%     
  GLENMARK    11     PHARMA                01-Aug-25   2,062.7     1,922.1     307    ₹-43,181      -6.8%     

  AFTER: Invested ₹12,148,513 | Cash ₹527,927 | Total ₹12,676,440 | Positions 18/20 | Slot ₹634,009

========================================================================
  REBALANCE #95  —  01 Oct 2025
  NAV: ₹12,731,889  |  Slot: ₹636,594  |  Cash: ₹527,927
========================================================================

  EXITS (4)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BHARTIARTL  78     TELECOM               01-Oct-24   1,684.6     1,867.6     400    ₹73,193       +10.9%    365d  
  ULTRACEMCO  69     CEMENT                01-Sep-25   12,826.0    12,095.0    49     ₹-35,819      -5.7%     30d   
  INDIGO      49     CAPITAL_GOODS         01-Jul-25   5,954.9     5,606.0     109    ₹-38,034      -5.9%     92d   
  COFORGE     —      IT                    02-Dec-24   1,722.2     1,593.5     370    ₹-47,633      -7.5%     303d  

  ENTRIES (4)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  TATAINVEST  1      FINANCIAL_SERVICES    4.180    0.91   +54.0%    +55.0%    1,058.0     601    ₹635,858    
  INDIANB     7      BANKING               2.695    1.00   +41.9%    +13.3%    737.0       863    ₹636,074    
  HEROMOTOCO  8      AUTO                  2.526    1.02   -6.6%     +29.9%    5,328.6     119    ₹634,108    
  NYKAA       10     CONSUMER_DISCRETIONARY  2.366    0.65   +20.0%    +14.0%    241.3       2638   ₹636,444    

  HOLDS (14)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  FORTIS      3      HEALTHCARE            01-Nov-24   634.6       989.7       992    ₹352,210      +55.9%    
  GVT&D       5      CAPITAL_GOODS         02-Jun-25   2,295.2     3,073.0     271    ₹210,785      +33.9%    
  MUTHOOTFIN  6      FINANCIAL_SERVICES    01-Aug-25   2,571.0     3,118.1     246    ₹134,605      +21.3%    
  JKCEMENT    13     CEMENT                02-Jun-25   5,464.8     6,305.0     113    ₹94,938       +15.4%    
  EICHERMOT   2      AUTO                  01-Sep-25   6,280.0     7,021.5     100    ₹74,150       +11.8%    
  ETERNAL     11     CONSUMER_DISCRETIONARY  01-Aug-25   304.8       329.0       2080   ₹50,440       +8.0%     
  MARUTI      4      AUTO                  01-Sep-25   14,887.0    15,965.0    42     ₹45,276       +7.2%     
  MFSL        17     FINANCIAL_SERVICES    02-Jun-25   1,522.3     1,621.9     408    ₹40,637       +6.5%     
  UNOMINDA    14     AUTO_ANCILLARY        01-Sep-25   1,313.6     1,323.4     482    ₹4,720        +0.7%     
  COROMANDEL  22     CHEMICALS             02-Jun-25   2,265.8     2,242.2     274    ₹-6,485       -1.0%     
  ENDURANCE   34     AUTO_ANCILLARY        01-Jul-25   2,873.2     2,812.6     226    ₹-13,695      -2.1%     
  BOSCHLTD    9      AUTO_ANCILLARY        01-Aug-25   40,390.0    38,320.0    15     ₹-31,050      -5.1%     
  GLENMARK    20     PHARMA                01-Aug-25   2,062.7     1,956.7     307    ₹-32,554      -5.1%     
  DALBHARAT   28     CHEMICALS             01-Sep-25   2,398.8     2,225.2     264    ₹-45,824      -7.2%     

  AFTER: Invested ₹12,206,120 | Cash ₹522,750 | Total ₹12,728,870 | Positions 18/20 | Slot ₹636,594

========================================================================
  REBALANCE #96  —  03 Nov 2025
  NAV: ₹12,592,630  |  Slot: ₹629,631  |  Cash: ₹522,750
========================================================================
  [SECTOR CAP≤4] dropped: BANKBARODA, AUBANK

  EXITS (8)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  JKCEMENT    74     CEMENT                02-Jun-25   5,464.8     5,899.5     113    ₹49,116       +8.0%     154d  
  ETERNAL     —      CONSUMER_DISCRETIONARY  01-Aug-25   304.8       322.6       2080   ₹37,128       +5.9%     94d   
  MFSL        47     FINANCIAL_SERVICES    02-Jun-25   1,522.3     1,559.9     408    ₹15,341       +2.5%     154d  
  COROMANDEL  104    CHEMICALS             02-Jun-25   2,265.8     2,135.8     274    ₹-35,633      -5.7%     154d  
  GLENMARK    114    PHARMA                01-Aug-25   2,062.7     1,898.4     307    ₹-50,453      -8.0%     94d   
  BOSCHLTD    129    AUTO_ANCILLARY        01-Aug-25   40,390.0    37,025.0    15     ₹-50,475      -8.3%     94d   
  DALBHARAT   88     CHEMICALS             01-Sep-25   2,398.8     2,081.9     264    ₹-83,660      -13.2%    63d   
  TATAINVEST  44     FINANCIAL_SERVICES    01-Oct-25   1,058.0     798.7       601    ₹-155,839     -24.5%    33d   

  ENTRIES (8)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  TVSMOTOR    5      AUTO                  2.916    1.17   +43.2%    +25.3%    3,498.3     179    ₹626,191    
  CANBK       6      FINANCIAL_SERVICES    2.888    1.09   +43.6%    +30.2%    139.6       4510   ₹629,596    
  BAJFINANCE  7      FINANCIAL_SERVICES    2.865    1.13   +51.9%    +18.4%    1,043.1     603    ₹628,989    
  BANKINDIA   9      FINANCIAL_SERVICES    2.651    1.12   +47.0%    +27.6%    142.1       4429   ₹629,494    
  SBIN        11     BANKING               2.569    0.97   +22.3%    +19.2%    932.9       674    ₹628,764    
  FEDERALBNK  14     BANKING               2.336    0.70   +29.4%    +18.2%    237.9       2646   ₹629,457    
  CUMMINSIND  15     CAPITAL_GOODS         2.306    1.14   +30.1%    +23.2%    4,359.9     144    ₹627,822    
  BEL         16     CAPITAL_GOODS         2.304    1.15   +57.6%    +10.5%    420.5       1497   ₹629,503    

  HOLDS (10)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  FORTIS      8      HEALTHCARE            01-Nov-24   634.6       1,030.7     992    ₹392,932      +62.4%    
  GVT&D       12     CAPITAL_GOODS         02-Jun-25   2,295.2     3,171.2     271    ₹237,397      +38.2%    
  MUTHOOTFIN  4      FINANCIAL_SERVICES    01-Aug-25   2,571.0     3,163.8     246    ₹145,850      +23.1%    
  INDIANB     1      BANKING               01-Oct-25   737.0       880.3       863    ₹123,668      +19.4%    
  EICHERMOT   2      AUTO                  01-Sep-25   6,280.0     7,023.5     100    ₹74,350       +11.8%    
  MARUTI      3      AUTO                  01-Sep-25   14,887.0    15,651.0    42     ₹32,088       +5.1%     
  NYKAA       13     CONSUMER_DISCRETIONARY  01-Oct-25   241.3       250.1       2638   ₹23,267       +3.7%     
  HEROMOTOCO  10     AUTO                  01-Oct-25   5,328.6     5,433.1     119    ₹12,431       +2.0%     
  ENDURANCE   40     AUTO_ANCILLARY        01-Jul-25   2,873.2     2,866.5     226    ₹-1,513       -0.2%     
  UNOMINDA    22     AUTO_ANCILLARY        01-Sep-25   1,313.6     1,263.7     482    ₹-24,082      -3.8%     

  AFTER: Invested ₹12,372,577 | Cash ₹214,080 | Total ₹12,586,657 | Positions 18/20 | Slot ₹629,631

========================================================================
  REBALANCE #97  —  01 Dec 2025
  NAV: ₹12,823,324  |  Slot: ₹641,166  |  Cash: ₹214,080
========================================================================
  [SECTOR CAP≤4] dropped: M&MFIN, AUBANK, BANKBARODA

  EXITS (4)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  FORTIS      55     HEALTHCARE            01-Nov-24   634.6       904.8       992    ₹268,088      +42.6%    395d  
  GVT&D       49     CAPITAL_GOODS         02-Jun-25   2,295.2     2,801.3     271    ₹137,155      +22.1%    182d  
  UNOMINDA    68     AUTO_ANCILLARY        01-Sep-25   1,313.6     1,308.1     482    ₹-2,649       -0.4%     91d   
  ENDURANCE   102    AUTO_ANCILLARY        01-Jul-25   2,873.2     2,689.6     226    ₹-41,493      -6.4%     153d  

  ENTRIES (4)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  GMRAIRPORT  12     INFRASTRUCTURE        2.509    0.81   +34.2%    +25.1%    107.6       5956   ₹641,104    
  AIAENG      17     METALS                2.208    0.77   +11.0%    +26.8%    3,854.1     166    ₹639,781    
  PNB         18     BANKING               2.115    1.05   +22.7%    +24.2%    125.3       5117   ₹641,160    
  RELIANCE    19     OIL_GAS               2.113    1.16   +21.4%    +15.4%    1,566.1     409    ₹640,535    

  HOLDS (14)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  MUTHOOTFIN  1      FINANCIAL_SERVICES    01-Aug-25   2,571.0     3,779.1     246    ₹297,213      +47.0%    
  INDIANB     3      BANKING               01-Oct-25   737.0       887.3       863    ₹129,709      +20.4%    
  HEROMOTOCO  13     AUTO                  01-Oct-25   5,328.6     6,175.1     119    ₹100,734      +15.9%    
  EICHERMOT   7      AUTO                  01-Sep-25   6,280.0     7,125.5     100    ₹84,550       +13.5%    
  NYKAA       14     CONSUMER_DISCRETIONARY  01-Oct-25   241.3       264.9       2638   ₹62,362       +9.8%     
  MARUTI      9      AUTO                  01-Sep-25   14,887.0    16,097.0    42     ₹50,820       +8.1%     
  FEDERALBNK  6      BANKING               03-Nov-25   237.9       256.6       2646   ₹49,507       +7.9%     
  CANBK       2      FINANCIAL_SERVICES    03-Nov-25   139.6       150.5       4510   ₹49,159       +7.8%     
  TVSMOTOR    15     AUTO                  03-Nov-25   3,498.3     3,649.0     179    ₹26,988       +4.3%     
  CUMMINSIND  23     CAPITAL_GOODS         03-Nov-25   4,359.9     4,523.6     144    ₹23,581       +3.8%     
  BANKINDIA   8      FINANCIAL_SERVICES    03-Nov-25   142.1       147.2       4429   ₹22,632       +3.6%     
  SBIN        11     BANKING               03-Nov-25   932.9       955.9       674    ₹15,492       +2.5%     
  BEL         28     CAPITAL_GOODS         03-Nov-25   420.5       415.5       1497   ₹-7,528       -1.2%     
  BAJFINANCE  10     FINANCIAL_SERVICES    03-Nov-25   1,043.1     1,021.1     603    ₹-13,266      -2.1%     

  AFTER: Invested ₹12,276,686 | Cash ₹543,596 | Total ₹12,820,281 | Positions 18/20 | Slot ₹641,166

========================================================================
  REBALANCE #98  —  01 Jan 2026
  NAV: ₹12,794,419  |  Slot: ₹639,721  |  Cash: ₹543,596
========================================================================
  [SECTOR CAP≤4] dropped: SBILIFE

  EXITS (5)
  Ticker      Rank   Sector                Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TVSMOTOR    —      AUTO                  03-Nov-25   3,498.3     3,781.2     179    ₹50,641       +8.1%     59d   
  HEROMOTOCO  29     AUTO                  01-Oct-25   5,328.6     5,729.8     119    ₹47,740       +7.5%     92d   
  PNB         38     BANKING               01-Dec-25   125.3       123.9       5117   ₹-6,959       -1.1%     31d   
  BEL         56     CAPITAL_GOODS         03-Nov-25   420.5       396.0       1497   ₹-36,670      -5.8%     59d   
  BAJFINANCE  47     FINANCIAL_SERVICES    03-Nov-25   1,043.1     973.1       603    ₹-42,210      -6.7%     59d   

  ENTRIES (5)
  Ticker      Rank   Sector                Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital     
  ──────────  ─────  ────────────────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────
  AUBANK      2      FINANCIAL_SERVICES    3.479    0.67   +81.2%    +36.6%    999.5       640    ₹639,648    
  UPL         4      CHEMICALS             2.928    1.01   +61.9%    +22.8%    805.3       794    ₹639,448    
  TITAN       10     CONSUMER_DISCRETIONARY  2.384    0.74   +22.7%    +20.3%    4,049.3     157    ₹635,740    
  INDUSTOWER  12     TELECOM               2.275    0.97   +32.2%    +27.1%    435.8       1467   ₹639,319    
  IDFCFIRSTB  13     BANKING               2.271    1.06   +37.4%    +22.7%    85.6        7472   ₹639,678    

  HOLDS (13)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  MUTHOOTFIN  3      FINANCIAL_SERVICES    01-Aug-25   2,571.0     3,806.8     246    ₹304,019      +48.1%    
  EICHERMOT   20     AUTO                  01-Sep-25   6,280.0     7,348.0     100    ₹106,800      +17.0%    
  INDIANB     21     BANKING               01-Oct-25   737.0       832.6       863    ₹82,460       +13.0%    
  MARUTI      11     AUTO                  01-Sep-25   14,887.0    16,708.0    42     ₹76,482       +12.2%    
  FEDERALBNK  1      BANKING               03-Nov-25   237.9       266.2       2646   ₹75,041       +11.9%    
  CANBK       5      FINANCIAL_SERVICES    03-Nov-25   139.6       154.2       4510   ₹66,026       +10.5%    
  NYKAA       7      CONSUMER_DISCRETIONARY  01-Oct-25   241.3       265.8       2638   ₹64,605       +10.2%    
  AIAENG      8      METALS                01-Dec-25   3,854.1     4,030.1     166    ₹29,216       +4.6%     
  SBIN        14     BANKING               03-Nov-25   932.9       967.3       674    ₹23,205       +3.7%     
  BANKINDIA   18     FINANCIAL_SERVICES    03-Nov-25   142.1       147.0       4429   ₹21,525       +3.4%     
  CUMMINSIND  24     CAPITAL_GOODS         03-Nov-25   4,359.9     4,450.4     144    ₹13,030       +2.1%     
  RELIANCE    9      OIL_GAS               01-Dec-25   1,566.1     1,575.6     409    ₹3,886        +0.6%     
  GMRAIRPORT  17     INFRASTRUCTURE        01-Dec-25   107.6       105.5       5956   ₹-12,746      -2.0%     

  AFTER: Invested ₹12,272,162 | Cash ₹518,464 | Total ₹12,790,627 | Positions 18/20 | Slot ₹639,721

========================================================================
  REBALANCE #99  —  02 Feb 2026
  NAV: ₹12,203,664  |  Slot: ₹610,183  |  Cash: ₹518,464
========================================================================
  [SECTOR CAP≤4] dropped: SHRIRAMFIN, LTF, ABCAPITAL, MFSL

  [REGIME OFF] Nifty 200 22,837.0 < SMA200 23,150.1 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  MUTHOOTFIN  15     FINANCIAL_SERVICES    01-Aug-25   2,571.0     3,505.5     246    ₹229,887      +36.3%    
  FEDERALBNK  5      BANKING               03-Nov-25   237.9       281.3       2646   ₹114,863      +18.2%    
  INDIANB     29     BANKING               01-Oct-25   737.0       834.8       863    ₹84,358       +13.3%    
  EICHERMOT   38     AUTO                  01-Sep-25   6,280.0     6,985.5     100    ₹70,550       +11.2%    
  SBIN        6      BANKING               03-Nov-25   932.9       1,010.5     674    ₹52,303       +8.3%     
  BANKINDIA   28     FINANCIAL_SERVICES    03-Nov-25   142.1       151.7       4429   ₹42,253       +6.7%     
  CANBK       13     FINANCIAL_SERVICES    03-Nov-25   139.6       146.5       4510   ₹30,984       +4.9%     
  AIAENG      21     METALS                01-Dec-25   3,854.1     4,016.2     166    ₹26,909       +4.2%     
  INDUSTOWER  24     TELECOM               01-Jan-26   435.8       431.8       1467   ₹-5,868       -0.9%     
  NYKAA       57     CONSUMER_DISCRETIONARY  01-Oct-25   241.3       237.6       2638   ₹-9,602       -1.5%     
  TITAN       43     CONSUMER_DISCRETIONARY  01-Jan-26   4,049.3     3,953.2     157    ₹-15,088      -2.4%     
  AUBANK      9      FINANCIAL_SERVICES    01-Jan-26   999.5       965.2       640    ₹-21,920      -3.4%     
  MARUTI      125    AUTO                  01-Sep-25   14,887.0    14,384.0    42     ₹-21,126      -3.4% ⚠   
  IDFCFIRSTB  41     BANKING               01-Jan-26   85.6        81.2        7472   ₹-32,877      -5.1%     
  CUMMINSIND  51     CAPITAL_GOODS         03-Nov-25   4,359.9     4,073.8     144    ₹-41,198      -6.6%     
  RELIANCE    114    OIL_GAS               01-Dec-25   1,566.1     1,390.4     409    ₹-71,861      -11.2% ⚠  
  GMRAIRPORT  54     INFRASTRUCTURE        01-Dec-25   107.6       94.0        5956   ₹-81,180      -12.7%    
  UPL         72     CHEMICALS             01-Jan-26   805.3       698.5       794    ₹-84,799      -13.3%    
  ⚠  WAZ < 0 (momentum below universe mean): RELIANCE, MARUTI

  AFTER: Invested ₹11,685,199 | Cash ₹518,464 | Total ₹12,203,664 | Positions 18/20 | Slot ₹610,183

========================================================================
  REBALANCE #100  —  02 Mar 2026
  NAV: ₹12,715,080  |  Slot: ₹635,754  |  Cash: ₹518,464
========================================================================

  [REGIME OFF] Nifty 200 22,835.9 < SMA200 23,303.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  MUTHOOTFIN  81     FINANCIAL_SERVICES    01-Aug-25   2,571.0     3,442.8     246    ₹214,470      +33.9%    
  INDIANB     11     BANKING               01-Oct-25   737.0       976.7       863    ₹206,775      +32.5%    
  SBIN        1      BANKING               03-Nov-25   932.9       1,168.8     674    ₹159,028      +25.3%    
  EICHERMOT   13     AUTO                  01-Sep-25   6,280.0     7,826.0     100    ₹154,600      +24.6%    
  FEDERALBNK  8      BANKING               03-Nov-25   237.9       295.0       2646   ₹150,981      +24.0%    
  BANKINDIA   15     FINANCIAL_SERVICES    03-Nov-25   142.1       172.3       4429   ₹133,623      +21.2%    
  CUMMINSIND  21     CAPITAL_GOODS         03-Nov-25   4,359.9     4,816.8     144    ₹65,797       +10.5%    
  CANBK       22     FINANCIAL_SERVICES    03-Nov-25   139.6       153.6       4510   ₹62,960       +10.0%    
  NYKAA       57     CONSUMER_DISCRETIONARY  01-Oct-25   241.3       259.1       2638   ₹47,062       +7.4%     
  TITAN       30     CONSUMER_DISCRETIONARY  01-Jan-26   4,049.3     4,270.3     157    ₹34,697       +5.5%     
  INDUSTOWER  49     TELECOM               01-Jan-26   435.8       448.5       1467   ₹18,704       +2.9%     
  AIAENG      116    METALS                01-Dec-25   3,854.1     3,738.9     166    ₹-19,123      -3.0% ⚠   
  MARUTI      139    AUTO                  01-Sep-25   14,887.0    14,388.0    42     ₹-20,958      -3.4% ⚠   
  AUBANK      36     FINANCIAL_SERVICES    01-Jan-26   999.5       951.2       640    ₹-30,848      -4.8%     
  GMRAIRPORT  98     INFRASTRUCTURE        01-Dec-25   107.6       96.5        5956   ₹-66,409      -10.4%    
  RELIANCE    177    OIL_GAS               01-Dec-25   1,566.1     1,358.0     409    ₹-85,113      -13.3% ⚠  
  IDFCFIRSTB  134    BANKING               01-Jan-26   85.6        71.8        7472   ₹-103,338     -16.2% ⚠  
  UPL         188    CHEMICALS             01-Jan-26   805.3       622.8       794    ₹-144,905     -22.7% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): AIAENG, IDFCFIRSTB, MARUTI, RELIANCE, UPL

  AFTER: Invested ₹12,196,616 | Cash ₹518,464 | Total ₹12,715,080 | Positions 18/20 | Slot ₹635,754

========================================================================
  REBALANCE #101  —  01 Apr 2026
  NAV: ₹11,611,066  |  Slot: ₹580,553  |  Cash: ₹518,464
========================================================================
  [SECTOR CAP≤4] dropped: MAHABANK

  [REGIME OFF] Nifty 200 20,935.2 < SMA200 23,188.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  MUTHOOTFIN  65     FINANCIAL_SERVICES    01-Aug-25   2,571.0     3,228.7     246    ₹161,804      +25.6%    
  INDIANB     6      BANKING               01-Oct-25   737.0       888.0       863    ₹130,270      +20.5%    
  FEDERALBNK  26     BANKING               03-Nov-25   237.9       267.6       2646   ₹78,613       +12.5%    
  EICHERMOT   53     AUTO                  01-Sep-25   6,280.0     6,825.5     100    ₹54,550       +8.7%     
  SBIN        12     BANKING               03-Nov-25   932.9       999.8       674    ₹45,087       +7.2%     
  CUMMINSIND  8      CAPITAL_GOODS         03-Nov-25   4,359.9     4,609.1     144    ₹35,888       +5.7%     
  TITAN       21     CONSUMER_DISCRETIONARY  01-Jan-26   4,049.3     4,065.5     157    ₹2,543        +0.4%     
  BANKINDIA   38     FINANCIAL_SERVICES    03-Nov-25   142.1       141.7       4429   ₹-1,905       -0.3%     
  NYKAA       47     CONSUMER_DISCRETIONARY  01-Oct-25   241.3       240.0       2638   ₹-3,403       -0.5%     
  INDUSTOWER  51     TELECOM               01-Jan-26   435.8       423.2       1467   ₹-18,411      -2.9%     
  AIAENG      76     METALS                01-Dec-25   3,854.1     3,681.0     166    ₹-28,735      -4.5%     
  CANBK       64     FINANCIAL_SERVICES    03-Nov-25   139.6       127.3       4510   ₹-55,473      -8.8%     
  AUBANK      36     FINANCIAL_SERVICES    01-Jan-26   999.5       874.9       640    ₹-79,712      -12.5%    
  RELIANCE    115    OIL_GAS               01-Dec-25   1,566.1     1,369.2     409    ₹-80,532      -12.6% ⚠  
  MARUTI      185    AUTO                  01-Sep-25   14,887.0    12,509.0    42     ₹-99,876      -16.0% ⚠  
  GMRAIRPORT  91     INFRASTRUCTURE        01-Dec-25   107.6       89.3        5956   ₹-109,293     -17.0%    
  UPL         172    CHEMICALS             01-Jan-26   805.3       594.5       794    ₹-167,415     -26.2% ⚠  
  IDFCFIRSTB  161    BANKING               01-Jan-26   85.6        60.2        7472   ₹-190,013     -29.7% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): RELIANCE, IDFCFIRSTB, UPL, MARUTI

  AFTER: Invested ₹11,092,602 | Cash ₹518,464 | Total ₹11,611,066 | Positions 18/20 | Slot ₹580,553

========================================================================
  REBALANCE #102  —  01 May 2026
  NAV: ₹12,301,489  |  Slot: ₹615,074  |  Cash: ₹518,464
========================================================================
  [SECTOR CAP≤4] dropped: ABB, CGPOWER, LLOYDSME, BHEL, KEI

  [REGIME OFF] Nifty 200 22,683.6 < SMA200 23,107.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (18)
  Ticker      Rank   Sector                Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%      
  ──────────  ─────  ────────────────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────
  MUTHOOTFIN  68     FINANCIAL_SERVICES    01-Aug-25   2,571.0     3,424.2     246    ₹209,897      +33.2%    
  CUMMINSIND  3      CAPITAL_GOODS         03-Nov-25   4,359.9     5,266.4     144    ₹130,540      +20.8%    
  FEDERALBNK  52     BANKING               03-Nov-25   237.9       287.0       2646   ₹129,813      +20.6%    
  INDIANB     63     BANKING               01-Oct-25   737.0       851.8       863    ₹99,072       +15.6%    
  EICHERMOT   73     AUTO                  01-Sep-25   6,280.0     7,109.0     100    ₹82,900       +13.2%    
  SBIN        54     BANKING               03-Nov-25   932.9       1,049.5     674    ₹78,620       +12.5%    
  NYKAA       42     CONSUMER_DISCRETIONARY  01-Oct-25   241.3       264.8       2638   ₹61,993       +9.7%     
  TITAN       37     CONSUMER_DISCRETIONARY  01-Jan-26   4,049.3     4,385.2     157    ₹52,736       +8.3%     
  AIAENG      67     METALS                01-Dec-25   3,854.1     3,949.6     166    ₹15,853       +2.5%     
  AUBANK      45     FINANCIAL_SERVICES    01-Jan-26   999.5       1,016.0     640    ₹10,560       +1.7%     
  BANKINDIA   137    FINANCIAL_SERVICES    03-Nov-25   142.1       139.9       4429   ₹-10,010      -1.6% ⚠   
  CANBK       98     FINANCIAL_SERVICES    03-Nov-25   139.6       134.6       4510   ₹-22,325      -3.5%     
  INDUSTOWER  143    TELECOM               01-Jan-26   435.8       410.0       1467   ₹-37,922      -5.9% ⚠   
  RELIANCE    96     OIL_GAS               01-Dec-25   1,566.1     1,430.8     409    ₹-55,338      -8.6%     
  GMRAIRPORT  101    INFRASTRUCTURE        01-Dec-25   107.6       96.4        5956   ₹-66,767      -10.4%    
  MARUTI      158    AUTO                  01-Sep-25   14,887.0    13,314.0    42     ₹-66,066      -10.6% ⚠  
  IDFCFIRSTB  184    BANKING               01-Jan-26   85.6        69.6        7472   ₹-119,328     -18.7% ⚠  
  UPL         178    CHEMICALS             01-Jan-26   805.3       641.8       794    ₹-129,819     -20.3% ⚠  
  ⚠  WAZ < 0 (momentum below universe mean): BANKINDIA, INDUSTOWER, MARUTI, UPL, IDFCFIRSTB

  AFTER: Invested ₹11,783,024 | Cash ₹518,464 | Total ₹12,301,489 | Positions 18/20 | Slot ₹615,074

========================================================================
  FINAL SUMMARY
========================================================================
  Period        : 2017-12-01 → 2026-05-22  (8.5 years)
  Starting Cap  : ₹2,000,000
  Final Value   : ₹12,090,342
  Total Return  : +504.5%
  CAGR          : +23.7%

  Closed Trades : 399  |  Open: 18
  Win Rate      : 52.9%  (211W / 188L)
  Profit Factor : 3.28
  Avg hold      : 128 days
  Total charges : ₹409,345
  Closed net P&L: ₹9,823,025
  Open unreal   : ₹153,265

  YEAR-BY-YEAR:
  2017  +  0.0%  
  2018  -  8.3%  ░░░░░░░░
  2019  + 10.6%  ██████████
  2020  + 34.8%  ██████████████████████████████████
  2021  + 74.9%  ████████████████████████████████████████
  2022  + 24.6%  ████████████████████████
  2023  + 50.0%  ████████████████████████████████████████
  2024  + 42.7%  ████████████████████████████████████████
  2025  +  0.5%  
  2026  -  4.1%  ░░░░

  Negative years: 2

  Rebalance NAV exported → lm250_rebal.csv (102 rows)
