=== MQ500 — Nifty500 Momentum+Quality, Top30, Monthly Rebalance, β≤1.2 | Regime ON [SMA200] ===
    top_n=30 buffer_in=25 buffer_out=60 beta_cap=1.2
Loading PIT universe...
  886 unique PIT tickers across all periods
Loading EPS data...
  871 stocks with EPS data (quality factor, no binary gate)
  Sector map loaded: 34 PIT dates
Loading cached data from /Users/jay/dev/relative_strength/data/cache/mom500_daily.pkl...
Fetching Nifty 50 (beta)...
  3127 bars
Fetching Nifty 500 (regime filter)...
  3128 bars
  Trading days in backtest: 2840 (2015-01-01 → 2026-07-01)
  Rebalance dates: 139

==============================================================================================
  MOM15 PIT BACKTEST  |  NAV/15 slot  |  2-Month Rebalance  |  Beta≤1.0  |  Regime ON [SMA200]
==============================================================================================

========================================================================
  REBALANCE #01  —  01 Jan 2015
  NAV: ₹2,000,000  |  Slot: ₹66,667  |  Cash: ₹2,000,000
========================================================================

  EXITS (0)
    —

  ENTRIES (12)
  [52w filter blocked 8: IL&FSTRANS(-25.8%), OPTOCIRCUI(-43.2%), KSK(-45.0%), HDIL(-40.0%), SHRENUJ(-33.0%), GITANJALI(-50.3%), VIDEOIND(-23.5%), PUNJLLOYD(-37.8%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JBFIND      1      FIN SVC    3.444    0.21   +251.6%   +68.0%    256.3       260    ₹66,641       +9.0%     
  ALBK        2      PSU BNK    2.333    0.17   +119.4%   +46.4%    175.0       381    ₹66,661       +9.2%     
  ALKYLAMINE  3      MFG        1.781    0.06   +206.1%   -6.6%     131.9       505    ₹66,622       -0.3%     
  ORIENTBANK  4      PSU BNK    1.702    -0.33  +92.8%    +18.5%    193.2       345    ₹66,640       +0.9%     
  SYNDIBANK   5      PSU BNK    1.489    -0.19  +81.2%    +25.2%    71.2        935    ₹66,607       +4.3%     
  ANDHRABANK  6      PSU BNK    1.417    -0.28  +108.8%   +17.5%    191.1       348    ₹66,513       +5.8%     
  COLPAL      7      FMCG       1.354    0.19   +41.1%    +10.4%    721.1       92     ₹66,344       +0.5%     
  BASF        8      MFG        1.279    -0.04  +96.5%    +4.6%     1,254.6     53     ₹66,495       +4.4%     
  VSTIND      9      FMCG       1.259    0.22   +19.5%    +21.2%    122.7       543    ₹66,604       +1.9%     
  GSKCONS     10     FMCG       1.031    0.17   +34.8%    +5.5%     5,532.8     12     ₹66,394       +2.0%     
  RALLIS      12     FMCG       0.675    0.11   +30.7%    -0.6%     190.2       350    ₹66,559       +2.2%     
  TATAELXSI   13     IT         0.747    0.08   +70.7%    -5.4%     262.2       254    ₹66,608       +1.4%     

  HOLDS (0)
    —

  AFTER: Invested ₹798,688 | Cash ₹1,200,364 | Total ₹1,999,052 | Positions 12/30 | Slot ₹66,667

========================================================================
  REBALANCE #02  —  02 Feb 2015
  NAV: ₹1,974,442  |  Slot: ₹65,815  |  Cash: ₹1,200,364
========================================================================

  EXITS (0)
    —

  ENTRIES (2)
  [52w filter blocked 6: OPTOCIRCUI(-46.1%), KSK(-40.1%), SHRENUJ(-35.4%), GITANJALI(-56.7%), VIDEOIND(-25.3%), PUNJLLOYD(-39.2%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HDIL        3      REALTY     1.585    0.05   +122.6%   +35.5%    110.2       597    ₹65,789       +31.5%    
  IL&FSTRANS  5      INFRA      1.395    0.33   +89.9%    +19.2%    200.3       328    ₹65,712       +7.5%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   2      IT         01-Jan-15   262.2       329.8       254    ₹17,166       +25.8%      +4.1%     
  RALLIS      16     FMCG       01-Jan-15   190.2       194.5       350    ₹1,506        +2.3% ⚠     +0.2%     
  COLPAL      6      FMCG       01-Jan-15   721.1       734.4       92     ₹1,220        +1.8%       -2.7%     
  SYNDIBANK   7      PSU BNK    01-Jan-15   71.2        72.1        935    ₹815          +1.2%       -0.2%     
  JBFIND      1      FIN SVC    01-Jan-15   256.3       255.6       260    ₹-179         -0.3%       +4.5%     
  GSKCONS     11     FMCG       01-Jan-15   5,532.8     5,299.5     12     ₹-2,800       -4.2% ⚠     -1.2%     
  BASF        8      MFG        01-Jan-15   1,254.6     1,198.3     53     ₹-2,987       -4.5%       -3.1%     
  VSTIND      12     FMCG       01-Jan-15   122.7       116.9       543    ₹-3,130       -4.7% ⚠     -2.5%     
  ORIENTBANK  10     PSU BNK    01-Jan-15   193.2       168.7       345    ₹-8,449       -12.7% ⚠    -6.7%     
  ANDHRABANK  15     PSU BNK    01-Jan-15   191.1       166.0       348    ₹-8,743       -13.1% ⚠    -8.7%     
  ALKYLAMINE  4      MFG        01-Jan-15   131.9       114.4       505    ₹-8,841       -13.3%      -12.0%    
  ALBK        9      PSU BNK    01-Jan-15   175.0       148.2       381    ₹-10,187      -15.3%      -8.8%     
  ⚠  WAZ < 0 (momentum below universe mean): ORIENTBANK, GSKCONS, VSTIND, ANDHRABANK, RALLIS

  AFTER: Invested ₹905,579 | Cash ₹1,068,707 | Total ₹1,974,286 | Positions 14/30 | Slot ₹65,815

========================================================================
  REBALANCE #03  —  02 Mar 2015
  NAV: ₹1,992,354  |  Slot: ₹66,412  |  Cash: ₹1,068,707
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
  [52w filter blocked 6: OPTOCIRCUI(-47.6%), KSK(-45.0%), SHRENUJ(-37.4%), GITANJALI(-53.7%), VIDEOIND(-23.8%), PUNJLLOYD(-37.0%)]
    —

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   1      IT         01-Jan-15   262.2       503.4       254    ₹61,252       +92.0%      +22.2%    
  RALLIS      6      FMCG       01-Jan-15   190.2       218.6       350    ₹9,947        +14.9%      +5.7%     
  COLPAL      7      FMCG       01-Jan-15   721.1       788.7       92     ₹6,218        +9.4%       +3.1%     
  HDIL        2      REALTY     02-Feb-15   110.2       117.2       597    ₹4,179        +6.4%       +7.0%     
  IL&FSTRANS  5      INFRA      02-Feb-15   200.3       199.7       328    ₹-196         -0.3%       +3.7%     
  GSKCONS     10     FMCG       01-Jan-15   5,532.8     5,405.3     12     ₹-1,530       -2.3% ⚠     -0.0%     
  SYNDIBANK   8      PSU BNK    01-Jan-15   71.2        66.6        935    ₹-4,344       -6.5%       +1.0%     
  ALKYLAMINE  4      MFG        01-Jan-15   131.9       116.2       505    ₹-7,941       -11.9%      -1.2%     
  JBFIND      3      FIN SVC    01-Jan-15   256.3       225.6       260    ₹-7,979       -12.0%      -1.9%     
  BASF        11     MFG        01-Jan-15   1,254.6     1,092.1     53     ₹-8,611       -13.0% ⚠    -5.0%     
  VSTIND      15     FMCG       01-Jan-15   122.7       105.7       543    ₹-9,197       -13.8% ⚠    -6.4%     
  ALBK        9      PSU BNK    01-Jan-15   175.0       141.8       381    ₹-12,639      -19.0% ⚠    -4.7%     
  ORIENTBANK  14     PSU BNK    01-Jan-15   193.2       145.8       345    ₹-16,353      -24.5% ⚠    -3.3%     
  ANDHRABANK  12     PSU BNK    01-Jan-15   191.1       135.5       348    ₹-19,349      -29.1% ⚠    -7.7%     
  ⚠  WAZ < 0 (momentum below universe mean): ALBK, GSKCONS, BASF, ANDHRABANK, ORIENTBANK, VSTIND

  AFTER: Invested ₹923,647 | Cash ₹1,068,707 | Total ₹1,992,354 | Positions 14/30 | Slot ₹66,412

========================================================================
  REBALANCE #04  —  01 Apr 2015
  NAV: ₹1,972,496  |  Slot: ₹65,750  |  Cash: ₹1,068,707
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
  [52w filter blocked 6: SHRENUJ(-42.0%), KSK(-53.3%), OPTOCIRCUI(-55.1%), GITANJALI(-60.7%), VIDEOIND(-26.6%), PUNJLLOYD(-50.0%)]
    —

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   1      IT         01-Jan-15   262.2       552.9       254    ₹73,829       +110.8%     +10.4%    
  COLPAL      5      FMCG       01-Jan-15   721.1       799.7       92     ₹7,225        +10.9%      -0.5%     
  GSKCONS     6      FMCG       01-Jan-15   5,532.8     5,892.2     12     ₹4,313        +6.5%       +2.0%     
  RALLIS      8      FMCG       01-Jan-15   190.2       200.4       350    ₹3,590        +5.4%       +0.9%     
  HDIL        4      REALTY     02-Feb-15   110.2       103.8       597    ₹-3,851       -5.9%       -1.7%     
  IL&FSTRANS  7      INFRA      02-Feb-15   200.3       180.9       328    ₹-6,386       -9.7%       -1.1%     
  ALKYLAMINE  3      MFG        01-Jan-15   131.9       117.2       505    ₹-7,418       -11.1%      +6.5%     
  BASF        9      MFG        01-Jan-15   1,254.6     1,093.8     53     ₹-8,521       -12.8% ⚠    +1.0%     
  JBFIND      2      FIN SVC    01-Jan-15   256.3       221.1       260    ₹-9,155       -13.7%      +4.4%     
  SYNDIBANK   10     PSU BNK    01-Jan-15   71.2        60.4        935    ₹-10,119      -15.2% ⚠    -1.5%     
  VSTIND      14     FMCG       01-Jan-15   122.7       98.5        543    ₹-13,139      -19.7% ⚠    -3.2%     
  ALBK        11     PSU BNK    01-Jan-15   175.0       137.1       381    ₹-14,410      -21.6% ⚠    -0.6%     
  ORIENTBANK  15     PSU BNK    01-Jan-15   193.2       133.6       345    ₹-20,540      -30.8% ⚠    -4.0%     
  ANDHRABANK  12     PSU BNK    01-Jan-15   191.1       128.4       348    ₹-21,817      -32.8% ⚠    -3.0%     
  ⚠  WAZ < 0 (momentum below universe mean): BASF, SYNDIBANK, ALBK, ANDHRABANK, VSTIND, ORIENTBANK

  AFTER: Invested ₹903,789 | Cash ₹1,068,707 | Total ₹1,972,496 | Positions 14/30 | Slot ₹65,750

========================================================================
  REBALANCE #05  —  04 May 2015
  NAV: ₹1,960,617  |  Slot: ₹65,354  |  Cash: ₹1,068,707
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
  [52w filter blocked 6: SHRENUJ(-45.8%), OPTOCIRCUI(-55.9%), GITANJALI(-60.6%), KSK(-53.8%), VIDEOIND(-29.3%), PUNJLLOYD(-51.5%)]
    —

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   1      IT         01-Jan-15   262.2       499.0       254    ₹60,132       +90.3%      -1.1%     
  COLPAL      5      FMCG       01-Jan-15   721.1       815.2       92     ₹8,657        +13.0%      +0.2%     
  HDIL        4      REALTY     02-Feb-15   110.2       121.5       597    ₹6,746        +10.3%      +2.2%     
  GSKCONS     2      FMCG       01-Jan-15   5,532.8     6,024.6     12     ₹5,901        +8.9%       +1.5%     
  RALLIS      8      FMCG       01-Jan-15   190.2       191.6       350    ₹517          +0.8%       -1.2%     
  ALKYLAMINE  6      MFG        01-Jan-15   131.9       119.3       505    ₹-6,392       -9.6%       +2.6%     
  JBFIND      3      FIN SVC    01-Jan-15   256.3       230.2       260    ₹-6,777       -10.2%      +3.4%     
  BASF        7      MFG        01-Jan-15   1,254.6     1,116.2     53     ₹-7,338       -11.0%      +0.4%     
  SYNDIBANK   9      PSU BNK    01-Jan-15   71.2        61.5        935    ₹-9,089       -13.6% ⚠    +0.2%     
  VSTIND      12     FMCG       01-Jan-15   122.7       100.6       543    ₹-11,995      -18.0% ⚠    -3.5%     
  IL&FSTRANS  10     INFRA      02-Feb-15   200.3       158.3       328    ₹-13,796      -21.0% ⚠    -4.7%     
  ORIENTBANK  11     PSU BNK    01-Jan-15   193.2       146.6       345    ₹-16,080      -24.1% ⚠    +3.8%     
  ALBK        14     PSU BNK    01-Jan-15   175.0       115.1       381    ₹-22,811      -34.2% ⚠    -7.8%     
  ANDHRABANK  18     PSU BNK    01-Jan-15   191.1       116.5       348    ₹-25,954      -39.0% ⚠    -4.4%     
  ⚠  WAZ < 0 (momentum below universe mean): SYNDIBANK, IL&FSTRANS, ORIENTBANK, VSTIND, ALBK, ANDHRABANK

  AFTER: Invested ₹891,911 | Cash ₹1,068,707 | Total ₹1,960,617 | Positions 14/30 | Slot ₹65,354

========================================================================
  REBALANCE #06  —  01 Jun 2015
  NAV: ₹1,948,641  |  Slot: ₹64,955  |  Cash: ₹1,068,707
========================================================================

  EXITS (0)
    —

  ENTRIES (1)
  [52w filter blocked 7: GAEL(-47.6%), GITANJALI(-60.0%), OPTOCIRCUI(-54.4%), SHRENUJ(-53.3%), VIDEOIND(-29.5%), KSK(-62.9%), PUNJLLOYD(-59.1%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HDFC        5      PVT BNK    1.679    0.15   +28.7%    -1.5%     232.6       279    ₹64,892       +1.7%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   1      IT         01-Jan-15   262.2       549.6       254    ₹73,001       +109.6%     +5.2%     
  COLPAL      3      FMCG       01-Jan-15   721.1       797.4       92     ₹7,017        +10.6%      -1.3%     
  GSKCONS     2      FMCG       01-Jan-15   5,532.8     5,941.3     12     ₹4,902        +7.4%       -0.6%     
  RALLIS      8      FMCG       01-Jan-15   190.2       195.9       350    ₹1,993        +3.0%       +1.9%     
  HDIL        7      REALTY     02-Feb-15   110.2       108.8       597    ₹-836         -1.3%       -1.9%     
  VSTIND      11     FMCG       01-Jan-15   122.7       104.3       543    ₹-9,973       -15.0% ⚠    -0.9%     
  BASF        6      MFG        01-Jan-15   1,254.6     1,052.4     53     ₹-10,717      -16.1%      -0.1%     
  JBFIND      4      FIN SVC    01-Jan-15   256.3       214.7       260    ₹-10,817      -16.2%      -5.4%     
  SYNDIBANK   15     PSU BNK    01-Jan-15   71.2        54.3        935    ₹-15,790      -23.7% ⚠    -3.1%     
  ALBK        14     PSU BNK    01-Jan-15   175.0       131.7       381    ₹-16,468      -24.7% ⚠    +4.2%     
  ALKYLAMINE  10     MFG        01-Jan-15   131.9       97.9        505    ₹-17,169      -25.8% ⚠    -9.1%     
  IL&FSTRANS  16     INFRA      02-Feb-15   200.3       147.8       328    ₹-17,230      -26.2% ⚠    -1.8%     
  ANDHRABANK  9      PSU BNK    01-Jan-15   191.1       138.8       348    ₹-18,204      -27.4%      +8.1%     
  ORIENTBANK  13     PSU BNK    01-Jan-15   193.2       135.3       345    ₹-19,963      -30.0% ⚠    +0.7%     
  ⚠  WAZ < 0 (momentum below universe mean): ALKYLAMINE, VSTIND, ORIENTBANK, ALBK, SYNDIBANK, IL&FSTRANS

  AFTER: Invested ₹944,827 | Cash ₹1,003,737 | Total ₹1,948,564 | Positions 15/30 | Slot ₹64,955

========================================================================
  REBALANCE #07  —  01 Jul 2015
  NAV: ₹1,937,311  |  Slot: ₹64,577  |  Cash: ₹1,003,737
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
  [52w filter blocked 7: GAEL(-48.6%), OPTOCIRCUI(-50.1%), VIDEOIND(-23.4%), GITANJALI(-59.3%), SHRENUJ(-47.0%), KSK(-64.7%), PUNJLLOYD(-59.4%)]
    —

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   2      IT         01-Jan-15   262.2       527.9       254    ₹67,485       +101.3%     +2.9%     
  COLPAL      5      FMCG       01-Jan-15   721.1       823.8       92     ₹9,450        +14.2%      +3.7%     
  RALLIS      7      FMCG       01-Jan-15   190.2       211.1       350    ₹7,312        +11.0%      +3.6%     
  GSKCONS     6      FMCG       01-Jan-15   5,532.8     5,835.8     12     ₹3,636        +5.5%       +0.6%     
  HDFC        3      PVT BNK    01-Jun-15   232.6       240.7       279    ₹2,264        +3.5%       +3.2%     
  JBFIND      1      FIN SVC    01-Jan-15   256.3       249.4       260    ₹-1,790       -2.7%       +14.4%    
  BASF        4      MFG        01-Jan-15   1,254.6     1,163.1     53     ₹-4,848       -7.3%       +5.2%     
  HDIL        13     REALTY     02-Feb-15   110.2       93.9        597    ₹-9,701       -14.7% ⚠    -0.9%     
  VSTIND      8      FMCG       01-Jan-15   122.7       104.3       543    ₹-9,989       -15.0%      +1.0%     
  ALKYLAMINE  10     MFG        01-Jan-15   131.9       102.5       505    ₹-14,842      -22.3% ⚠    +1.6%     
  ANDHRABANK  12     PSU BNK    01-Jan-15   191.1       129.1       348    ₹-21,584      -32.5% ⚠    +4.5%     
  ALBK        16     PSU BNK    01-Jan-15   175.0       117.4       381    ₹-21,932      -32.9% ⚠    -1.2%     
  IL&FSTRANS  19     INFRA      02-Feb-15   200.3       134.4       328    ₹-21,628      -32.9% ⚠    -0.3%     
  SYNDIBANK   20     PSU BNK    01-Jan-15   71.2        47.2        935    ₹-22,496      -33.8% ⚠    -2.9%     
  ORIENTBANK  15     PSU BNK    01-Jan-15   193.2       127.0       345    ₹-22,842      -34.3% ⚠    +2.3%     
  ⚠  WAZ < 0 (momentum below universe mean): ALKYLAMINE, ANDHRABANK, HDIL, ORIENTBANK, ALBK, IL&FSTRANS, SYNDIBANK

  AFTER: Invested ₹933,574 | Cash ₹1,003,737 | Total ₹1,937,311 | Positions 15/30 | Slot ₹64,577

========================================================================
  REBALANCE #08  —  03 Aug 2015
  NAV: ₹2,024,504  |  Slot: ₹67,483  |  Cash: ₹1,003,737
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
  [52w filter blocked 7: GAEL(-29.9%), VIDEOIND(-22.2%), GITANJALI(-46.1%), SHRENUJ(-49.2%), OPTOCIRCUI(-44.2%), KSK(-57.5%), PUNJLLOYD(-41.8%)]
    —

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   1      IT         01-Jan-15   262.2       796.4       254    ₹135,679      +203.7%     +21.1%    
  JBFIND      3      FIN SVC    01-Jan-15   256.3       299.5       260    ₹11,226       +16.8%      +9.2%     
  COLPAL      9      FMCG       01-Jan-15   721.1       794.8       92     ₹6,776        +10.2%      -2.4%     
  GSKCONS     7      FMCG       01-Jan-15   5,532.8     5,957.0     12     ₹5,090        +7.7%       +0.5%     
  HDFC        2      PVT BNK    01-Jun-15   232.6       247.7       279    ₹4,223        +6.5%       -0.0%     
  RALLIS      10     FMCG       01-Jan-15   190.2       187.5       350    ₹-944         -1.4% ⚠     -8.7%     
  BASF        6      MFG        01-Jan-15   1,254.6     1,168.2     53     ₹-4,577       -6.9%       +2.7%     
  VSTIND      5      FMCG       01-Jan-15   122.7       110.9       543    ₹-6,364       -9.6%       +2.6%     
  ANDHRABANK  4      PSU BNK    01-Jan-15   191.1       153.9       348    ₹-12,946      -19.5%      +12.9%    
  HDIL        15     REALTY     02-Feb-15   110.2       86.4        597    ₹-14,179      -21.6% ⚠    -1.1%     
  ALKYLAMINE  13     MFG        01-Jan-15   131.9       102.1       505    ₹-15,052      -22.6% ⚠    +0.7%     
  ORIENTBANK  12     PSU BNK    01-Jan-15   193.2       139.8       345    ₹-18,410      -27.6% ⚠    +8.1%     
  ALBK        11     PSU BNK    01-Jan-15   175.0       118.3       381    ₹-21,590      -32.4% ⚠    +6.3%     
  IL&FSTRANS  17     INFRA      02-Feb-15   200.3       135.1       328    ₹-21,387      -32.5% ⚠    +0.7%     
  SYNDIBANK   19     PSU BNK    01-Jan-15   71.2        47.9        935    ₹-21,859      -32.8% ⚠    +3.6%     
  ⚠  WAZ < 0 (momentum below universe mean): RALLIS, ALBK, ORIENTBANK, ALKYLAMINE, HDIL, IL&FSTRANS, SYNDIBANK

  AFTER: Invested ₹1,020,767 | Cash ₹1,003,737 | Total ₹2,024,504 | Positions 15/30 | Slot ₹67,483

========================================================================
  REBALANCE #09  —  01 Sep 2015
  NAV: ₹1,935,911  |  Slot: ₹64,530  |  Cash: ₹1,003,737
========================================================================

  [REGIME OFF] Nifty 500 6,522.2 < SMA200 6,936.7 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   1      IT         01-Jan-15   262.2       799.0       254    ₹136,336      +204.7%     -0.4%     
  GSKCONS     7      FMCG       01-Jan-15   5,532.8     5,924.0     12     ₹4,695        +7.1%       +0.5%     
  COLPAL      3      FMCG       01-Jan-15   721.1       771.2       92     ₹4,607        +6.9%       -2.3%     
  HDFC        4      PVT BNK    01-Jun-15   232.6       226.2       279    ₹-1,790       -2.8%       -4.9%     
  RALLIS      11     FMCG       01-Jan-15   190.2       182.6       350    ₹-2,659       -4.0% ⚠     -5.1%     
  BASF        9      MFG        01-Jan-15   1,254.6     1,119.2     53     ₹-7,175       -10.8%      -6.2%     
  JBFIND      6      FIN SVC    01-Jan-15   256.3       223.1       260    ₹-8,642       -13.0%      -7.9%     
  ALKYLAMINE  8      MFG        01-Jan-15   131.9       109.3       505    ₹-11,418      -17.1%      +1.4%     
  VSTIND      12     FMCG       01-Jan-15   122.7       96.6        543    ₹-14,144      -21.2% ⚠    -4.8%     
  ANDHRABANK  10     PSU BNK    01-Jan-15   191.1       138.6       348    ₹-18,293      -27.5%      -9.4%     
  ORIENTBANK  13     PSU BNK    01-Jan-15   193.2       121.5       345    ₹-24,709      -37.1% ⚠    -10.0%    
  SYNDIBANK   15     PSU BNK    01-Jan-15   71.2        44.0        935    ₹-25,443      -38.2% ⚠    -9.5%     
  ALBK        14     PSU BNK    01-Jan-15   175.0       105.7       381    ₹-26,399      -39.6% ⚠    -6.3%     
  HDIL        19     REALTY     02-Feb-15   110.2       58.2        597    ₹-31,044      -47.2%      -18.3%    
  IL&FSTRANS  21     INFRA      02-Feb-15   200.3       88.1        328    ₹-36,830      -56.0% ⚠    -14.5%    
  ⚠  WAZ < 0 (momentum below universe mean): RALLIS, VSTIND, ORIENTBANK, ALBK, SYNDIBANK, IL&FSTRANS

  AFTER: Invested ₹932,174 | Cash ₹1,003,737 | Total ₹1,935,911 | Positions 15/30 | Slot ₹64,530

========================================================================
  REBALANCE #10  —  01 Oct 2015
  NAV: ₹1,961,156  |  Slot: ₹65,372  |  Cash: ₹1,003,737
========================================================================

  [REGIME OFF] Nifty 500 6,654.5 < SMA200 6,905.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   1      IT         01-Jan-15   262.2       831.4       254    ₹144,556      +217.0%     +0.2%     
  COLPAL      5      FMCG       01-Jan-15   721.1       770.9       92     ₹4,582        +6.9%       +0.4%     
  HDFC        3      PVT BNK    01-Jun-15   232.6       241.5       279    ₹2,472        +3.8%       +2.4%     
  GSKCONS     6      FMCG       01-Jan-15   5,532.8     5,730.2     12     ₹2,369        +3.6%       -0.9%     
  RALLIS      12     FMCG       01-Jan-15   190.2       188.5       350    ₹-576         -0.9% ⚠     -1.3%     
  JBFIND      9      FIN SVC    01-Jan-15   256.3       220.3       260    ₹-9,364       -14.1%      +0.1%     
  VSTIND      7      FMCG       01-Jan-15   122.7       104.5       543    ₹-9,850       -14.8%      +3.6%     
  ALKYLAMINE  8      MFG        01-Jan-15   131.9       110.3       505    ₹-10,925      -16.4%      +0.6%     
  BASF        13     MFG        01-Jan-15   1,254.6     1,014.9     53     ₹-12,707      -19.1% ⚠    -4.9%     
  ANDHRABANK  4      PSU BNK    01-Jan-15   191.1       143.6       348    ₹-16,539      -24.9%      -0.9%     
  HDIL        16     REALTY     02-Feb-15   110.2       72.3        597    ₹-22,626      -34.4%      +9.3%     
  SYNDIBANK   11     PSU BNK    01-Jan-15   71.2        45.8        935    ₹-23,747      -35.7% ⚠    +0.9%     
  ALBK        10     PSU BNK    01-Jan-15   175.0       109.2       381    ₹-25,061      -37.6% ⚠    +0.9%     
  ORIENTBANK  14     PSU BNK    01-Jan-15   193.2       119.4       345    ₹-25,440      -38.2% ⚠    -3.1%     
  IL&FSTRANS  20     INFRA      02-Feb-15   200.3       94.2        328    ₹-34,805      -53.0% ⚠    +1.1%     
  ⚠  WAZ < 0 (momentum below universe mean): ALBK, SYNDIBANK, RALLIS, BASF, ORIENTBANK, IL&FSTRANS

  AFTER: Invested ₹957,419 | Cash ₹1,003,737 | Total ₹1,961,156 | Positions 15/30 | Slot ₹65,372

========================================================================
  REBALANCE #11  —  02 Nov 2015
  NAV: ₹1,933,857  |  Slot: ₹64,462  |  Cash: ₹1,003,737
========================================================================
  ⚠ Insufficient scored stocks — skipping rebalance

========================================================================
  REBALANCE #12  —  01 Dec 2015
  NAV: ₹1,963,560  |  Slot: ₹65,452  |  Cash: ₹1,003,737
========================================================================
  ⚠ Insufficient scored stocks — skipping rebalance

========================================================================
  REBALANCE #13  —  01 Jan 2016
  NAV: ₹1,991,389  |  Slot: ₹66,380  |  Cash: ₹1,003,737
========================================================================

  [REGIME OFF] Nifty 500 6,753.6 < SMA200 6,817.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   1      IT         01-Jan-15   262.2       995.0       254    ₹186,118      +279.4%     +4.3%     
  GSKCONS     4      FMCG       01-Jan-15   5,532.8     6,098.2     12     ₹6,784        +10.2%      +0.2%     
  COLPAL      7      FMCG       01-Jan-15   721.1       788.9       92     ₹6,233        +9.4%       -0.2%     
  HDFC        3      PVT BNK    01-Jun-15   232.6       246.2       279    ₹3,803        +5.9%       +1.5%     
  ALKYLAMINE  5      MFG        01-Jan-15   131.9       132.6       505    ₹351          +0.5%       +4.3%     
  JBFIND      11     FIN SVC    01-Jan-15   256.3       235.6       260    ₹-5,389       -8.1%       +3.2%     
  VSTIND      9      FMCG       01-Jan-15   122.7       112.4       543    ₹-5,550       -8.3%       +2.7%     
  RALLIS      15     FMCG       01-Jan-15   190.2       157.8       350    ₹-11,328      -17.0% ⚠    +1.3%     
  BASF        13     MFG        01-Jan-15   1,254.6     903.1       53     ₹-18,632      -28.0% ⚠    +0.6%     
  HDIL        8      REALTY     02-Feb-15   110.2       78.3        597    ₹-19,014      -28.9%      +10.2%    
  ANDHRABANK  14     PSU BNK    01-Jan-15   191.1       123.9       348    ₹-23,395      -35.2% ⚠    -2.0%     
  ORIENTBANK  17     PSU BNK    01-Jan-15   193.2       106.0       345    ₹-30,059      -45.1% ⚠    -4.7%     
  SYNDIBANK   18     PSU BNK    01-Jan-15   71.2        39.1        935    ₹-30,026      -45.1% ⚠    -2.6%     
  ALBK        16     PSU BNK    01-Jan-15   175.0       93.8        381    ₹-30,912      -46.4% ⚠    -2.1%     
  IL&FSTRANS  12     INFRA      02-Feb-15   200.3       89.3        328    ₹-36,413      -55.4% ⚠    +7.6%     
  ⚠  WAZ < 0 (momentum below universe mean): IL&FSTRANS, BASF, ANDHRABANK, RALLIS, ALBK, ORIENTBANK, SYNDIBANK

  AFTER: Invested ₹987,652 | Cash ₹1,003,737 | Total ₹1,991,389 | Positions 15/30 | Slot ₹66,380

========================================================================
  REBALANCE #14  —  01 Feb 2016
  NAV: ₹1,911,974  |  Slot: ₹63,732  |  Cash: ₹1,003,737
========================================================================

  [REGIME OFF] Nifty 500 6,341.6 < SMA200 6,745.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   1      IT         01-Jan-15   262.2       994.8       254    ₹186,073      +279.4%     +14.3%    
  HDFC        6      PVT BNK    01-Jun-15   232.6       239.7       279    ₹1,974        +3.0%       +1.2%     
  GSKCONS     4      FMCG       01-Jan-15   5,532.8     5,615.8     12     ₹996          +1.5%       -1.0%     
  COLPAL      11     FMCG       01-Jan-15   721.1       696.0       92     ₹-2,316       -3.5% ⚠     -4.5%     
  VSTIND      2      FMCG       01-Jan-15   122.7       111.5       543    ₹-6,073       -9.1%       +0.8%     
  ALKYLAMINE  5      MFG        01-Jan-15   131.9       114.7       505    ₹-8,707       -13.1%      -4.2%     
  JBFIND      8      FIN SVC    01-Jan-15   256.3       219.7       260    ₹-9,519       -14.3%      +0.5%     
  RALLIS      12     FMCG       01-Jan-15   190.2       143.4       350    ₹-16,367      -24.6% ⚠    -0.2%     
  HDIL        7      REALTY     02-Feb-15   110.2       72.2        597    ₹-22,716      -34.5%      +0.2%     
  BASF        13     MFG        01-Jan-15   1,254.6     774.9       53     ₹-25,424      -38.2% ⚠    -5.3%     
  ANDHRABANK  15     PSU BNK    01-Jan-15   191.1       103.8       348    ₹-30,395      -45.7% ⚠    -1.9%     
  SYNDIBANK   20     PSU BNK    01-Jan-15   71.2        30.8        935    ₹-37,824      -56.8% ⚠    -5.5%     
  ALBK        18     PSU BNK    01-Jan-15   175.0       75.0        381    ₹-38,087      -57.1% ⚠    -5.8%     
  ORIENTBANK  19     PSU BNK    01-Jan-15   193.2       81.2        345    ₹-38,629      -58.0% ⚠    -8.0%     
  IL&FSTRANS  14     INFRA      02-Feb-15   200.3       78.9        328    ₹-39,831      -60.6% ⚠    +0.9%     
  ⚠  WAZ < 0 (momentum below universe mean): COLPAL, RALLIS, BASF, IL&FSTRANS, ANDHRABANK, ALBK, ORIENTBANK, SYNDIBANK

  AFTER: Invested ₹908,237 | Cash ₹1,003,737 | Total ₹1,911,974 | Positions 15/30 | Slot ₹63,732

========================================================================
  REBALANCE #15  —  01 Mar 2016
  NAV: ₹1,796,890  |  Slot: ₹59,896  |  Cash: ₹1,003,737
========================================================================

  [REGIME OFF] Nifty 500 6,020.0 < SMA200 6,653.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   1      IT         01-Jan-15   262.2       792.5       254    ₹134,691      +202.2%     -4.0%     
  HDFC        8      PVT BNK    01-Jun-15   232.6       222.6       279    ₹-2,796       -4.3%       -0.5%     
  GSKCONS     5      FMCG       01-Jan-15   5,532.8     5,185.5     12     ₹-4,168       -6.3%       -3.9%     
  COLPAL      11     FMCG       01-Jan-15   721.1       665.7       92     ₹-5,098       -7.7% ⚠     -1.8%     
  VSTIND      3      FMCG       01-Jan-15   122.7       104.0       543    ₹-10,118      -15.2%      -1.4%     
  ALKYLAMINE  2      MFG        01-Jan-15   131.9       105.9       505    ₹-13,126      -19.7%      -3.2%     
  JBFIND      10     FIN SVC    01-Jan-15   256.3       175.5       260    ₹-21,019      -31.5%      -1.3%     
  RALLIS      12     FMCG       01-Jan-15   190.2       129.7       350    ₹-21,161      -31.8% ⚠    -2.3%     
  BASF        9      MFG        01-Jan-15   1,254.6     762.7       53     ₹-26,070      -39.2%      +1.1%     
  HDIL        6      REALTY     02-Feb-15   110.2       64.9        597    ₹-27,044      -41.1%      +2.2%     
  ANDHRABANK  13     PSU BNK    01-Jan-15   191.1       93.0        348    ₹-34,160      -51.4% ⚠    -3.4%     
  SYNDIBANK   19     PSU BNK    01-Jan-15   71.2        27.1        935    ₹-41,278      -62.0% ⚠    -4.4%     
  ALBK        21     PSU BNK    01-Jan-15   175.0       62.5        381    ₹-42,834      -64.3% ⚠    -8.6%     
  ORIENTBANK  20     PSU BNK    01-Jan-15   193.2       66.4        345    ₹-43,730      -65.6% ⚠    -5.8%     
  IL&FSTRANS  16     INFRA      02-Feb-15   200.3       66.1        328    ₹-44,017      -67.0% ⚠    -2.3%     
  ⚠  WAZ < 0 (momentum below universe mean): COLPAL, RALLIS, ANDHRABANK, IL&FSTRANS, SYNDIBANK, ORIENTBANK, ALBK

  AFTER: Invested ₹793,153 | Cash ₹1,003,737 | Total ₹1,796,890 | Positions 15/30 | Slot ₹59,896

========================================================================
  REBALANCE #16  —  01 Apr 2016
  NAV: ₹1,875,237  |  Slot: ₹62,508  |  Cash: ₹1,003,737
========================================================================

  [REGIME OFF] Nifty 500 6,445.5 < SMA200 6,601.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   1      IT         01-Jan-15   262.2       841.9       254    ₹147,245      +221.1%     +0.2%     
  GSKCONS     6      FMCG       01-Jan-15   5,532.8     5,730.9     12     ₹2,377        +3.6%       +2.3%     
  HDFC        2      PVT BNK    01-Jun-15   232.6       240.7       279    ₹2,270        +3.5%       +2.6%     
  COLPAL      16     FMCG       01-Jan-15   721.1       673.5       92     ₹-4,381       -6.6% ⚠     -0.1%     
  ALKYLAMINE  8      MFG        01-Jan-15   131.9       115.8       505    ₹-8,136       -12.2%      +0.5%     
  VSTIND      4      FMCG       01-Jan-15   122.7       106.1       543    ₹-8,982       -13.5%      -0.6%     
  RALLIS      9      FMCG       01-Jan-15   190.2       148.0       350    ₹-14,759      -22.2%      +2.4%     
  JBFIND      12     FIN SVC    01-Jan-15   256.3       182.8       260    ₹-19,108      -28.7% ⚠    +1.2%     
  HDIL        3      REALTY     02-Feb-15   110.2       77.3        597    ₹-19,611      -29.8%      +10.3%    
  BASF        10     MFG        01-Jan-15   1,254.6     838.4       53     ₹-22,058      -33.2%      +4.1%     
  ANDHRABANK  7      PSU BNK    01-Jan-15   191.1       115.1       348    ₹-26,442      -39.8%      +9.9%     
  ALBK        11     PSU BNK    01-Jan-15   175.0       85.5        381    ₹-34,072      -51.1% ⚠    +4.5%     
  SYNDIBANK   14     PSU BNK    01-Jan-15   71.2        32.6        935    ₹-36,112      -54.2% ⚠    +7.0%     
  ORIENTBANK  17     PSU BNK    01-Jan-15   193.2       79.0        345    ₹-39,375      -59.1% ⚠    +5.3%     
  IL&FSTRANS  15     INFRA      02-Feb-15   200.3       71.0        328    ₹-42,435      -64.6% ⚠    +2.2%     
  ⚠  WAZ < 0 (momentum below universe mean): ALBK, JBFIND, SYNDIBANK, IL&FSTRANS, COLPAL, ORIENTBANK

  AFTER: Invested ₹871,500 | Cash ₹1,003,737 | Total ₹1,875,237 | Positions 15/30 | Slot ₹62,508

========================================================================
  REBALANCE #17  —  02 May 2016
  NAV: ₹1,898,896  |  Slot: ₹63,297  |  Cash: ₹1,003,737
========================================================================

  EXITS (0)
    —

  ENTRIES (1)
  [52w filter blocked 4: BRFL(-33.0%), KSK(-45.2%), VIDEOIND(-34.3%), PUNJLLOYD(-35.4%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  GAEL        2      FMCG       2.482    0.73   +31.3%    +25.9%    12.9        4913   ₹63,291       +10.4%    

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   3      IT         01-Jan-15   262.2       842.1       254    ₹147,295      +221.1%     -1.4%     
  HDFC        1      PVT BNK    01-Jun-15   232.6       252.8       279    ₹5,627        +8.7%       +2.6%     
  GSKCONS     11     FMCG       01-Jan-15   5,532.8     5,514.0     12     ₹-225         -0.3% ⚠     -2.8%     
  COLPAL      13     FMCG       01-Jan-15   721.1       688.8       92     ₹-2,971       -4.5% ⚠     +1.1%     
  VSTIND      6      FMCG       01-Jan-15   122.7       112.1       543    ₹-5,736       -8.6%       +2.8%     
  RALLIS      4      FMCG       01-Jan-15   190.2       171.5       350    ₹-6,534       -9.8%       +5.0%     
  ALKYLAMINE  9      MFG        01-Jan-15   131.9       115.3       505    ₹-8,372       -12.6% ⚠    -1.6%     
  JBFIND      10     FIN SVC    01-Jan-15   256.3       221.5       260    ₹-9,042       -13.6%      +8.1%     
  HDIL        7      REALTY     02-Feb-15   110.2       83.4        597    ₹-16,000      -24.3%      +4.7%     
  BASF        5      MFG        01-Jan-15   1,254.6     887.8       53     ₹-19,442      -29.2%      +2.9%     
  ANDHRABANK  8      PSU BNK    01-Jan-15   191.1       104.2       348    ₹-30,265      -45.5%      -2.2%     
  SYNDIBANK   12     PSU BNK    01-Jan-15   71.2        33.0        935    ₹-35,744      -53.7% ⚠    +0.8%     
  ALBK        14     PSU BNK    01-Jan-15   175.0       78.6        381    ₹-36,733      -55.1% ⚠    -3.2%     
  ORIENTBANK  16     PSU BNK    01-Jan-15   193.2       75.9        345    ₹-40,464      -60.7% ⚠    -2.0%     
  IL&FSTRANS  15     INFRA      02-Feb-15   200.3       74.4        328    ₹-41,317      -62.9% ⚠    +0.0%     
  ⚠  WAZ < 0 (momentum below universe mean): ALKYLAMINE, GSKCONS, SYNDIBANK, COLPAL, ALBK, IL&FSTRANS, ORIENTBANK

  AFTER: Invested ₹958,449 | Cash ₹940,371 | Total ₹1,898,821 | Positions 16/30 | Slot ₹63,297

========================================================================
  REBALANCE #18  —  01 Jun 2016
  NAV: ₹1,889,785  |  Slot: ₹62,993  |  Cash: ₹940,371
========================================================================

  EXITS (0)
    —

  ENTRIES (1)
  [52w filter blocked 5: NITINFIRE(-41.6%), BRFL(-35.7%), KSK(-46.6%), VIDEOIND(-35.8%), PUNJLLOYD(-43.5%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RELCAPITAL  5      FIN SVC    1.539    0.96   +5.9%     +24.8%    332.1       189    ₹62,761       +0.6%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   7      IT         01-Jan-15   262.2       788.4       254    ₹133,641      +200.6%     -3.6%     
  HDFC        1      PVT BNK    01-Jun-15   232.6       265.8       279    ₹9,255        +14.3%      +1.9%     
  GSKCONS     12     FMCG       01-Jan-15   5,532.8     5,496.4     12     ₹-437         -0.7% ⚠     -1.3%     
  GAEL        2      FMCG       02-May-16   12.9        12.6        4913   ₹-1,180       -1.9%       -1.0%     
  RALLIS      3      FMCG       01-Jan-15   190.2       184.8       350    ₹-1,863       -2.8%       +3.1%     
  COLPAL      13     FMCG       01-Jan-15   721.1       688.4       92     ₹-3,012       -4.5% ⚠     +1.3%     
  VSTIND      9      FMCG       01-Jan-15   122.7       108.9       543    ₹-7,467       -11.2%      -1.4%     
  HDIL        4      REALTY     02-Feb-15   110.2       97.8        597    ₹-7,433       -11.3%      +6.3%     
  ALKYLAMINE  8      MFG        01-Jan-15   131.9       115.7       505    ₹-8,183       -12.3%      -0.6%     
  JBFIND      11     FIN SVC    01-Jan-15   256.3       207.7       260    ₹-12,643      -19.0%      +0.1%     
  BASF        6      MFG        01-Jan-15   1,254.6     974.3       53     ₹-14,858      -22.3%      +3.9%     
  ANDHRABANK  15     PSU BNK    01-Jan-15   191.1       94.2        348    ₹-33,743      -50.7% ⚠    -0.6%     
  SYNDIBANK   14     PSU BNK    01-Jan-15   71.2        32.4        935    ₹-36,289      -54.5% ⚠    +2.9%     
  ALBK        17     PSU BNK    01-Jan-15   175.0       74.0        381    ₹-38,460      -57.7% ⚠    -1.4%     
  ORIENTBANK  18     PSU BNK    01-Jan-15   193.2       69.2        345    ₹-42,781      -64.2% ⚠    -0.9%     
  IL&FSTRANS  19     INFRA      02-Feb-15   200.3       67.7        328    ₹-43,506      -66.2% ⚠    -2.6%     
  ⚠  WAZ < 0 (momentum below universe mean): GSKCONS, COLPAL, SYNDIBANK, ANDHRABANK, ALBK, ORIENTBANK, IL&FSTRANS

  AFTER: Invested ₹1,012,175 | Cash ₹877,536 | Total ₹1,889,711 | Positions 17/30 | Slot ₹62,993

========================================================================
  REBALANCE #19  —  01 Jul 2016
  NAV: ₹1,973,846  |  Slot: ₹65,795  |  Cash: ₹877,536
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
  [52w filter blocked 5: NITINFIRE(-43.9%), BRFL(-27.1%), KSK(-45.6%), VIDEOIND(-35.4%), PUNJLLOYD(-40.7%)]
    —

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   11     IT         01-Jan-15   262.2       773.6       254    ₹129,885      +195.0%     -0.5%     
  GAEL        1      FMCG       02-May-16   12.9        15.1        4913   ₹11,030       +17.4%      +5.9%     
  HDFC        2      PVT BNK    01-Jun-15   232.6       267.8       279    ₹9,822        +15.1%      +1.2%     
  ALKYLAMINE  3      MFG        01-Jan-15   131.9       139.8       505    ₹3,960        +5.9%       +8.3%     
  GSKCONS     13     FMCG       01-Jan-15   5,532.8     5,764.7     12     ₹2,783        +4.2% ⚠     +3.7%     
  COLPAL      9      FMCG       01-Jan-15   721.1       745.8       92     ₹2,268        +3.4% ⚠     +4.3%     
  RALLIS      6      FMCG       01-Jan-15   190.2       195.0       350    ₹1,687        +2.5%       +2.6%     
  RELCAPITAL  8      FIN SVC    01-Jun-16   332.1       328.5       189    ₹-678         -1.1%       +0.7%     
  HDIL        4      REALTY     02-Feb-15   110.2       103.4       597    ₹-4,060       -6.2%       +4.3%     
  VSTIND      10     FMCG       01-Jan-15   122.7       112.3       543    ₹-5,600       -8.4% ⚠     +0.9%     
  BASF        5      MFG        01-Jan-15   1,254.6     1,108.4     53     ₹-7,751       -11.7%      +9.3%     
  JBFIND      14     FIN SVC    01-Jan-15   256.3       213.7       260    ₹-11,081      -16.6%      +3.0%     
  ALBK        7      PSU BNK    01-Jan-15   175.0       116.9       381    ₹-22,108      -33.2%      +12.5%    
  ANDHRABANK  16     PSU BNK    01-Jan-15   191.1       110.5       348    ₹-28,048      -42.2% ⚠    +6.5%     
  SYNDIBANK   17     PSU BNK    01-Jan-15   71.2        35.8        935    ₹-33,088      -49.7% ⚠    +5.0%     
  ORIENTBANK  12     PSU BNK    01-Jan-15   193.2       94.9        345    ₹-33,901      -50.9% ⚠    +10.2%    
  IL&FSTRANS  19     INFRA      02-Feb-15   200.3       78.6        328    ₹-39,943      -60.8% ⚠    +9.0%     
  ⚠  WAZ < 0 (momentum below universe mean): COLPAL, VSTIND, ORIENTBANK, GSKCONS, ANDHRABANK, SYNDIBANK, IL&FSTRANS

  AFTER: Invested ₹1,096,311 | Cash ₹877,536 | Total ₹1,973,846 | Positions 17/30 | Slot ₹65,795

========================================================================
  REBALANCE #20  —  01 Aug 2016
  NAV: ₹2,005,637  |  Slot: ₹66,855  |  Cash: ₹877,536
========================================================================

  EXITS (0)
    —

  ENTRIES (1)
  [52w filter blocked 3: KSK(-46.9%), VIDEOIND(-35.4%), PUNJLLOYD(-43.8%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BRFL        5      CONSUMP    1.927    0.27   +45.1%    +26.9%    189.9       352    ₹66,827       +7.8%     

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   15     IT         01-Jan-15   262.2       733.6       254    ₹119,717      +179.7% ⚠   -1.5%     
  GAEL        1      FMCG       02-May-16   12.9        19.0        4913   ₹30,082       +47.5%      +14.0%    
  HDFC        4      PVT BNK    01-Jun-15   232.6       283.3       279    ₹14,135       +21.8%      +1.9%     
  VSTIND      3      FMCG       01-Jan-15   122.7       136.9       543    ₹7,717        +11.6%      +10.1%    
  GSKCONS     7      FMCG       01-Jan-15   5,532.8     6,164.8     12     ₹7,584        +11.4%      +3.0%     
  RELCAPITAL  8      FIN SVC    01-Jun-16   332.1       368.6       189    ₹6,898        +11.0%      +6.6%     
  COLPAL      11     FMCG       01-Jan-15   721.1       771.4       92     ₹4,624        +7.0% ⚠     +1.7%     
  RALLIS      13     FMCG       01-Jan-15   190.2       195.4       350    ₹1,827        +2.7% ⚠     +2.6%     
  HDIL        9      REALTY     02-Feb-15   110.2       100.8       597    ₹-5,642       -8.6%       -0.9%     
  ALKYLAMINE  10     MFG        01-Jan-15   131.9       119.5       505    ₹-6,261       -9.4%       -9.9%     
  BASF        14     MFG        01-Jan-15   1,254.6     1,020.5     53     ₹-12,407      -18.7% ⚠    -3.7%     
  JBFIND      21     FIN SVC    01-Jan-15   256.3       204.9       260    ₹-13,365      -20.1% ⚠    -0.2%     
  ALBK        2      PSU BNK    01-Jan-15   175.0       128.3       381    ₹-17,781      -26.7%      +3.5%     
  ORIENTBANK  6      PSU BNK    01-Jan-15   193.2       110.2       345    ₹-28,612      -42.9%      -0.1%     
  SYNDIBANK   12     PSU BNK    01-Jan-15   71.2        40.0        935    ₹-29,220      -43.9% ⚠    -0.3%     
  ANDHRABANK  16     PSU BNK    01-Jan-15   191.1       104.9       348    ₹-30,019      -45.1% ⚠    -4.2%     
  IL&FSTRANS  18     INFRA      02-Feb-15   200.3       71.3        328    ₹-42,309      -64.4% ⚠    -6.3%     
  ⚠  WAZ < 0 (momentum below universe mean): COLPAL, RALLIS, SYNDIBANK, BASF, TATAELXSI, ANDHRABANK, IL&FSTRANS, JBFIND

  AFTER: Invested ₹1,194,928 | Cash ₹810,629 | Total ₹2,005,557 | Positions 18/30 | Slot ₹66,855

========================================================================
  REBALANCE #21  —  01 Sep 2016
  NAV: ₹2,048,461  |  Slot: ₹68,282  |  Cash: ₹810,629
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
  [52w filter blocked 4: KSK(-50.1%), NITINFIRE(-51.4%), VIDEOIND(-29.1%), PUNJLLOYD(-23.4%)]
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   19     IT         01-Jan-15   262.2       696.5       254    ₹110,308      +165.6% ⚠   -3.4%     
  GAEL        3      FMCG       02-May-16   12.9        21.2        4913   ₹40,818       +64.5%      +7.4%     
  RELCAPITAL  4      FIN SVC    01-Jun-16   332.1       433.5       189    ₹19,167       +30.5%      +9.1%     
  HDFC        5      PVT BNK    01-Jun-15   232.6       292.7       279    ₹16,769       +25.8%      +2.5%     
  VSTIND      2      FMCG       01-Jan-15   122.7       154.1       543    ₹17,099       +25.7%      +6.8%     
  GSKCONS     7      FMCG       01-Jan-15   5,532.8     6,005.5     12     ₹5,673        +8.5% ⚠     -0.7%     
  COLPAL      10     FMCG       01-Jan-15   721.1       770.6       92     ₹4,550        +6.9% ⚠     -0.0%     
  ALBK        1      PSU BNK    01-Jan-15   175.0       184.7       381    ₹3,716        +5.6%       +5.5%     
  RALLIS      9      FMCG       01-Jan-15   190.2       199.8       350    ₹3,361        +5.0% ⚠     +1.4%     
  ALKYLAMINE  12     MFG        01-Jan-15   131.9       118.1       505    ₹-6,996       -10.5% ⚠    -2.3%     
  BASF        13     MFG        01-Jan-15   1,254.6     1,094.1     53     ₹-8,507       -12.8% ⚠    +1.2%     
  BRFL        11     CONSUMP    01-Aug-16   189.9       163.5       352    ₹-9,275       -13.9%      -4.0%     
  HDIL        14     REALTY     02-Feb-15   110.2       89.5        597    ₹-12,358      -18.8% ⚠    -6.2%     
  JBFIND      20     FIN SVC    01-Jan-15   256.3       200.9       260    ₹-14,398      -21.6% ⚠    -1.5%     
  SYNDIBANK   8      PSU BNK    01-Jan-15   71.2        45.6        935    ₹-24,016      -36.1%      +4.3%     
  ANDHRABANK  15     PSU BNK    01-Jan-15   191.1       115.4       348    ₹-26,369      -39.6% ⚠    +2.3%     
  ORIENTBANK  6      PSU BNK    01-Jan-15   193.2       113.4       345    ₹-27,524      -41.3%      +0.8%     
  IL&FSTRANS  16     INFRA      02-Feb-15   200.3       71.8        328    ₹-42,145      -64.1% ⚠    +0.8%     
  ⚠  WAZ < 0 (momentum below universe mean): GSKCONS, RALLIS, COLPAL, ALKYLAMINE, BASF, HDIL, ANDHRABANK, IL&FSTRANS, TATAELXSI, JBFIND

  AFTER: Invested ₹1,237,831 | Cash ₹810,629 | Total ₹2,048,461 | Positions 18/30 | Slot ₹68,282

========================================================================
  REBALANCE #22  —  03 Oct 2016
  NAV: ₹2,071,318  |  Slot: ₹69,044  |  Cash: ₹810,629
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
  [52w filter blocked 3: NITINFIRE(-49.4%), VIDEOIND(-28.1%), PUNJLLOYD(-26.9%)]
    —

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   19     IT         01-Jan-15   262.2       624.7       254    ₹92,061       +138.2% ⚠   -6.9%     
  GAEL        5      FMCG       02-May-16   12.9        19.5        4913   ₹32,560       +51.4%      -1.1%     
  RELCAPITAL  2      FIN SVC    01-Jun-16   332.1       480.2       189    ₹27,993       +44.6%      +4.4%     
  VSTIND      1      FMCG       01-Jan-15   122.7       156.9       543    ₹18,580       +27.9%      +1.9%     
  HDFC        4      PVT BNK    01-Jun-15   232.6       293.5       279    ₹16,991       +26.2%      +0.2%     
  COLPAL      10     FMCG       01-Jan-15   721.1       796.7       92     ₹6,951        +10.5% ⚠    +1.2%     
  RALLIS      11     FMCG       01-Jan-15   190.2       209.6       350    ₹6,786        +10.2% ⚠    +5.8%     
  GSKCONS     12     FMCG       01-Jan-15   5,532.8     5,958.3     12     ₹5,106        +7.7% ⚠     +0.7%     
  ALBK        3      PSU BNK    01-Jan-15   175.0       184.9       381    ₹3,779        +5.7%       +3.4%     
  BASF        9      MFG        01-Jan-15   1,254.6     1,214.7     53     ₹-2,115       -3.2% ⚠     +6.4%     
  ALKYLAMINE  15     MFG        01-Jan-15   131.9       118.1       505    ₹-7,006       -10.5% ⚠    -1.7%     
  JBFIND      18     FIN SVC    01-Jan-15   256.3       227.2       260    ₹-7,560       -11.3% ⚠    +4.2%     
  BRFL        13     CONSUMP    01-Aug-16   189.9       164.6       352    ₹-8,906       -13.3% ⚠    +1.4%     
  HDIL        16     REALTY     02-Feb-15   110.2       82.9        597    ₹-16,268      -24.7% ⚠    -4.0%     
  SYNDIBANK   6      PSU BNK    01-Jan-15   71.2        52.4        935    ₹-17,591      -26.4%      +6.3%     
  ORIENTBANK  8      PSU BNK    01-Jan-15   193.2       129.2       345    ₹-22,049      -33.1%      +4.0%     
  ANDHRABANK  14     PSU BNK    01-Jan-15   191.1       120.1       348    ₹-24,705      -37.1% ⚠    +1.4%     
  IL&FSTRANS  7      INFRA      02-Feb-15   200.3       103.2       328    ₹-31,879      -48.5%      +9.1%     
  ⚠  WAZ < 0 (momentum below universe mean): BASF, COLPAL, RALLIS, GSKCONS, BRFL, ANDHRABANK, ALKYLAMINE, HDIL, JBFIND, TATAELXSI

  AFTER: Invested ₹1,260,688 | Cash ₹810,629 | Total ₹2,071,318 | Positions 18/30 | Slot ₹69,044

========================================================================
  REBALANCE #23  —  01 Nov 2016
  NAV: ₹2,080,057  |  Slot: ₹69,335  |  Cash: ₹810,629
========================================================================

  EXITS (0)
    —

  ENTRIES (1)
  [52w filter blocked 3: NITINFIRE(-29.7%), VIDEOIND(-27.2%), PUNJLLOYD(-28.1%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  NAVKARCORP  10     INFRA      0.955    -0.01  +26.2%    -3.3%     202.3       342    ₹69,187       +2.5%     

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   19     IT         01-Jan-15   262.2       584.6       254    ₹81,870       +122.9% ⚠   -2.7%     
  GAEL        3      FMCG       02-May-16   12.9        24.2        4913   ₹55,387       +87.5%      +10.6%    
  VSTIND      1      FMCG       01-Jan-15   122.7       167.9       543    ₹24,592       +36.9%      +5.1%     
  RELCAPITAL  5      FIN SVC    01-Jun-16   332.1       438.1       189    ₹20,042       +31.9%      -3.7%     
  HDFC        7      PVT BNK    01-Jun-15   232.6       287.2       279    ₹15,248       +23.5%      -0.3%     
  COLPAL      11     FMCG       01-Jan-15   721.1       788.7       92     ₹6,220        +9.4% ⚠     +2.1%     
  ALBK        2      PSU BNK    01-Jan-15   175.0       187.4       381    ₹4,739        +7.1%       +2.4%     
  GSKCONS     15     FMCG       01-Jan-15   5,532.8     5,849.3     12     ₹3,798        +5.7% ⚠     -0.7%     
  RALLIS      14     FMCG       01-Jan-15   190.2       195.4       350    ₹1,827        +2.7% ⚠     -3.2%     
  ALKYLAMINE  13     MFG        01-Jan-15   131.9       128.9       505    ₹-1,523       -2.3% ⚠     +1.1%     
  BASF        6      MFG        01-Jan-15   1,254.6     1,213.8     53     ₹-2,162       -3.3%       +2.5%     
  JBFIND      17     FIN SVC    01-Jan-15   256.3       242.6       260    ₹-3,555       -5.3%       +2.1%     
  BRFL        20     CONSUMP    01-Aug-16   189.9       153.8       352    ₹-12,690      -19.0% ⚠    -2.9%     
  HDIL        18     REALTY     02-Feb-15   110.2       80.9        597    ₹-17,492      -26.6% ⚠    -0.8%     
  SYNDIBANK   8      PSU BNK    01-Jan-15   71.2        51.2        935    ₹-18,766      -28.2%      -1.0%     
  ORIENTBANK  12     PSU BNK    01-Jan-15   193.2       129.7       345    ₹-21,878      -32.8% ⚠    +0.9%     
  ANDHRABANK  16     PSU BNK    01-Jan-15   191.1       121.0       348    ₹-24,413      -36.7% ⚠    -0.4%     
  IL&FSTRANS  4      INFRA      02-Feb-15   200.3       109.6       328    ₹-29,780      -45.3%      +3.0%     
  ⚠  WAZ < 0 (momentum below universe mean): COLPAL, ORIENTBANK, ALKYLAMINE, RALLIS, GSKCONS, ANDHRABANK, HDIL, TATAELXSI, BRFL

  AFTER: Invested ₹1,338,614 | Cash ₹741,361 | Total ₹2,079,975 | Positions 19/30 | Slot ₹69,335

========================================================================
  REBALANCE #24  —  01 Dec 2016
  NAV: ₹1,965,464  |  Slot: ₹65,515  |  Cash: ₹741,361
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
  [52w filter blocked 3: NITINFIRE(-45.4%), VIDEOIND(-27.3%), PUNJLLOYD(-34.6%)]
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   19     IT         01-Jan-15   262.2       580.2       254    ₹80,755       +121.2% ⚠   +7.9%     
  GAEL        3      FMCG       02-May-16   12.9        20.6        4913   ₹38,104       +60.2%      -0.8%     
  VSTIND      5      FMCG       01-Jan-15   122.7       151.1       543    ₹15,449       +23.2%      -2.3%     
  HDFC        8      PVT BNK    01-Jun-15   232.6       273.0       279    ₹11,269       +17.4%      -1.4%     
  ALBK        2      PSU BNK    01-Jan-15   175.0       204.0       381    ₹11,066       +16.6%      +2.5%     
  RELCAPITAL  13     FIN SVC    01-Jun-16   332.1       364.7       189    ₹6,168        +9.8% ⚠     -4.5%     
  COLPAL      12     FMCG       01-Jan-15   721.1       757.6       92     ₹3,351        +5.1% ⚠     -0.4%     
  RALLIS      14     FMCG       01-Jan-15   190.2       178.8       350    ₹-3,969       -6.0% ⚠     +0.1%     
  GSKCONS     18     FMCG       01-Jan-15   5,532.8     4,946.7     12     ₹-7,034       -10.6% ⚠    -2.6%     
  ALKYLAMINE  11     MFG        01-Jan-15   131.9       116.9       505    ₹-7,580       -11.4% ⚠    -0.5%     
  NAVKARCORP  15     INFRA      01-Nov-16   202.3       175.6       342    ₹-9,114       -13.2% ⚠    -3.0%     
  BASF        9      MFG        01-Jan-15   1,254.6     1,027.9     53     ₹-12,016      -18.1%      -3.2%     
  JBFIND      16     FIN SVC    01-Jan-15   256.3       203.7       260    ₹-13,691      -20.5%      -2.6%     
  BRFL        20     CONSUMP    01-Aug-16   189.9       138.1       352    ₹-18,216      -27.3% ⚠    -4.9%     
  SYNDIBANK   4      PSU BNK    01-Jan-15   71.2        51.7        935    ₹-18,305      -27.5%      +1.1%     
  ANDHRABANK  7      PSU BNK    01-Jan-15   191.1       123.9       348    ₹-23,391      -35.2%      +0.1%     
  ORIENTBANK  6      PSU BNK    01-Jan-15   193.2       122.9       345    ₹-24,226      -36.4%      -3.1%     
  HDIL        17     REALTY     02-Feb-15   110.2       61.3        597    ₹-29,193      -44.4% ⚠    -3.4%     
  IL&FSTRANS  1      INFRA      02-Feb-15   200.3       101.3       328    ₹-32,469      -49.4%      +3.6%     
  ⚠  WAZ < 0 (momentum below universe mean): ALKYLAMINE, COLPAL, RELCAPITAL, RALLIS, NAVKARCORP, HDIL, GSKCONS, TATAELXSI, BRFL

  AFTER: Invested ₹1,224,103 | Cash ₹741,361 | Total ₹1,965,464 | Positions 19/30 | Slot ₹65,515

========================================================================
  REBALANCE #25  —  02 Jan 2017
  NAV: ₹1,967,106  |  Slot: ₹65,570  |  Cash: ₹741,361
========================================================================

  [REGIME OFF] Nifty 500 7,002.5 < SMA200 7,008.0 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   12     IT         01-Jan-15   262.2       629.8       254    ₹93,357       +140.2% ⚠   +3.3%     
  GAEL        1      FMCG       02-May-16   12.9        21.6        4913   ₹42,823       +67.7%      +1.3%     
  VSTIND      2      FMCG       01-Jan-15   122.7       163.7       543    ₹22,295       +33.5%      +4.6%     
  HDFC        8      PVT BNK    01-Jun-15   232.6       272.9       279    ₹11,260       +17.4% ⚠    +0.5%     
  RELCAPITAL  13     FIN SVC    01-Jun-16   332.1       370.1       189    ₹7,193        +11.5% ⚠    +1.0%     
  ALBK        3      PSU BNK    01-Jan-15   175.0       183.9       381    ₹3,402        +5.1%       -0.4%     
  COLPAL      16     FMCG       01-Jan-15   721.1       735.2       92     ₹1,295        +2.0% ⚠     -0.6%     
  RALLIS      6      FMCG       01-Jan-15   190.2       179.6       350    ₹-3,705       -5.6%       +3.4%     
  BASF        5      MFG        01-Jan-15   1,254.6     1,128.5     53     ₹-6,683       -10.0%      +5.0%     
  ALKYLAMINE  7      MFG        01-Jan-15   131.9       115.7       505    ₹-8,193       -12.3% ⚠    +1.6%     
  GSKCONS     20     FMCG       01-Jan-15   5,532.8     4,807.7     12     ₹-8,702       -13.1% ⚠    -1.0%     
  NAVKARCORP  15     INFRA      01-Nov-16   202.3       167.6       342    ₹-11,867      -17.2% ⚠    -0.4%     
  JBFIND      19     FIN SVC    01-Jan-15   256.3       210.7       260    ₹-11,850      -17.8% ⚠    +2.1%     
  BRFL        17     CONSUMP    01-Aug-16   189.9       148.8       352    ₹-14,450      -21.6% ⚠    +4.8%     
  SYNDIBANK   11     PSU BNK    01-Jan-15   71.2        43.1        935    ₹-26,318      -39.5% ⚠    -7.4%     
  HDIL        18     REALTY     02-Feb-15   110.2       61.7        597    ₹-28,954      -44.0% ⚠    +4.1%     
  ANDHRABANK  9      PSU BNK    01-Jan-15   191.1       105.3       348    ₹-29,858      -44.9% ⚠    -4.3%     
  ORIENTBANK  14     PSU BNK    01-Jan-15   193.2       104.2       345    ₹-30,681      -46.0% ⚠    -5.6%     
  IL&FSTRANS  4      INFRA      02-Feb-15   200.3       103.5       328    ₹-31,764      -48.3%      +2.0%     
  ⚠  WAZ < 0 (momentum below universe mean): ALKYLAMINE, HDFC, ANDHRABANK, SYNDIBANK, TATAELXSI, RELCAPITAL, ORIENTBANK, NAVKARCORP, COLPAL, BRFL, HDIL, JBFIND, GSKCONS

  AFTER: Invested ₹1,225,746 | Cash ₹741,361 | Total ₹1,967,106 | Positions 19/30 | Slot ₹65,570

========================================================================
  REBALANCE #26  —  01 Feb 2017
  NAV: ₹2,064,074  |  Slot: ₹68,802  |  Cash: ₹741,361
========================================================================

  EXITS (1)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ALBK        1      PSU BNK    01-Jan-15   175.0       226.1       381    ₹19,469       +29.2%    762d  

  ENTRIES (1)
  [52w filter blocked 1: PUNJLLOYD(-26.5%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  VIDEOIND    18     CON DUR    0.555    0.02   -7.7%     +0.0%     104.1       661    ₹68,777       -0.0%     

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   11     IT         01-Jan-15   262.2       646.5       254    ₹97,609       +146.5%     +0.5%     
  GAEL        5      FMCG       02-May-16   12.9        22.6        4913   ₹47,719       +75.4%      +1.8%     
  VSTIND      6      FMCG       01-Jan-15   122.7       167.7       543    ₹24,454       +36.7%      +1.7%     
  HDFC        3      PVT BNK    01-Jun-15   232.6       297.7       279    ₹18,168       +28.0%      +4.3%     
  RELCAPITAL  13     FIN SVC    01-Jun-16   332.1       391.3       189    ₹11,201       +17.8% ⚠    +2.3%     
  RALLIS      2      FMCG       01-Jan-15   190.2       212.7       350    ₹7,886        +11.8%      +9.7%     
  COLPAL      14     FMCG       01-Jan-15   721.1       730.5       92     ₹860          +1.3% ⚠     -0.3%     
  BASF        4      MFG        01-Jan-15   1,254.6     1,260.3     53     ₹299          +0.4%       +7.4%     
  ALKYLAMINE  12     MFG        01-Jan-15   131.9       130.0       505    ₹-995         -1.5%       +9.6%     
  GSKCONS     19     FMCG       01-Jan-15   5,532.8     4,982.4     12     ₹-6,605       -9.9% ⚠     +1.7%     
  JBFIND      21     FIN SVC    01-Jan-15   256.3       221.9       260    ₹-8,960       -13.4% ⚠    -1.8%     
  NAVKARCORP  18     INFRA      01-Nov-16   202.3       170.1       342    ₹-10,995      -15.9% ⚠    -1.1%     
  BRFL        17     CONSUMP    01-Aug-16   189.9       155.1       352    ₹-12,232      -18.3% ⚠    +2.8%     
  SYNDIBANK   10     PSU BNK    01-Jan-15   71.2        48.0        935    ₹-21,729      -32.6%      +3.9%     
  ANDHRABANK  8      PSU BNK    01-Jan-15   191.1       127.4       348    ₹-22,165      -33.3%      +8.8%     
  ORIENTBANK  9      PSU BNK    01-Jan-15   193.2       126.7       345    ₹-22,920      -34.4%      +7.8%     
  HDIL        16     REALTY     02-Feb-15   110.2       66.1        597    ₹-26,358      -40.1% ⚠    +5.5%     
  IL&FSTRANS  7      INFRA      02-Feb-15   200.3       111.5       328    ₹-29,140      -44.3%      +1.3%     
  ⚠  WAZ < 0 (momentum below universe mean): RELCAPITAL, COLPAL, HDIL, BRFL, NAVKARCORP, GSKCONS, JBFIND

  AFTER: Invested ₹1,305,360 | Cash ₹758,632 | Total ₹2,063,992 | Positions 19/30 | Slot ₹68,802

========================================================================
  REBALANCE #27  —  01 Mar 2017
  NAV: ₹2,128,637  |  Slot: ₹70,955  |  Cash: ₹758,632
========================================================================
  ⚠ Insufficient scored stocks — skipping rebalance

========================================================================
  REBALANCE #28  —  03 Apr 2017
  NAV: ₹2,207,638  |  Slot: ₹73,588  |  Cash: ₹758,632
========================================================================
  ⚠ Insufficient scored stocks — skipping rebalance

========================================================================
  REBALANCE #29  —  02 May 2017
  NAV: ₹2,289,923  |  Slot: ₹76,331  |  Cash: ₹758,632
========================================================================
  ⚠ Insufficient scored stocks — skipping rebalance

========================================================================
  REBALANCE #30  —  01 Jun 2017
  NAV: ₹2,191,804  |  Slot: ₹73,060  |  Cash: ₹758,632
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  RELCAPITAL  12     FIN SVC    01-Jun-16   332.1       463.6       189    ₹24,869       +39.6%    365d  
  HDIL        13     REALTY     02-Feb-15   110.2       90.8        597    ₹-11,612      -17.6%    850d  
  BRFL        —      OTHER      01-Aug-16   189.9       151.0       352    ₹-13,675      -20.5%    304d  
  ORIENTBANK  10     PSU BNK    01-Jan-15   193.2       137.0       345    ₹-19,389      -29.1%    882d  
  ANDHRABANK  11     PSU BNK    01-Jan-15   191.1       132.3       348    ₹-20,472      -30.8%    882d  
  IL&FSTRANS  15     INFRA      02-Feb-15   200.3       101.5       328    ₹-32,420      -49.3%    850d  

  ENTRIES (2)
  [52w filter blocked 5: STCINDIA(-42.3%), JUSTDIAL(-37.9%), JINDALPOLY(-21.5%), ORISSAMINE(-29.7%), ABAN(-36.5%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  DHANUKA     3      FMCG       1.483    0.61   +43.0%    +14.3%    826.7       88     ₹72,749       +5.0%     
  JYOTHYLAB   15     FMCG       0.891    0.62   +27.5%    -1.7%     157.2       464    ₹72,933       -1.2%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GAEL        4      FMCG       02-May-16   12.9        30.9        4913   ₹88,536       +139.9%     -2.9%     
  TATAELXSI   25     IT         01-Jan-15   262.2       611.9       254    ₹88,811       +133.3% ⚠   -4.5%     
  VSTIND      3      FMCG       01-Jan-15   122.7       211.3       543    ₹48,149       +72.3%      +1.5%     
  HDFC        1      PVT BNK    01-Jun-15   232.6       371.3       279    ₹38,709       +59.7%      +2.9%     
  COLPAL      6      FMCG       01-Jan-15   721.1       848.8       92     ₹11,745       +17.7%      +2.9%     
  RALLIS      14     FMCG       01-Jan-15   190.2       217.7       350    ₹9,622        +14.5% ⚠    +1.7%     
  ALKYLAMINE  16     MFG        01-Jan-15   131.9       144.0       505    ₹6,118        +9.2% ⚠     -7.2%     
  NAVKARCORP  8      INFRA      01-Nov-16   202.3       219.1       342    ₹5,746        +8.3%       +2.3%     
  BASF        7      MFG        01-Jan-15   1,254.6     1,350.0     53     ₹5,058        +7.6%       +3.6%     
  JBFIND      21     FIN SVC    01-Jan-15   256.3       268.2       260    ₹3,095        +4.6%       -2.1%     
  GSKCONS     20     FMCG       01-Jan-15   5,532.8     5,110.1     12     ₹-5,072       -7.6% ⚠     +0.8%     
  SYNDIBANK   5      PSU BNK    01-Jan-15   71.2        59.5        935    ₹-10,942      -16.4%      -1.2%     
  VIDEOIND    28     CON DUR    01-Feb-17   104.1       39.0        661    ₹-42,965      -62.5% ⚠    -45.0%    
  ⚠  WAZ < 0 (momentum below universe mean): ALKYLAMINE, RALLIS, GSKCONS, TATAELXSI, VIDEOIND

  AFTER: Invested ₹1,257,311 | Cash ₹934,320 | Total ₹2,191,632 | Positions 15/30 | Slot ₹73,060

========================================================================
  REBALANCE #31  —  03 Jul 2017
  NAV: ₹2,252,271  |  Slot: ₹75,076  |  Cash: ₹934,320
========================================================================
  ⚠ Insufficient scored stocks — skipping rebalance

========================================================================
  REBALANCE #32  —  01 Aug 2017
  NAV: ₹2,204,355  |  Slot: ₹73,479  |  Cash: ₹934,320
========================================================================
  ⚠ Insufficient scored stocks — skipping rebalance

========================================================================
  REBALANCE #33  —  01 Sep 2017
  NAV: ₹2,182,764  |  Slot: ₹72,759  |  Cash: ₹934,320
========================================================================
  ⚠ Insufficient scored stocks — skipping rebalance

========================================================================
  REBALANCE #34  —  03 Oct 2017
  NAV: ₹2,173,308  |  Slot: ₹72,444  |  Cash: ₹934,320
========================================================================
  ⚠ Insufficient scored stocks — skipping rebalance

========================================================================
  REBALANCE #35  —  01 Nov 2017
  NAV: ₹2,326,692  |  Slot: ₹77,556  |  Cash: ₹934,320
========================================================================
  ⚠ Insufficient scored stocks — skipping rebalance

========================================================================
  REBALANCE #36  —  01 Dec 2017
  NAV: ₹2,342,242  |  Slot: ₹78,075  |  Cash: ₹934,320
========================================================================
  ⚠ Insufficient scored stocks — skipping rebalance

========================================================================
  REBALANCE #37  —  01 Jan 2018
  NAV: ₹2,438,219  |  Slot: ₹81,274  |  Cash: ₹934,320
========================================================================

  EXITS (1)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VSTIND      —      OTHER      01-Jan-15   122.7       221.7       543    ₹53,805       +80.8%    1096d 

  ENTRIES (5)
  [52w filter blocked 6: IL&FSTRANS(-33.9%), ANDHRABANK(-29.4%), HDIL(-36.8%), ORIENTBANK(-26.7%), ABAN(-22.4%), BRFL(-49.2%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JUSTDIAL    5      IT         1.555    0.47   +62.3%    +41.4%    525.9       154    ₹80,989       +2.9%     
  ALBK        8      PSU BNK    1.865    0.13   +83.1%    +44.4%    317.4       256    ₹81,265       -1.9%     
  JINDALPOLY  15     METAL      0.741    0.47   +30.7%    +13.5%    372.3       218    ₹81,152       +2.4%     
  BFUTILITIE  17     ENERGY     1.020    -0.43  +34.0%    +26.7%    498.8       162    ₹80,806       +4.3%     
  SIEMENS     18     ENERGY     0.636    0.35   +17.6%    +4.2%     687.9       118    ₹81,171       +2.4%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GAEL        7      FMCG       02-May-16   12.9        46.6        4913   ₹165,501      +261.5%     +3.1%     
  TATAELXSI   12     IT         01-Jan-15   262.2       867.7       254    ₹153,776      +230.9%     +2.1%     
  ALKYLAMINE  2      MFG        01-Jan-15   131.9       248.7       505    ₹58,993       +88.5%      +6.0%     
  HDFC        4      PVT BNK    01-Jun-15   232.6       425.6       279    ₹53,864       +83.0%      -0.1%     
  BASF        3      MFG        01-Jan-15   1,254.6     2,086.7     53     ₹44,102       +66.3%      +2.2%     
  RALLIS      6      FMCG       01-Jan-15   190.2       243.5       350    ₹18,654       +28.0%      +7.6%     
  COLPAL      9      FMCG       01-Jan-15   721.1       909.9       92     ₹17,368       +26.2% ⚠    +2.6%     
  GSKCONS     1      FMCG       01-Jan-15   5,532.8     6,390.7     12     ₹10,294       +15.5%      +3.4%     
  JYOTHYLAB   20     FMCG       01-Jun-17   157.2       164.3       464    ₹3,285        +4.5% ⚠     -0.1%     
  NAVKARCORP  14     INFRA      01-Nov-16   202.3       190.7       342    ₹-3,967       -5.7% ⚠     +1.0%     
  JBFIND      21     FIN SVC    01-Jan-15   256.3       232.8       260    ₹-6,113       -9.2%       +3.6%     
  SYNDIBANK   19     PSU BNK    01-Jan-15   71.2        61.1        935    ₹-9,519       -14.3% ⚠    -1.8%     
  DHANUKA     10     FMCG       01-Jun-17   826.7       694.3       88     ₹-11,652      -16.0% ⚠    +4.9%     
  VIDEOIND    24     CON DUR    01-Feb-17   104.1       20.5        661    ₹-55,194      -80.2% ⚠    +15.9%    
  ⚠  WAZ < 0 (momentum below universe mean): COLPAL, DHANUKA, NAVKARCORP, SYNDIBANK, JYOTHYLAB, VIDEOIND

  AFTER: Invested ₹1,788,872 | Cash ₹648,866 | Total ₹2,437,738 | Positions 19/30 | Slot ₹81,274

========================================================================
  REBALANCE #38  —  01 Feb 2018
  NAV: ₹2,472,446  |  Slot: ₹82,415  |  Cash: ₹648,866
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
  [52w filter blocked 6: IL&FSTRANS(-36.6%), ANDHRABANK(-35.6%), HDIL(-44.4%), ABAN(-26.4%), ORIENTBANK(-27.6%), BRFL(-70.8%)]
    —

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GAEL        3      FMCG       02-May-16   12.9        58.3        4913   ₹223,174      +352.6%     +0.9%     
  TATAELXSI   9      IT         01-Jan-15   262.2       922.2       254    ₹167,618      +251.6%     -1.7%     
  ALKYLAMINE  4      MFG        01-Jan-15   131.9       260.9       505    ₹65,152       +97.8%      -2.6%     
  HDFC        1      PVT BNK    01-Jun-15   232.6       457.0       279    ₹62,614       +96.5%      +2.7%     
  BASF        6      MFG        01-Jan-15   1,254.6     2,015.1     53     ₹40,303       +60.6%      -4.8%     
  COLPAL      5      FMCG       01-Jan-15   721.1       932.9       92     ₹19,483       +29.4%      -0.1%     
  GSKCONS     2      FMCG       01-Jan-15   5,532.8     6,515.3     12     ₹11,790       +17.8%      +2.4%     
  RALLIS      10     FMCG       01-Jan-15   190.2       220.6       350    ₹10,647       +16.0% ⚠    -4.7%     
  SIEMENS     15     ENERGY     01-Jan-18   687.9       742.2       118    ₹6,409        +7.9%       +3.1%     
  JYOTHYLAB   18     FMCG       01-Jun-17   157.2       158.5       464    ₹616          +0.8% ⚠     -1.1%     
  JUSTDIAL    7      IT         01-Jan-18   525.9       512.2       154    ₹-2,118       -2.6%       -7.9%     
  BFUTILITIE  19     ENERGY     01-Jan-18   498.8       469.0       162    ₹-4,828       -6.0% ⚠     -7.7%     
  ALBK        14     PSU BNK    01-Jan-18   317.4       297.6       256    ₹-5,074       -6.2%       -6.0%     
  NAVKARCORP  11     INFRA      01-Nov-16   202.3       185.2       342    ₹-5,848       -8.5% ⚠     -4.5%     
  JINDALPOLY  16     METAL      01-Jan-18   372.3       326.2       218    ₹-10,034      -12.4% ⚠    -5.3%     
  DHANUKA     8      FMCG       01-Jun-17   826.7       676.0       88     ₹-13,263      -18.2%      -5.4%     
  SYNDIBANK   20     PSU BNK    01-Jan-15   71.2        55.8        935    ₹-14,394      -21.6% ⚠    -7.1%     
  JBFIND      24     FIN SVC    01-Jan-15   256.3       174.0       260    ₹-21,401      -32.1% ⚠    -15.1%    
  VIDEOIND    22     CON DUR    01-Feb-17   104.1       18.2        661    ₹-56,747      -82.5% ⚠    -10.8%    
  ⚠  WAZ < 0 (momentum below universe mean): RALLIS, NAVKARCORP, JINDALPOLY, JYOTHYLAB, BFUTILITIE, SYNDIBANK, VIDEOIND, JBFIND

  AFTER: Invested ₹1,823,581 | Cash ₹648,866 | Total ₹2,472,446 | Positions 19/30 | Slot ₹82,415

========================================================================
  REBALANCE #39  —  01 Mar 2018
  NAV: ₹2,416,484  |  Slot: ₹80,549  |  Cash: ₹648,866
========================================================================

  EXITS (0)
    —

  ENTRIES (8)
  [52w filter blocked 3: RAIN(-23.7%), IL&FSTRANS(-43.5%), ANDHRABANK(-49.5%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  VESUVIUS    5      MFG        1.777    0.06   +12.7%    +16.7%    129.4       622    ₹80,496       +0.7%     
  CRISIL      8      MNC        1.184    0.15   -3.4%     +4.8%     1,688.6     47     ₹79,362       -0.3%     
  VBL         10     FMCG       2.728    0.44   +65.9%    +27.9%    37.3        2159   ₹80,533       -1.6%     
  KSB         11     MFG        0.995    0.28   +27.4%    -7.8%     147.5       546    ₹80,516       -2.3%     
  CASTROLIND  12     ENERGY     0.944    0.37   -0.6%     +0.3%     137.6       585    ₹80,519       +3.8%     
  ACC         14     MFG        1.039    0.23   +14.8%    -2.9%     1,500.3     53     ₹79,518       -1.5%     
  AMBUJACEM   15     INFRA      0.931    0.06   +10.4%    -3.8%     221.7       363    ₹80,464       -2.0%     
  CIEINDIA    18     AUTO       0.968    0.18   +17.6%    -5.2%     224.0       359    ₹80,419       +3.5%     

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GAEL        1      FMCG       02-May-16   12.9        65.9        4913   ₹260,237      +411.2%     +9.3%     
  TATAELXSI   13     IT         01-Jan-15   262.2       940.1       254    ₹172,179      +258.5%     +2.3%     
  HDFC        4      PVT BNK    01-Jun-15   232.6       430.2       279    ₹55,135       +85.0%      -0.8%     
  ALKYLAMINE  7      MFG        01-Jan-15   131.9       237.3       505    ₹53,197       +79.8%      -1.5%     
  BASF        6      MFG        01-Jan-15   1,254.6     1,999.4     53     ₹39,472       +59.4%      -0.9%     
  COLPAL      9      FMCG       01-Jan-15   721.1       872.2       92     ₹13,899       +20.9%      -1.9%     
  GSKCONS     2      FMCG       01-Jan-15   5,532.8     6,620.1     12     ₹13,047       +19.7%      +3.7%     
  RALLIS      20     FMCG       01-Jan-15   190.2       208.0       350    ₹6,227        +9.4% ⚠     -0.9%     
  JYOTHYLAB   25     FMCG       01-Jun-17   157.2       149.5       464    ₹-3,573       -4.9% ⚠     -2.1%     
  SIEMENS     24     ENERGY     01-Jan-18   687.9       646.6       118    ₹-4,878       -6.0% ⚠     -5.4%     
  BFUTILITIE  28     ENERGY     01-Jan-18   498.8       436.2       162    ₹-10,141      -12.6% ⚠    -4.6%     
  JINDALPOLY  23     METAL      01-Jan-18   372.3       319.0       218    ₹-11,613      -14.3% ⚠    -0.1%     
  JUSTDIAL    17     IT         01-Jan-18   525.9       448.6       154    ₹-11,904      -14.7% ⚠    -4.2%     
  NAVKARCORP  19     INFRA      01-Nov-16   202.3       170.9       342    ₹-10,739      -15.5% ⚠    -1.3%     
  ALBK        26     PSU BNK    01-Jan-18   317.4       267.0       256    ₹-12,907      -15.9% ⚠    -5.2%     
  DHANUKA     16     FMCG       01-Jun-17   826.7       581.1       88     ₹-21,611      -29.7% ⚠    -6.5%     
  SYNDIBANK   29     PSU BNK    01-Jan-15   71.2        49.1        935    ₹-20,661      -31.0% ⚠    -7.6%     
  JBFIND      32     FIN SVC    01-Jan-15   256.3       149.2       260    ₹-27,836      -41.8% ⚠    -11.6%    
  VIDEOIND    31     CON DUR    01-Feb-17   104.1       14.2        661    ₹-59,391      -86.4% ⚠    -16.6%    
  ⚠  WAZ < 0 (momentum below universe mean): DHANUKA, JUSTDIAL, NAVKARCORP, RALLIS, JINDALPOLY, SIEMENS, JYOTHYLAB, ALBK, BFUTILITIE, SYNDIBANK, VIDEOIND, JBFIND

  AFTER: Invested ₹2,409,446 | Cash ₹6,276 | Total ₹2,415,722 | Positions 27/30 | Slot ₹80,549

========================================================================
  REBALANCE #40  —  02 Apr 2018
  NAV: ₹2,300,248  |  Slot: ₹76,675  |  Cash: ₹6,276
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VESUVIUS    —      OTHER      01-Mar-18   129.4       123.0       622    ₹-3,998       -5.0%     32d   
  JUSTDIAL    12     IT         01-Jan-18   525.9       483.5       154    ₹-6,537       -8.1%     91d   
  VIDEOIND    —      OTHER      01-Feb-17   104.1       12.4        661    ₹-60,581      -88.1%    425d  

  ENTRIES (1)
  [52w filter blocked 5: IL&FSTRANS(-50.7%), ANDHRABANK(-53.7%), HDIL(-61.0%), ABAN(-42.5%), ORIENTBANK(-58.4%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RAIN        2      METAL      3.330    0.61   +286.5%   +5.4%     361.6       212    ₹76,657       +0.6%     

  HOLDS (24)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GAEL        3      FMCG       02-May-16   12.9        58.3        4913   ₹222,936      +352.2%     -2.5%     
  TATAELXSI   11     IT         01-Jan-15   262.2       901.9       254    ₹162,463      +243.9%     +0.1%     
  HDFC        1      PVT BNK    01-Jun-15   232.6       443.3       279    ₹58,775       +90.6%      +2.9%     
  ALKYLAMINE  7      MFG        01-Jan-15   131.9       232.2       505    ₹50,632       +76.0%      +0.6%     
  BASF        5      MFG        01-Jan-15   1,254.6     1,902.8     53     ₹34,353       +51.7%      -2.5%     
  COLPAL      9      FMCG       01-Jan-15   721.1       876.7       92     ₹14,311       +21.6%      +0.9%     
  RALLIS      19     FMCG       01-Jan-15   190.2       211.2       350    ₹7,359        +11.1% ⚠    +2.7%     
  JYOTHYLAB   13     FMCG       01-Jun-17   157.2       170.9       464    ₹6,357        +8.7%       +7.2%     
  GSKCONS     14     FMCG       01-Jan-15   5,532.8     5,683.6     12     ₹1,809        +2.7% ⚠     -8.9%     
  CASTROLIND  4      ENERGY     01-Mar-18   137.6       139.8       585    ₹1,277        +1.6%       +1.6%     
  KSB         8      MFG        01-Mar-18   147.5       148.8       546    ₹731          +0.9%       +1.1%     
  VBL         10     FMCG       01-Mar-18   37.3        37.0        2159   ₹-555         -0.7%       +2.3%     
  CRISIL      6      MNC        01-Mar-18   1,688.6     1,622.5     47     ₹-3,105       -3.9%       -3.4%     
  AMBUJACEM   18     INFRA      01-Mar-18   221.7       210.3       363    ₹-4,116       -5.1% ⚠     +1.0%     
  ACC         16     MFG        01-Mar-18   1,500.3     1,411.4     53     ₹-4,714       -5.9% ⚠     -1.8%     
  CIEINDIA    20     AUTO       01-Mar-18   224.0       203.6       359    ₹-7,337       -9.1% ⚠     -1.1%     
  SIEMENS     25     ENERGY     01-Jan-18   687.9       617.8       118    ₹-8,272       -10.2% ⚠    -1.5%     
  NAVKARCORP  15     INFRA      01-Nov-16   202.3       170.6       342    ₹-10,841      -15.7% ⚠    +7.1%     
  ALBK        23     PSU BNK    01-Jan-18   317.4       254.3       256    ₹-16,171      -19.9% ⚠    +0.4%     
  BFUTILITIE  28     ENERGY     01-Jan-18   498.8       392.2       162    ₹-17,269      -21.4% ⚠    -2.5%     
  JINDALPOLY  24     METAL      01-Jan-18   372.3       289.7       218    ₹-18,008      -22.2% ⚠    -3.6%     
  DHANUKA     17     FMCG       01-Jun-17   826.7       536.6       88     ₹-25,530      -35.1% ⚠    -1.7%     
  SYNDIBANK   27     PSU BNK    01-Jan-15   71.2        44.7        935    ₹-24,776      -37.2% ⚠    -1.4%     
  JBFIND      32     FIN SVC    01-Jan-15   256.3       77.3        260    ₹-46,530      -69.8% ⚠    -33.9%    
  ⚠  WAZ < 0 (momentum below universe mean): GSKCONS, NAVKARCORP, ACC, DHANUKA, AMBUJACEM, RALLIS, CIEINDIA, ALBK, JINDALPOLY, SIEMENS, SYNDIBANK, BFUTILITIE, JBFIND

  AFTER: Invested ₹2,211,484 | Cash ₹88,673 | Total ₹2,300,157 | Positions 25/30 | Slot ₹76,675

========================================================================
  REBALANCE #41  —  02 May 2018
  NAV: ₹2,432,140  |  Slot: ₹81,071  |  Cash: ₹88,673
========================================================================

  EXITS (0)
    —

  ENTRIES (0)
  [52w filter blocked 5: IL&FSTRANS(-50.2%), ANDHRABANK(-55.2%), HDIL(-67.3%), ABAN(-39.1%), ORIENTBANK(-59.9%)]
    —

  HOLDS (25)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  GAEL        2      FMCG       02-May-16   12.9        68.0        4913   ₹270,750      +427.8%     +1.5%     
  TATAELXSI   5      IT         01-Jan-15   262.2       1,093.7     254    ₹211,190      +317.1%     +7.1%     
  HDFC        1      PVT BNK    01-Jun-15   232.6       452.0       279    ₹61,228       +94.4%      +2.1%     
  ALKYLAMINE  9      MFG        01-Jan-15   131.9       255.6       505    ₹62,473       +93.8%      +3.6%     
  BASF        3      MFG        01-Jan-15   1,254.6     2,120.9     53     ₹45,912       +69.0%      +6.1%     
  COLPAL      7      FMCG       01-Jan-15   721.1       936.8       92     ₹19,839       +29.9%      +2.7%     
  GSKCONS     11     FMCG       01-Jan-15   5,532.8     5,923.6     12     ₹4,690        +7.1%       +0.4%     
  RALLIS      18     FMCG       01-Jan-15   190.2       200.0       350    ₹3,458        +5.2% ⚠     -3.0%     
  KSB         6      MFG        01-Mar-18   147.5       152.6       546    ₹2,829        +3.5%       -0.1%     
  VBL         20     FMCG       01-Mar-18   37.3        38.6        2159   ₹2,776        +3.4%       +3.4%     
  CIEINDIA    8      AUTO       01-Mar-18   224.0       229.4       359    ₹1,932        +2.4%       +4.3%     
  JYOTHYLAB   16     FMCG       01-Jun-17   157.2       158.6       464    ₹636          +0.9%       -1.0%     
  CASTROLIND  4      ENERGY     01-Mar-18   137.6       138.4       585    ₹419          +0.5%       +0.4%     
  AMBUJACEM   13     INFRA      01-Mar-18   221.7       216.8       363    ₹-1,782       -2.2%       +1.0%     
  CRISIL      12     MNC        01-Mar-18   1,688.6     1,624.0     47     ₹-3,033       -3.8% ⚠     -3.1%     
  ACC         17     MFG        01-Mar-18   1,500.3     1,429.9     53     ₹-3,731       -4.7% ⚠     -0.3%     
  SIEMENS     25     ENERGY     01-Jan-18   687.9       632.9       118    ₹-6,486       -8.0% ⚠     +2.6%     
  NAVKARCORP  15     INFRA      01-Nov-16   202.3       176.9       342    ₹-8,670       -12.5% ⚠    +1.6%     
  BFUTILITIE  26     ENERGY     01-Jan-18   498.8       417.1       162    ₹-13,235      -16.4% ⚠    +0.9%     
  ALBK        23     PSU BNK    01-Jan-18   317.4       258.4       256    ₹-15,115      -18.6% ⚠    -1.3%     
  RAIN        10     METAL      02-Apr-18   361.6       289.0       212    ₹-15,391      -20.1%      -10.7%    
  JINDALPOLY  24     METAL      01-Jan-18   372.3       257.6       218    ₹-25,001      -30.8% ⚠    -10.4%    
  DHANUKA     19     FMCG       01-Jun-17   826.7       556.2       88     ₹-23,807      -32.7% ⚠    +0.1%     
  SYNDIBANK   27     PSU BNK    01-Jan-15   71.2        43.4        935    ₹-26,034      -39.1% ⚠    -3.9%     
  JBFIND      31     FIN SVC    01-Jan-15   256.3       102.2       260    ₹-40,082      -60.1% ⚠    -2.2%     
  ⚠  WAZ < 0 (momentum below universe mean): CRISIL, NAVKARCORP, ACC, RALLIS, DHANUKA, ALBK, JINDALPOLY, SIEMENS, BFUTILITIE, SYNDIBANK, JBFIND

  AFTER: Invested ₹2,343,467 | Cash ₹88,673 | Total ₹2,432,140 | Positions 25/30 | Slot ₹81,071

========================================================================
  REBALANCE #42  —  01 Jun 2018
  NAV: ₹2,304,933  |  Slot: ₹76,831  |  Cash: ₹88,673
========================================================================

  EXITS (19)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  GAEL        168    FMCG       02-May-16   12.9        53.2        4913   ₹198,109      +313.0%   760d  
  BASF        83     MFG        01-Jan-15   1,254.6     1,961.2     53     ₹37,447       +56.3%    1247d 
  GSKCONS     118    FMCG       01-Jan-15   5,532.8     6,405.2     12     ₹10,469       +15.8%    1247d 
  KSB         85     MFG        01-Mar-18   147.5       152.8       546    ₹2,931        +3.6%     92d   
  CIEINDIA    157    AUTO       01-Mar-18   224.0       226.6       359    ₹932          +1.2%     92d   
  RALLIS      231    FMCG       01-Jan-15   190.2       191.3       350    ₹390          +0.6%     1247d 
  CRISIL      207    MNC        01-Mar-18   1,688.6     1,563.1     47     ₹-5,897       -7.4%     92d   
  ALBK        300    PSU BNK    01-Jan-18   317.4       284.2       256    ₹-8,500       -10.5%    151d  
  CASTROLIND  279    ENERGY     01-Mar-18   137.6       117.8       585    ₹-11,623      -14.4%    92d   
  SIEMENS     315    ENERGY     01-Jan-18   687.9       574.9       118    ₹-13,330      -16.4%    151d  
  ACC         304    MFG        01-Mar-18   1,500.3     1,231.4     53     ₹-14,253      -17.9%    92d   
  AMBUJACEM   303    INFRA      01-Mar-18   221.7       179.8       363    ₹-15,214      -18.9%    92d   
  BFUTILITIE  344    ENERGY     01-Jan-18   498.8       369.1       162    ₹-21,011      -26.0%    151d  
  NAVKARCORP  275    INFRA      01-Nov-16   202.3       145.9       342    ₹-19,289      -27.9%    577d  
  DHANUKA     251    FMCG       01-Jun-17   826.7       529.1       88     ₹-26,192      -36.0%    365d  
  SYNDIBANK   337    PSU BNK    01-Jan-15   71.2        43.3        935    ₹-26,098      -39.2%    1247d 
  JINDALPOLY  323    METAL      01-Jan-18   372.3       221.2       218    ₹-32,935      -40.6%    151d  
  RAIN        203    METAL      02-Apr-18   361.6       210.7       212    ₹-31,996      -41.7%    60d   
  JBFIND      362    FIN SVC    01-Jan-15   256.3       86.4        260    ₹-44,177      -66.3%    1247d 

  ENTRIES (19)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HEG         1      METAL      6.682    0.75   +1137.2%  +19.0%    571.2       134    ₹76,545       +1.3%     
  GRAPHITE    2      METAL      5.033    0.45   +667.3%   +25.1%    670.8       114    ₹76,466       +4.5%     
  BRITANNIA   3      FMCG       3.293    0.26   +71.0%    +16.4%    2,571.5     29     ₹74,572       +2.9%     
  HINDUNILVR  4      FMCG       3.260    0.37   +50.5%    +20.6%    1,385.4     55     ₹76,195       +2.3%     
  JUBLFOOD    5      CONSUMP    3.394    0.45   +166.5%   +23.0%    245.3       313    ₹76,777       -1.3%     
  FSL         6      IT         3.114    0.77   +127.4%   +46.1%    58.2        1319   ₹76,830       +7.8%     
  PIDILITIND  7      MFG        2.982    0.47   +49.2%    +24.2%    538.3       142    ₹76,437       +0.5%     
  DABUR       8      FMCG       2.869    0.17   +39.8%    +19.0%    354.9       216    ₹76,669       +3.3%     
  KOTAKBANK   9      PVT BNK    2.868    0.66   +36.2%    +20.9%    262.2       292    ₹76,571       +3.5%     
  COFORGE     10     IT         2.944    0.72   +120.6%   +32.7%    201.6       381    ₹76,791       +3.3%     
  MPHASIS     12     IT         2.775    0.22   +87.4%    +23.0%    868.5       88     ₹76,424       +0.6%     
  M&M         13     AUTO       2.818    0.70   +33.8%    +23.8%    829.0       92     ₹76,271       +4.7%     
  HDFCBANK    14     PVT BNK    2.686    0.53   +31.0%    +12.7%    487.5       157    ₹76,541       +4.9%     
  BAJFINANCE  15     FIN SVC    2.693    0.65   +59.3%    +26.7%    201.1       382    ₹76,816       +2.1%     
  SONATSOFTW  16     IT         2.697    0.87   +158.2%   +13.5%    108.8       706    ₹76,808       +2.5%     
  TECHM       18     IT         2.703    0.03   +89.2%    +14.4%    516.6       148    ₹76,459       +2.4%     
  DMART       19     FMCG       2.663    0.47   +114.2%   +13.6%    1,531.1     50     ₹76,555       +3.3%     
  SOLARINDS   20     DEFENCE    2.535    0.39   +39.4%    +13.3%    1,133.0     67     ₹75,909       +3.7%     
  VIPIND      21     CON DUR    2.538    0.67   +120.0%   +21.8%    398.6       192    ₹76,534       +1.3%     

  HOLDS (6)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   24     IT         01-Jan-15   262.2       1,091.8     254    ₹210,698      +316.3%     +1.6%     
  HDFC        17     PVT BNK    01-Jun-15   232.6       487.5       279    ₹71,127       +109.6%     +4.9%     
  ALKYLAMINE  58     MFG        01-Jan-15   131.9       249.1       505    ₹59,184       +88.8%      -0.1%     
  COLPAL      11     FMCG       01-Jan-15   721.1       1,043.3     92     ₹29,640       +44.7%      +4.3%     
  VBL         40     FMCG       01-Mar-18   37.3        44.6        2159   ₹15,684       +19.5%      +7.1%     
  JYOTHYLAB   45     FMCG       01-Jun-17   157.2       184.0       464    ₹12,460       +17.1%      +2.8%     

  AFTER: Invested ₹2,268,894 | Cash ₹34,315 | Total ₹2,303,208 | Positions 25/30 | Slot ₹76,831

========================================================================
  REBALANCE #43  —  02 Jul 2018
  NAV: ₹2,330,463  |  Slot: ₹77,682  |  Cash: ₹34,315
========================================================================

  [REGIME OFF] Nifty 500 9,109.0 < SMA200 9,211.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (25)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   8      IT         01-Jan-15   262.2       1,209.4     254    ₹240,577      +361.2%     +3.9%     
  HDFC        20     PVT BNK    01-Jun-15   232.6       478.9       279    ₹68,720       +105.9%     +0.3%     
  ALKYLAMINE  52     MFG        01-Jan-15   131.9       241.1       505    ₹55,151       +82.8%      -2.3%     
  COLPAL      45     FMCG       01-Jan-15   721.1       976.9       92     ₹23,533       +35.5%      -2.1%     
  JYOTHYLAB   38     FMCG       01-Jun-17   157.2       204.8       464    ₹22,078       +30.3%      +6.4%     
  VBL         51     FMCG       01-Mar-18   37.3        43.1        2159   ₹12,465       +15.5%      -0.4%     
  JUBLFOOD    3      CONSUMP    01-Jun-18   245.3       276.3       313    ₹9,707        +12.6%      +3.4%     
  BAJFINANCE  12     FIN SVC    01-Jun-18   201.1       222.6       382    ₹8,214        +10.7%      +2.0%     
  BRITANNIA   2      FMCG       01-Jun-18   2,571.5     2,757.5     29     ₹5,395        +7.2%       +3.8%     
  GRAPHITE    6      METAL      01-Jun-18   670.8       701.0       114    ₹3,447        +4.5%       +7.9%     
  HINDUNILVR  4      FMCG       01-Jun-18   1,385.4     1,443.8     55     ₹3,213        +4.2%       +2.6%     
  MPHASIS     9      IT         01-Jun-18   868.5       903.3       88     ₹3,071        +4.0%       +1.5%     
  HEG         1      METAL      01-Jun-18   571.2       590.2       134    ₹2,545        +3.3%       +6.3%     
  VIPIND      10     CON DUR    01-Jun-18   398.6       408.8       192    ₹1,958        +2.6%       -1.4%     
  KOTAKBANK   11     PVT BNK    01-Jun-18   262.2       266.2       292    ₹1,159        +1.5%       +1.3%     
  COFORGE     15     IT         01-Jun-18   201.6       201.1       381    ₹-173         -0.2%       +0.8%     
  DMART       18     FMCG       01-Jun-18   1,531.1     1,525.9     50     ₹-260         -0.3%       +1.8%     
  DABUR       16     FMCG       01-Jun-18   354.9       352.1       216    ₹-605         -0.8%       -0.2%     
  M&M         21     AUTO       01-Jun-18   829.0       815.3       92     ₹-1,265       -1.7%       -1.2%     
  HDFCBANK    19     PVT BNK    01-Jun-18   487.5       478.9       157    ₹-1,355       -1.8%       +0.3%     
  SOLARINDS   40     DEFENCE    01-Jun-18   1,133.0     1,075.9     67     ₹-3,822       -5.0%       -3.5%     
  PIDILITIND  28     MFG        01-Jun-18   538.3       509.6       142    ₹-4,073       -5.3%       -0.6%     
  TECHM       25     IT         01-Jun-18   516.6       484.2       148    ₹-4,793       -6.3%       -4.4%     
  FSL         17     IT         01-Jun-18   58.2        53.2        1319   ₹-6,594       -8.6%       -6.3%     
  SONATSOFTW  37     IT         01-Jun-18   108.8       91.4        706    ₹-12,245      -15.9%      -4.6%     

  AFTER: Invested ₹2,296,149 | Cash ₹34,315 | Total ₹2,330,463 | Positions 25/30 | Slot ₹77,682

========================================================================
  REBALANCE #44  —  01 Aug 2018
  NAV: ₹2,502,112  |  Slot: ₹83,404  |  Cash: ₹34,315
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ALKYLAMINE  69     MFG        01-Jan-15   131.9       249.8       505    ₹59,532       +89.4%    1308d 
  COLPAL      84     FMCG       01-Jan-15   721.1       939.3       92     ₹20,069       +30.3%    1308d 
  JYOTHYLAB   59     FMCG       01-Jun-17   157.2       195.2       464    ₹17,654       +24.2%    426d  
  VBL         102    FMCG       01-Mar-18   37.3        43.0        2159   ₹12,365       +15.4%    153d  
  PIDILITIND  38     MFG        01-Jun-18   538.3       543.6       142    ₹758          +1.0%     61d   
  KOTAKBANK   43     PVT BNK    01-Jun-18   262.2       261.3       292    ₹-269         -0.4%     61d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  GLAXO       4      HEALTH     3.381    0.31   +32.0%    +36.8%    1,303.6     63     ₹82,124       +7.0%     
  RELIANCE    8      OIL&GAS    3.013    0.79   +50.6%    +25.8%    527.8       158    ₹83,389       +8.4%     
  BAJAJFINSV  10     FIN SVC    2.774    0.82   +41.9%    +30.7%    696.2       119    ₹82,852       +5.8%     
  PAGEIND     12     MFG        2.741    0.68   +82.6%    +26.9%    27,338.4    3      ₹82,015       +4.9%     
  TCS         13     IT         2.722    0.26   +61.6%    +14.7%    1,618.1     51     ₹82,521       +1.6%     
  RELAXO      15     CON DUR    2.685    0.19   +70.7%    +19.8%    398.8       209    ₹83,352       +5.1%     
  BATAINDIA   16     CON DUR    2.667    0.68   +64.4%    +21.6%    878.9       94     ₹82,621       +7.6%     

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   20     IT         01-Jan-15   262.2       1,301.3     254    ₹263,934      +396.2%     +2.4%     
  HDFC        30     PVT BNK    01-Jun-15   232.6       498.6       279    ₹74,226       +114.4%     +0.0%     
  BAJFINANCE  5      FIN SVC    01-Jun-18   201.1       264.0       382    ₹24,038       +31.3%      +5.8%     
  HEG         1      METAL      01-Jun-18   571.2       747.1       134    ₹23,570       +30.8%      +10.1%    
  GRAPHITE    2      METAL      01-Jun-18   670.8       821.6       114    ₹17,197       +22.5%      +6.1%     
  VIPIND      7      CON DUR    01-Jun-18   398.6       471.6       192    ₹14,013       +18.3%      +7.5%     
  MPHASIS     9      IT         01-Jun-18   868.5       1,006.5     88     ₹12,148       +15.9%      +6.6%     
  DABUR       14     FMCG       01-Jun-18   354.9       403.3       216    ₹10,435       +13.6%      +11.8%    
  COFORGE     11     IT         01-Jun-18   201.6       227.3       381    ₹9,828        +12.8%      +6.2%     
  BRITANNIA   3      FMCG       01-Jun-18   2,571.5     2,882.3     29     ₹9,015        +12.1%      +2.0%     
  JUBLFOOD    24     CONSUMP    01-Jun-18   245.3       273.8       313    ₹8,911        +11.6%      -1.1%     
  HINDUNILVR  6      FMCG       01-Jun-18   1,385.4     1,523.6     55     ₹7,603        +10.0%      +3.2%     
  DMART       25     FMCG       01-Jun-18   1,531.1     1,669.7     50     ₹6,930        +9.1%       +6.2%     
  SOLARINDS   21     DEFENCE    01-Jun-18   1,133.0     1,212.4     67     ₹5,324        +7.0%       +8.4%     
  M&M         37     AUTO       01-Jun-18   829.0       861.9       92     ₹3,020        +4.0%       +1.6%     
  HDFCBANK    29     PVT BNK    01-Jun-18   487.5       498.6       157    ₹1,744        +2.3%       +0.0%     
  TECHM       28     IT         01-Jun-18   516.6       513.2       148    ₹-502         -0.7%       +5.1%     
  FSL         27     IT         01-Jun-18   58.2        56.9        1319   ₹-1,805       -2.3%       +0.2%     
  SONATSOFTW  32     IT         01-Jun-18   108.8       105.9       706    ₹-2,044       -2.7%       +6.2%     

  AFTER: Invested ₹2,497,121 | Cash ₹4,303 | Total ₹2,501,424 | Positions 26/30 | Slot ₹83,404

========================================================================
  REBALANCE #45  —  03 Sep 2018
  NAV: ₹2,582,766  |  Slot: ₹86,092  |  Cash: ₹4,303
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HDFC        115    PVT BNK    01-Jun-15   232.6       479.3       279    ₹68,836       +106.1%   1190d 
  M&M         74     AUTO       01-Jun-18   829.0       878.6       92     ₹4,559        +6.0%     94d   
  DMART       69     FMCG       01-Jun-18   1,531.1     1,604.7     50     ₹3,678        +4.8%     94d   
  SOLARINDS   84     DEFENCE    01-Jun-18   1,133.0     1,151.8     67     ₹1,262        +1.7%     94d   
  HDFCBANK    113    PVT BNK    01-Jun-18   487.5       479.3       157    ₹-1,289       -1.7%     94d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  INFY        8      IT         2.798    0.30   +60.8%    +18.6%    577.9       148    ₹85,522       +2.5%     
  GODREJCP    10     FMCG       2.817    0.30   +56.4%    +26.5%    894.1       96     ₹85,835       +3.6%     
  TORNTPHARM  17     HEALTH     2.573    0.40   +52.3%    +29.1%    826.6       104    ₹85,965       +6.0%     
  HAVELLS     18     CON DUR    2.641    0.97   +48.7%    +34.5%    680.5       126    ₹85,747       +4.0%     
  TASTYBITE   19     FMCG       2.398    0.88   +89.4%    +40.5%    10,408.2    8      ₹83,265       +14.5%    

  HOLDS (21)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   23     IT         01-Jan-15   262.2       1,310.8     254    ₹266,330      +399.8%     +1.8%     
  VIPIND      2      CON DUR    01-Jun-18   398.6       599.4       192    ₹38,545       +50.4%      +6.3%     
  BAJFINANCE  16     FIN SVC    01-Jun-18   201.1       264.1       382    ₹24,084       +31.4%      -3.5%     
  HEG         1      METAL      01-Jun-18   571.2       733.1       134    ₹21,685       +28.3%      +2.2%     
  COFORGE     6      IT         01-Jun-18   201.6       251.5       381    ₹19,014       +24.8%      +4.5%     
  DABUR       10     FMCG       01-Jun-18   354.9       435.1       216    ₹17,306       +22.6%      +2.1%     
  JUBLFOOD    13     CONSUMP    01-Jun-18   245.3       299.4       313    ₹16,927       +22.0%      +1.0%     
  GRAPHITE    21     METAL      01-Jun-18   670.8       812.7       114    ₹16,179       +21.2%      -1.0%     
  MPHASIS     15     IT         01-Jun-18   868.5       1,042.5     88     ₹15,312       +20.0%      +2.9%     
  PAGEIND     8      MFG        01-Aug-18   27,338.4    31,091.5    3      ₹11,259       +13.7%      +1.8%     
  BRITANNIA   14     FMCG       01-Jun-18   2,571.5     2,916.1     29     ₹9,994        +13.4%      -0.8%     
  BATAINDIA   5      CON DUR    01-Aug-18   878.9       993.8       94     ₹10,793       +13.1%      +3.7%     
  GLAXO       3      HEALTH     01-Aug-18   1,303.6     1,420.5     63     ₹7,367        +9.0%       +4.9%     
  TECHM       36     IT         01-Jun-18   516.6       561.9       148    ₹6,697        +8.8%       +5.5%     
  HINDUNILVR  35     FMCG       01-Jun-18   1,385.4     1,492.7     55     ₹5,902        +7.7%       -2.8%     
  TCS         12     IT         01-Aug-18   1,618.1     1,680.6     51     ₹3,190        +3.9%       +1.3%     
  RELAXO      17     CON DUR    01-Aug-18   398.8       412.6       209    ₹2,880        +3.5%       +1.1%     
  RELIANCE    4      OIL&GAS    01-Aug-18   527.8       544.1       158    ₹2,574        +3.1%       -0.1%     
  FSL         65     IT         01-Jun-18   58.2        60.0        1319   ₹2,369        +3.1%       +7.3%     
  SONATSOFTW  52     IT         01-Jun-18   108.8       105.2       706    ₹-2,528       -3.3%       +0.3%     
  BAJAJFINSV  63     FIN SVC    01-Aug-18   696.2       662.6       119    ₹-3,997       -4.8%       -3.8%     

  AFTER: Invested ₹2,557,584 | Cash ₹24,675 | Total ₹2,582,259 | Positions 26/30 | Slot ₹86,092

========================================================================
  REBALANCE #46  —  01 Oct 2018
  NAV: ₹2,264,984  |  Slot: ₹75,499  |  Cash: ₹24,675
========================================================================

  [REGIME OFF] Nifty 500 9,165.5 < SMA200 9,384.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (26)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   108    IT         01-Jan-15   262.2       1,053.7     254    ₹201,030      +301.8%     -9.6%     
  DABUR       13     FMCG       01-Jun-18   354.9       409.8       216    ₹11,845       +15.5%      -2.0%     
  TCS         1      IT         01-Aug-18   1,618.1     1,846.5     51     ₹11,651       +14.1%      +6.3%     
  MPHASIS     12     IT         01-Jun-18   868.5       968.9       88     ₹8,840        +11.6%      -3.5%     
  PAGEIND     9      MFG        01-Aug-18   27,338.4    30,462.0    3      ₹9,371        +11.4%      +0.7%     
  TECHM       10     IT         01-Jun-18   516.6       572.3       148    ₹8,247        +10.8%      +1.7%     
  COFORGE     11     IT         01-Jun-18   201.6       214.9       381    ₹5,089        +6.6%       -6.0%     
  BAJFINANCE  101    FIN SVC    01-Jun-18   201.1       214.1       382    ₹4,988        +6.5%       -10.5%    
  INFY        3      IT         03-Sep-18   577.9       601.6       148    ₹3,521        +4.1%       +3.3%     
  HINDUNILVR  34     FMCG       01-Jun-18   1,385.4     1,442.8     55     ₹3,158        +4.1%       -0.1%     
  RELIANCE    4      OIL&GAS    01-Aug-18   527.8       545.2       158    ₹2,752        +3.3%       -0.5%     
  BATAINDIA   18     CON DUR    01-Aug-18   878.9       903.9       94     ₹2,348        +2.8%       -2.9%     
  GRAPHITE    36     METAL      01-Jun-18   670.8       687.3       114    ₹1,886        +2.5%       -8.3%     
  BRITANNIA   56     FMCG       01-Jun-18   2,571.5     2,595.3     29     ₹691          +0.9%       -2.7%     
  HEG         22     METAL      01-Jun-18   571.2       576.0       134    ₹637          +0.8%       -12.5%    
  SONATSOFTW  7      IT         01-Jun-18   108.8       109.4       706    ₹426          +0.6%       -3.7%     
  JUBLFOOD    46     CONSUMP    01-Jun-18   245.3       243.8       313    ₹-461         -0.6%       -7.2%     
  VIPIND      58     CON DUR    01-Jun-18   398.6       388.9       192    ₹-1,856       -2.4%       -20.3%    
  GLAXO       53     HEALTH     01-Aug-18   1,303.6     1,187.5     63     ₹-7,313       -8.9%       -8.9%     
  TORNTPHARM  21     HEALTH     03-Sep-18   826.6       748.5       104    ₹-8,121       -9.4%       -3.8%     
  RELAXO      44     CON DUR    01-Aug-18   398.8       346.4       209    ₹-10,959      -13.1%      -11.0%    
  BAJAJFINSV  90     FIN SVC    01-Aug-18   696.2       585.8       119    ₹-13,136      -15.9%      -7.3%     
  FSL         86     IT         01-Jun-18   58.2        48.9        1319   ₹-12,344      -16.1%      -9.1%     
  GODREJCP    69     FMCG       03-Sep-18   894.1       745.4       96     ₹-14,276      -16.6%      -4.4%     
  HAVELLS     54     CON DUR    03-Sep-18   680.5       565.4       126    ₹-14,513      -16.9%      -7.2%     
  TASTYBITE   50     FMCG       03-Sep-18   10,408.2    8,046.6     8      ₹-18,892      -22.7%      -12.4%    

  AFTER: Invested ₹2,240,309 | Cash ₹24,675 | Total ₹2,264,984 | Positions 26/30 | Slot ₹75,499

========================================================================
  REBALANCE #47  —  01 Nov 2018
  NAV: ₹2,203,071  |  Slot: ₹73,436  |  Cash: ₹24,675
========================================================================

  [REGIME OFF] Nifty 500 8,772.5 < SMA200 9,309.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (26)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   204    IT         01-Jan-15   262.2       974.8       254    ₹180,999      +271.7% ⚠   +3.0%     
  HEG         14     METAL      01-Jun-18   571.2       741.6       134    ₹22,835       +29.8%      +5.0%     
  GRAPHITE    33     METAL      01-Jun-18   670.8       799.9       114    ₹14,721       +19.3%      +5.3%     
  BAJFINANCE  59     FIN SVC    01-Jun-18   201.1       234.5       382    ₹12,752       +16.6%      +5.0%     
  VIPIND      34     CON DUR    01-Jun-18   398.6       441.2       192    ₹8,186        +10.7%      +4.9%     
  COFORGE     19     IT         01-Jun-18   201.6       220.4       381    ₹7,176        +9.3%       -0.2%     
  BATAINDIA   20     CON DUR    01-Aug-18   878.9       948.3       94     ₹6,518        +7.9%       +8.7%     
  TECHM       5      IT         01-Jun-18   516.6       543.1       148    ₹3,914        +5.1%       +3.1%     
  HINDUNILVR  36     FMCG       01-Jun-18   1,385.4     1,420.4     55     ₹1,927        +2.5%       +1.9%     
  TCS         13     IT         01-Aug-18   1,618.1     1,588.0     51     ₹-1,531       -1.9%       -0.3%     
  DABUR       66     FMCG       01-Jun-18   354.9       345.8       216    ₹-1,978       -2.6%       -8.2%     
  RELAXO      25     CON DUR    01-Aug-18   398.8       385.3       209    ₹-2,825       -3.4%       +5.2%     
  BRITANNIA   108    FMCG       01-Jun-18   2,571.5     2,479.6     29     ₹-2,664       -3.6%       +0.1%     
  PAGEIND     35     MFG        01-Aug-18   27,338.4    26,175.8    3      ₹-3,488       -4.3%       -2.8%     
  INFY        18     IT         03-Sep-18   577.9       543.1       148    ₹-5,148       -6.0%       -1.1%     
  TORNTPHARM  4      HEALTH     03-Sep-18   826.6       775.5       104    ₹-5,309       -6.2%       +4.5%     
  MPHASIS     56     IT         01-Jun-18   868.5       812.7       88     ₹-4,907       -6.4%       -7.1%     
  HAVELLS     42     CON DUR    03-Sep-18   680.5       612.7       126    ₹-8,546       -10.0%      +4.8%     
  GLAXO       120    HEALTH     01-Aug-18   1,303.6     1,164.3     63     ₹-8,775       -10.7%      +1.4%     
  RELIANCE    96     OIL&GAS    01-Aug-18   527.8       467.5       158    ₹-9,518       -11.4%      -3.7%     
  JUBLFOOD    161    CONSUMP    01-Jun-18   245.3       217.0       313    ₹-8,845       -11.5%      -5.0%     
  SONATSOFTW  29     IT         01-Jun-18   108.8       94.6        706    ₹-10,047      -13.1%      -0.4%     
  FSL         68     IT         01-Jun-18   58.2        47.7        1319   ₹-13,909      -18.1%      -0.7%     
  TASTYBITE   27     FMCG       03-Sep-18   10,408.2    8,238.2     8      ₹-17,360      -20.8%      -1.3%     
  BAJAJFINSV  216    FIN SVC    01-Aug-18   696.2       536.5       119    ₹-19,007      -22.9%      -2.9%     
  GODREJCP    187    FMCG       03-Sep-18   894.1       660.0       96     ₹-22,476      -26.2% ⚠    -4.1%     
  ⚠  WAZ < 0 (momentum below universe mean): GODREJCP, TATAELXSI

  AFTER: Invested ₹2,178,396 | Cash ₹24,675 | Total ₹2,203,071 | Positions 26/30 | Slot ₹73,436

========================================================================
  REBALANCE #48  —  03 Dec 2018
  NAV: ₹2,254,621  |  Slot: ₹75,154  |  Cash: ₹24,675
========================================================================

  [REGIME OFF] Nifty 500 9,126.9 < SMA200 9,242.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (26)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   288    IT         01-Jan-15   262.2       935.1       254    ₹170,897      +256.6% ⚠   +1.9%     
  VIPIND      53     CON DUR    01-Jun-18   398.6       514.9       192    ₹22,331       +29.2%      +9.5%     
  HEG         35     METAL      01-Jun-18   571.2       716.1       134    ₹19,414       +25.4%      -1.7%     
  BAJFINANCE  61     FIN SVC    01-Jun-18   201.1       243.3       382    ₹16,106       +21.0%      +4.3%     
  HINDUNILVR  3      FMCG       01-Jun-18   1,385.4     1,613.0     55     ₹12,518       +16.4%      +7.1%     
  GRAPHITE    93     METAL      01-Jun-18   670.8       765.0       114    ₹10,748       +14.1%      -1.8%     
  BATAINDIA   25     CON DUR    01-Aug-18   878.9       993.5       94     ₹10,767       +13.0%      +6.7%     
  DABUR       102    FMCG       01-Jun-18   354.9       386.6       216    ₹6,828        +8.9%       +2.7%     
  BRITANNIA   56     FMCG       01-Jun-18   2,571.5     2,748.7     29     ₹5,140        +6.9%       +4.2%     
  JUBLFOOD    62     CONSUMP    01-Jun-18   245.3       261.4       313    ₹5,025        +6.5%       +10.2%    
  TECHM       20     IT         01-Jun-18   516.6       537.0       148    ₹3,022        +4.0%       +1.4%     
  COFORGE     51     IT         01-Jun-18   201.6       203.2       381    ₹613          +0.8%       -1.8%     
  TCS         12     IT         01-Aug-18   1,618.1     1,626.3     51     ₹421          +0.5%       +3.4%     
  RELIANCE    92     OIL&GAS    01-Aug-18   527.8       511.9       158    ₹-2,511       -3.0%       +2.7%     
  HAVELLS     46     CON DUR    03-Sep-18   680.5       657.5       126    ₹-2,909       -3.4%       +4.1%     
  TORNTPHARM  24     HEALTH     03-Sep-18   826.6       794.1       104    ₹-3,382       -3.9%       +4.5%     
  MPHASIS     88     IT         01-Jun-18   868.5       834.0       88     ₹-3,028       -4.0%       +3.6%     
  INFY        22     IT         03-Sep-18   577.9       546.0       148    ₹-4,708       -5.5%       +2.3%     
  RELAXO      83     CON DUR    01-Aug-18   398.8       369.0       209    ₹-6,232       -7.5%       -1.7%     
  PAGEIND     159    MFG        01-Aug-18   27,338.4    24,355.1    3      ₹-8,950       -10.9% ⚠    -3.0%     
  GLAXO       165    HEALTH     01-Aug-18   1,303.6     1,130.5     63     ₹-10,905      -13.3% ⚠    +1.2%     
  BAJAJFINSV  117    FIN SVC    01-Aug-18   696.2       597.5       119    ₹-11,744      -14.2%      +4.0%     
  SONATSOFTW  75     IT         01-Jun-18   108.8       93.1        706    ₹-11,111      -14.5%      -3.9%     
  GODREJCP    139    FMCG       03-Sep-18   894.1       725.9       96     ₹-16,150      -18.8%      +5.4%     
  TASTYBITE   67     FMCG       03-Sep-18   10,408.2    8,387.2     8      ₹-16,168      -19.4%      +1.2%     
  FSL         134    IT         01-Jun-18   58.2        41.7        1319   ₹-21,787      -28.4%      -2.3%     
  ⚠  WAZ < 0 (momentum below universe mean): PAGEIND, GLAXO, TATAELXSI

  AFTER: Invested ₹2,229,945 | Cash ₹24,675 | Total ₹2,254,621 | Positions 26/30 | Slot ₹75,154

========================================================================
  REBALANCE #49  —  01 Jan 2019
  NAV: ₹2,239,644  |  Slot: ₹74,655  |  Cash: ₹24,675
========================================================================

  [REGIME OFF] Nifty 500 9,197.9 < SMA200 9,229.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (26)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   235    IT         01-Jan-15   262.2       931.2       254    ₹169,929      +255.1% ⚠   +1.4%     
  BAJFINANCE  5      FIN SVC    01-Jun-18   201.1       257.6       382    ₹21,595       +28.1%      +4.3%     
  VIPIND      15     CON DUR    01-Jun-18   398.6       500.8       192    ₹19,627       +25.6%      +0.7%     
  BATAINDIA   3      CON DUR    01-Aug-18   878.9       1,051.6     94     ₹16,232       +19.6%      +3.1%     
  HINDUNILVR  2      FMCG       01-Jun-18   1,385.4     1,591.3     55     ₹11,327       +14.9%      -0.1%     
  HEG         87     METAL      01-Jun-18   571.2       638.6       134    ₹9,033        +11.8%      -4.7%     
  DABUR       69     FMCG       01-Jun-18   354.9       394.9       216    ₹8,628        +11.3%      -1.1%     
  BRITANNIA   7      FMCG       01-Jun-18   2,571.5     2,756.3     29     ₹5,360        +7.2%       +0.5%     
  COFORGE     16     IT         01-Jun-18   201.6       212.0       381    ₹3,991        +5.2%       +1.9%     
  TECHM       30     IT         01-Jun-18   516.6       541.9       148    ₹3,736        +4.9%       +1.6%     
  JUBLFOOD    52     CONSUMP    01-Jun-18   245.3       245.8       313    ₹160          +0.2%       -0.5%     
  TORNTPHARM  28     HEALTH     03-Sep-18   826.6       804.2       104    ₹-2,325       -2.7%       +1.5%     
  HAVELLS     27     CON DUR    03-Sep-18   680.5       660.0       126    ₹-2,585       -3.0%       +1.0%     
  MPHASIS     101    IT         01-Jun-18   868.5       840.0       88     ₹-2,503       -3.3%       +1.6%     
  TCS         51     IT         01-Aug-18   1,618.1     1,561.0     51     ₹-2,910       -3.5%       -1.6%     
  GLAXO       57     HEALTH     01-Aug-18   1,303.6     1,226.2     63     ₹-4,871       -5.9%       +3.7%     
  RELIANCE    113    OIL&GAS    01-Aug-18   527.8       496.2       158    ₹-4,990       -6.0%       +0.2%     
  INFY        53     IT         03-Sep-18   577.9       541.7       148    ₹-5,347       -6.3%       +0.1%     
  BAJAJFINSV  33     FIN SVC    01-Aug-18   696.2       649.6       119    ₹-5,549       -6.7%       +4.8%     
  GRAPHITE    261    METAL      01-Jun-18   670.8       619.5       114    ₹-5,844       -7.6%       -8.5%     
  RELAXO      96     CON DUR    01-Aug-18   398.8       361.6       209    ₹-7,772       -9.3%       +0.2%     
  SONATSOFTW  151    IT         01-Jun-18   108.8       93.3        706    ₹-10,904      -14.2%      -1.3%     
  TASTYBITE   89     FMCG       03-Sep-18   10,408.2    8,880.0     8      ₹-12,226      -14.7%      +1.4%     
  GODREJCP    44     FMCG       03-Sep-18   894.1       759.7       96     ₹-12,903      -15.0%      +1.5%     
  PAGEIND     265    MFG        01-Aug-18   27,338.4    22,837.2    3      ₹-13,504      -16.5% ⚠    -0.2%     
  FSL         190    IT         01-Jun-18   58.2        38.4        1319   ₹-26,118      -34.0% ⚠    -2.3%     
  ⚠  WAZ < 0 (momentum below universe mean): FSL, TATAELXSI, PAGEIND

  AFTER: Invested ₹2,214,969 | Cash ₹24,675 | Total ₹2,239,644 | Positions 26/30 | Slot ₹74,655

========================================================================
  REBALANCE #50  —  01 Feb 2019
  NAV: ₹2,205,980  |  Slot: ₹73,533  |  Cash: ₹24,675
========================================================================

  [REGIME OFF] Nifty 500 9,056.3 < SMA200 9,229.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (26)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   311    IT         01-Jan-15   262.2       806.6       254    ₹138,264      +207.6% ⚠   -4.9%     
  BAJFINANCE  13     FIN SVC    01-Jun-18   201.1       254.9       382    ₹20,541       +26.7%      +2.2%     
  BATAINDIA   4      CON DUR    01-Aug-18   878.9       1,068.3     94     ₹17,803       +21.5%      +1.3%     
  VIPIND      39     CON DUR    01-Jun-18   398.6       481.9       192    ₹16,000       +20.9%      +1.0%     
  COFORGE     20     IT         01-Jun-18   201.6       243.0       381    ₹15,773       +20.5%      +5.6%     
  DABUR       5      FMCG       01-Jun-18   354.9       423.0       216    ₹14,694       +19.2%      +5.0%     
  HINDUNILVR  6      FMCG       01-Jun-18   1,385.4     1,589.7     55     ₹11,240       +14.8%      +1.8%     
  BRITANNIA   1      FMCG       01-Jun-18   2,571.5     2,887.7     29     ₹9,171        +12.3%      +2.4%     
  TECHM       43     IT         01-Jun-18   516.6       562.6       148    ₹6,803        +8.9%       +3.9%     
  JUBLFOOD    25     CONSUMP    01-Jun-18   245.3       266.9       313    ₹6,747        +8.8%       +10.6%    
  INFY        7      IT         03-Sep-18   577.9       620.1       148    ₹6,246        +7.3%       +5.2%     
  RELIANCE    8      OIL&GAS    01-Aug-18   527.8       553.3       158    ₹4,028        +4.8%       +5.0%     
  TCS         21     IT         01-Aug-18   1,618.1     1,668.9     51     ₹2,592        +3.1%       +4.9%     
  HAVELLS     28     CON DUR    03-Sep-18   680.5       700.2       126    ₹2,484        +2.9%       +5.4%     
  TORNTPHARM  42     HEALTH     03-Sep-18   826.6       814.7       104    ₹-1,234       -1.4%       -2.7%     
  MPHASIS     48     IT         01-Jun-18   868.5       843.5       88     ₹-2,198       -2.9%       +6.9%     
  SONATSOFTW  67     IT         01-Jun-18   108.8       100.8       706    ₹-5,642       -7.3%       +3.8%     
  RELAXO      66     CON DUR    01-Aug-18   398.8       366.1       209    ₹-6,841       -8.2%       +1.3%     
  GLAXO       63     HEALTH     01-Aug-18   1,303.6     1,152.2     63     ₹-9,535       -11.6%      -1.2%     
  BAJAJFINSV  26     FIN SVC    01-Aug-18   696.2       607.3       119    ₹-10,587      -12.8%      -2.9%     
  PAGEIND     155    MFG        01-Aug-18   27,338.4    22,094.4    3      ₹-15,732      -19.2%      +3.1%     
  TASTYBITE   130    FMCG       03-Sep-18   10,408.2    8,189.3     8      ₹-17,751      -21.3%      -1.2%     
  HEG         383    METAL      01-Jun-18   571.2       441.4       134    ₹-17,396      -22.7% ⚠    -14.9%    
  GODREJCP    102    FMCG       03-Sep-18   894.1       668.7       96     ₹-21,638      -25.2%      -6.8%     
  FSL         128    IT         01-Jun-18   58.2        40.1        1319   ₹-23,874      -31.1%      +3.7%     
  GRAPHITE    380    METAL      01-Jun-18   670.8       457.1       114    ₹-24,355      -31.9% ⚠    -12.2%    
  ⚠  WAZ < 0 (momentum below universe mean): TATAELXSI, GRAPHITE, HEG

  AFTER: Invested ₹2,181,305 | Cash ₹24,675 | Total ₹2,205,980 | Positions 26/30 | Slot ₹73,533

========================================================================
  REBALANCE #51  —  01 Mar 2019
  NAV: ₹2,163,083  |  Slot: ₹72,103  |  Cash: ₹24,675
========================================================================

  [REGIME OFF] Nifty 500 9,037.5 < SMA200 9,188.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (26)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   269    IT         01-Jan-15   262.2       826.6       254    ₹143,356      +215.2% ⚠   +1.6%     
  BATAINDIA   1      CON DUR    01-Aug-18   878.9       1,195.4     94     ₹29,751       +36.0%      +2.5%     
  BAJFINANCE  12     FIN SVC    01-Jun-18   201.1       258.0       382    ₹21,740       +28.3%      +1.3%     
  TECHM       5      IT         01-Jun-18   516.6       626.8       148    ₹16,304       +21.3%      +3.7%     
  COFORGE     7      IT         01-Jun-18   201.6       243.7       381    ₹16,050       +20.9%      +2.0%     
  DABUR       13     FMCG       01-Jun-18   354.9       416.8       216    ₹13,369       +17.4%      +2.3%     
  HINDUNILVR  32     FMCG       01-Jun-18   1,385.4     1,532.6     55     ₹8,098        +10.6%      -1.6%     
  BRITANNIA   49     FMCG       01-Jun-18   2,571.5     2,733.4     29     ₹4,696        +6.3%       +0.8%     
  INFY        8      IT         03-Sep-18   577.9       607.6       148    ₹4,410        +5.2%       +0.4%     
  RELIANCE    20     OIL&GAS    01-Aug-18   527.8       542.7       158    ₹2,357        +2.8%       -0.4%     
  JUBLFOOD    70     CONSUMP    01-Jun-18   245.3       250.9       313    ₹1,753        +2.3%       -0.7%     
  TCS         29     IT         01-Aug-18   1,618.1     1,640.5     51     ₹1,144        +1.4%       +0.3%     
  VIPIND      149    CON DUR    01-Jun-18   398.6       404.2       192    ₹1,066        +1.4% ⚠     -5.1%     
  MPHASIS     69     IT         01-Jun-18   868.5       863.1       88     ₹-470         -0.6%       +1.5%     
  HAVELLS     34     CON DUR    03-Sep-18   680.5       672.1       126    ₹-1,059       -1.2%       +0.6%     
  TORNTPHARM  35     HEALTH     03-Sep-18   826.6       814.7       104    ₹-1,238       -1.4%       -0.6%     
  SONATSOFTW  60     IT         01-Jun-18   108.8       104.9       706    ₹-2,733       -3.6%       +2.1%     
  BAJAJFINSV  25     FIN SVC    01-Aug-18   696.2       641.5       119    ₹-6,514       -7.9%       +3.1%     
  RELAXO      54     CON DUR    01-Aug-18   398.8       362.6       209    ₹-7,567       -9.1%       -0.2%     
  GLAXO       94     HEALTH     01-Aug-18   1,303.6     1,088.1     63     ₹-13,571      -16.5%      -2.2%     
  TASTYBITE   133    FMCG       03-Sep-18   10,408.2    8,086.1     8      ₹-18,577      -22.3%      +2.2%     
  PAGEIND     207    MFG        01-Aug-18   27,338.4    20,557.8    3      ₹-20,342      -24.8% ⚠    -0.8%     
  GODREJCP    171    FMCG       03-Sep-18   894.1       640.9       96     ₹-24,308      -28.3% ⚠    -0.9%     
  HEG         388    METAL      01-Jun-18   571.2       362.8       134    ₹-27,936      -36.5% ⚠    -8.1%     
  FSL         188    IT         01-Jun-18   58.2        35.4        1319   ₹-30,187      -39.3% ⚠    +1.5%     
  GRAPHITE    385    METAL      01-Jun-18   670.8       347.2       114    ₹-36,886      -48.2% ⚠    -7.7%     
  ⚠  WAZ < 0 (momentum below universe mean): VIPIND, GODREJCP, FSL, PAGEIND, TATAELXSI, GRAPHITE, HEG

  AFTER: Invested ₹2,138,407 | Cash ₹24,675 | Total ₹2,163,083 | Positions 26/30 | Slot ₹72,103

========================================================================
  REBALANCE #52  —  01 Apr 2019
  NAV: ₹2,265,030  |  Slot: ₹75,501  |  Cash: ₹24,675
========================================================================

  EXITS (14)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TATAELXSI   283    IT         01-Jan-15   262.2       871.4       254    ₹154,725      +232.3%   1551d 
  VIPIND      94     CON DUR    01-Jun-18   398.6       472.4       192    ₹14,176       +18.5%    304d  
  HINDUNILVR  113    FMCG       01-Jun-18   1,385.4     1,493.2     55     ₹5,928        +7.8%     304d  
  DABUR       135    FMCG       01-Jun-18   354.9       375.3       216    ₹4,402        +5.7%     304d  
  BRITANNIA   83     FMCG       01-Jun-18   2,571.5     2,709.0     29     ₹3,990        +5.4%     304d  
  SONATSOFTW  97     IT         01-Jun-18   108.8       105.4       706    ₹-2,417       -3.1%     304d  
  MPHASIS     134    IT         01-Jun-18   868.5       824.3       88     ₹-3,888       -5.1%     304d  
  PAGEIND     85     MFG        01-Aug-18   27,338.4    23,593.7    3      ₹-11,234      -13.7%    243d  
  TASTYBITE   176    FMCG       03-Sep-18   10,408.2    8,556.1     8      ₹-14,816      -17.8%    210d  
  GLAXO       220    HEALTH     01-Aug-18   1,303.6     1,063.8     63     ₹-15,103      -18.4%    243d  
  GODREJCP    302    FMCG       03-Sep-18   894.1       638.9       96     ₹-24,503      -28.5%    210d  
  FSL         175    IT         01-Jun-18   58.2        38.3        1319   ₹-26,274      -34.2%    304d  
  HEG         384    METAL      01-Jun-18   571.2       368.1       134    ₹-27,224      -35.6%    304d  
  GRAPHITE    381    METAL      01-Jun-18   670.8       378.1       114    ₹-33,367      -43.6%    304d  

  ENTRIES (15)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ASTRAZEN    2      HEALTH     3.661    0.16   +112.3%   +33.7%    1,979.9     38     ₹75,235       +3.2%     
  PRAJIND     4      ENERGY     3.008    0.16   +89.4%    +46.7%    139.6       540    ₹75,390       +2.0%     
  IPCALAB     6      HEALTH     2.957    0.44   +46.5%    +24.1%    474.3       159    ₹75,407       +7.8%     
  PGHL        7      HEALTH     3.396    0.49   +151.9%   +28.6%    2,796.3     27     ₹75,499       +11.1%    
  AXISBANK    8      PVT BNK    2.887    0.16   +45.5%    +24.0%    761.5       99     ₹75,386       +2.8%     
  TITAN       9      CON DUR    2.779    -0.06  +29.0%    +26.2%    1,094.0     69     ₹75,486       +2.9%     
  MUTHOOTFIN  10     FIN SVC    2.729    0.08   +55.7%    +24.5%    537.4       140    ₹75,234       +5.2%     
  DIVISLAB    11     HEALTH     2.736    0.02   +59.9%    +17.1%    1,641.9     45     ₹73,884       +3.0%     
  GODFRYPHLP  12     FMCG       2.671    0.50   +40.3%    +33.2%    341.3       221    ₹75,419       +8.8%     
  MANAPPURAM  13     FIN SVC    2.738    0.23   +22.5%    +39.6%    109.8       687    ₹75,447       +4.7%     
  HDFCBANK    14     PVT BNK    2.613    0.14   +25.2%    +9.8%     534.0       141    ₹75,294       +3.5%     
  HDFC        16     PVT BNK    2.613    0.14   +25.2%    +9.8%     534.0       141    ₹75,294       +3.5%     
  PIIND       17     MFG        2.634    0.09   +26.6%    +21.1%    1,014.5     74     ₹75,075       +4.2%     
  JBCHEPHARM  19     HEALTH     2.594    0.28   +27.0%    +23.5%    169.1       446    ₹75,426       +8.6%     
  UPL         21     MFG        2.440    0.10   +32.3%    +24.6%    580.5       130    ₹75,465       +3.7%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   1      CON DUR    01-Aug-18   878.9       1,295.6     94     ₹39,170       +47.4%      +3.3%     
  BAJFINANCE  5      FIN SVC    01-Jun-18   201.1       291.0       382    ₹34,359       +44.7%      +5.5%     
  COFORGE     44     IT         01-Jun-18   201.6       245.6       381    ₹16,790       +21.9%      +1.1%     
  JUBLFOOD    52     CONSUMP    01-Jun-18   245.3       287.0       313    ₹13,043       +17.0%      +5.1%     
  RELIANCE    3      OIL&GAS    01-Aug-18   527.8       616.1       158    ₹13,952       +16.7%      +4.9%     
  TECHM       51     IT         01-Jun-18   516.6       591.9       148    ₹11,141       +14.6%      -0.5%     
  HAVELLS     22     CON DUR    03-Sep-18   680.5       736.8       126    ₹7,092        +8.3%       +4.1%     
  INFY        15     IT         03-Sep-18   577.9       618.5       148    ₹6,010        +7.0%       +3.3%     
  TORNTPHARM  18     HEALTH     03-Sep-18   826.6       866.8       104    ₹4,178        +4.9%       +2.5%     
  TCS         24     IT         01-Aug-18   1,618.1     1,670.3     51     ₹2,663        +3.2%       +1.5%     
  BAJAJFINSV  34     FIN SVC    01-Aug-18   696.2       713.0       119    ₹1,999        +2.4%       +5.0%     
  RELAXO      20     CON DUR    01-Aug-18   398.8       398.1       209    ₹-139         -0.2%       +7.1%     

  AFTER: Invested ₹2,258,011 | Cash ₹5,678 | Total ₹2,263,689 | Positions 27/30 | Slot ₹75,501

========================================================================
  REBALANCE #53  —  02 May 2019
  NAV: ₹2,271,146  |  Slot: ₹75,705  |  Cash: ₹5,678
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COFORGE     85     IT         01-Jun-18   201.6       236.2       381    ₹13,188       +17.2%    335d  
  JUBLFOOD    93     CONSUMP    01-Jun-18   245.3       264.5       313    ₹6,006        +7.8%     335d  
  TORNTPHARM  130    HEALTH     03-Sep-18   826.6       807.0       104    ₹-2,040       -2.4%     241d  
  MANAPPURAM  70     FIN SVC    01-Apr-19   109.8       103.1       687    ₹-4,589       -6.1%     31d   
  JBCHEPHARM  97     HEALTH     01-Apr-19   169.1       156.8       446    ₹-5,476       -7.3%     31d   
  GODFRYPHLP  62     FMCG       01-Apr-19   341.3       316.1       221    ₹-5,564       -7.4%     31d   
  PRAJIND     72     ENERGY     01-Apr-19   139.6       124.7       540    ₹-8,054       -10.7%    31d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BALRAMCHIN  3      FMCG       3.850    0.09   +138.1%   +50.0%    144.9       522    ₹75,621       +8.8%     
  ICICIGI     4      FIN SVC    3.241    0.20   +43.7%    +32.7%    1,053.0     71     ₹74,766       +4.0%     
  GILLETTE    5      FMCG       3.174    0.08   +15.6%    +16.3%    6,727.6     11     ₹74,003       +3.3%     
  WIPRO       15     IT         2.544    0.16   +34.3%    +11.7%    134.8       561    ₹75,624       +3.7%     
  GREAVESCOT  19     AUTO       2.514    0.30   +19.9%    +29.2%    144.8       522    ₹75,592       +3.9%     
  SHREECEM    20     INFRA      2.502    0.48   +16.5%    +23.8%    19,375.8    3      ₹58,127       +3.5%     
  HCLTECH     21     IT         2.390    0.22   +14.8%    +23.1%    454.6       166    ₹75,456       +4.3%     

  HOLDS (20)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   1      CON DUR    01-Aug-18   878.9       1,369.8     94     ₹46,141       +55.8%      +3.3%     
  BAJFINANCE  8      FIN SVC    01-Jun-18   201.1       303.7       382    ₹39,186       +51.0%      +3.4%     
  TECHM       22     IT         01-Jun-18   516.6       630.7       148    ₹16,884       +22.1%      +3.8%     
  RELIANCE    12     OIL&GAS    01-Aug-18   527.8       621.9       158    ₹14,876       +17.8%      +3.0%     
  ASTRAZEN    2      HEALTH     01-Apr-19   1,979.9     2,252.7     38     ₹10,367       +13.8%      +5.1%     
  TCS         7      IT         01-Aug-18   1,618.1     1,821.3     51     ₹10,368       +12.6%      +3.9%     
  RELAXO      6      CON DUR    01-Aug-18   398.8       434.5       209    ₹7,453        +8.9%       +3.1%     
  BAJAJFINSV  14     FIN SVC    01-Aug-18   696.2       754.9       119    ₹6,978        +8.4%       +1.9%     
  HAVELLS     38     CON DUR    03-Sep-18   680.5       732.4       126    ₹6,535        +7.6%       +1.2%     
  INFY        44     IT         03-Sep-18   577.9       598.6       148    ₹3,064        +3.6%       -0.8%     
  UPL         16     MFG        01-Apr-19   580.5       598.9       130    ₹2,391        +3.2%       +2.4%     
  HDFCBANK    10     PVT BNK    01-Apr-19   534.0       544.2       141    ₹1,433        +1.9%       +3.3%     
  HDFC        11     PVT BNK    01-Apr-19   534.0       544.2       141    ₹1,433        +1.9%       +3.3%     
  TITAN       23     CON DUR    01-Apr-19   1,094.0     1,111.4     69     ₹1,199        +1.6%       +1.5%     
  PIIND       17     MFG        01-Apr-19   1,014.5     1,018.4     74     ₹286          +0.4%       +1.5%     
  IPCALAB     9      HEALTH     01-Apr-19   474.3       474.4       159    ₹19           +0.0%       +2.2%     
  DIVISLAB    18     HEALTH     01-Apr-19   1,641.9     1,633.3     45     ₹-386         -0.5%       +0.1%     
  MUTHOOTFIN  33     FIN SVC    01-Apr-19   537.4       531.1       140    ₹-884         -1.2%       +0.2%     
  AXISBANK    13     PVT BNK    01-Apr-19   761.5       748.3       99     ₹-1,305       -1.7%       -0.5%     
  PGHL        39     HEALTH     01-Apr-19   2,796.3     2,729.8     27     ₹-1,794       -2.4%       -0.8%     

  AFTER: Invested ₹2,239,971 | Cash ₹30,571 | Total ₹2,270,542 | Positions 27/30 | Slot ₹75,705

========================================================================
  REBALANCE #54  —  03 Jun 2019
  NAV: ₹2,315,378  |  Slot: ₹77,179  |  Cash: ₹30,571
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TCS         49     IT         01-Aug-18   1,618.1     1,843.5     51     ₹11,496       +13.9%    306d  
  TECHM       227    IT         01-Jun-18   516.6       570.9       148    ₹8,029        +10.5%    367d  
  HAVELLS     71     CON DUR    03-Sep-18   680.5       731.4       126    ₹6,410        +7.5%     273d  
  INFY        113    IT         03-Sep-18   577.9       609.9       148    ₹4,743        +5.5%     273d  
  RELAXO      61     CON DUR    01-Aug-18   398.8       416.0       209    ₹3,591        +4.3%     306d  
  GREAVESCOT  52     AUTO       02-May-19   144.8       147.9       522    ₹1,595        +2.1%     32d   
  WIPRO       53     IT         02-May-19   134.8       133.7       561    ₹-605         -0.8%     32d   
  ASTRAZEN    82     HEALTH     01-Apr-19   1,979.9     1,914.8     38     ₹-2,471       -3.3%     63d   
  HCLTECH     119    IT         02-May-19   454.6       437.8       166    ₹-2,781       -3.7%     32d   
  DIVISLAB    96     HEALTH     01-Apr-19   1,641.9     1,539.9     45     ₹-4,590       -6.2%     63d   

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JKCEMENT    2      MFG        3.197    0.52   +13.2%    +47.0%    1,010.0     76     ₹76,757       +8.7%     
  HONAUT      4      MFG        2.996    0.33   +44.5%    +22.9%    26,282.6    2      ₹52,565       +7.2%     
  JUSTDIAL    5      IT         2.936    0.47   +84.6%    +59.4%    795.0       97     ₹77,115       +17.0%    
  GUJGASLTD   7      OIL&GAS    2.874    0.57   +10.2%    +55.1%    177.2       435    ₹77,073       +9.8%     
  ATUL        8      MFG        2.808    0.11   +42.0%    +19.8%    3,884.6     19     ₹73,807       +5.1%     
  HEIDELBERG  10     MFG        2.789    0.62   +48.4%    +37.4%    163.6       471    ₹77,077       +5.3%     
  PNCINFRA    12     INFRA      2.689    0.67   +21.8%    +50.1%    196.5       392    ₹77,010       +13.6%    
  SRF         16     MFG        2.566    0.42   +51.7%    +27.8%    558.0       138    ₹76,997       +2.6%     
  DCMSHRIRAM  18     MFG        2.505    0.79   +96.5%    +34.9%    485.0       159    ₹77,117       -1.9%     
  ULTRACEMCO  21     INFRA      2.424    0.58   +27.1%    +28.1%    4,607.3     16     ₹73,717       +2.8%     
  DEEPAKNTR   22     MFG        2.472    0.62   +32.3%    +40.4%    309.9       249    ₹77,172       +8.2%     

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  6      FIN SVC    01-Jun-18   201.1       340.4       382    ₹53,235       +69.3%      +6.1%     
  BATAINDIA   17     CON DUR    01-Aug-18   878.9       1,272.7     94     ₹37,015       +44.8%      +0.7%     
  BAJAJFINSV  9      FIN SVC    01-Aug-18   696.2       831.5       119    ₹16,095       +19.4%      +4.8%     
  PGHL        3      HEALTH     01-Apr-19   2,796.3     3,295.3     27     ₹13,473       +17.8%      +5.4%     
  RELIANCE    42     OIL&GAS    01-Aug-18   527.8       602.1       158    ₹11,739       +14.1%      +2.7%     
  TITAN       20     CON DUR    01-Apr-19   1,094.0     1,236.5     69     ₹9,830        +13.0%      +5.1%     
  ICICIGI     1      FIN SVC    02-May-19   1,053.0     1,164.0     71     ₹7,875        +10.5%      +8.1%     
  PIIND       15     MFG        01-Apr-19   1,014.5     1,116.0     74     ₹7,509        +10.0%      +4.8%     
  SHREECEM    11     INFRA      02-May-19   19,375.8    21,301.5    3      ₹5,777        +9.9%       +6.1%     
  UPL         28     MFG        01-Apr-19   580.5       630.5       130    ₹6,494        +8.6%       +2.1%     
  MUTHOOTFIN  19     FIN SVC    01-Apr-19   537.4       573.4       140    ₹5,046        +6.7%       +4.1%     
  HDFCBANK    13     PVT BNK    01-Apr-19   534.0       567.6       141    ₹4,732        +6.3%       +3.3%     
  HDFC        14     PVT BNK    01-Apr-19   534.0       567.6       141    ₹4,732        +6.3%       +3.3%     
  AXISBANK    46     PVT BNK    01-Apr-19   761.5       808.3       99     ₹4,633        +6.1%       +4.0%     
  BALRAMCHIN  33     FMCG       02-May-19   144.9       145.9       522    ₹538          +0.7%       +4.1%     
  GILLETTE    23     FMCG       02-May-19   6,727.6     6,604.1     11     ₹-1,358       -1.8%       +3.0%     
  IPCALAB     39     HEALTH     01-Apr-19   474.3       451.0       159    ₹-3,702       -4.9%       -0.9%     

  AFTER: Invested ₹2,286,405 | Cash ₹28,004 | Total ₹2,314,409 | Positions 28/30 | Slot ₹77,179

========================================================================
  REBALANCE #55  —  01 Jul 2019
  NAV: ₹2,302,044  |  Slot: ₹76,735  |  Cash: ₹28,004
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  RELIANCE    124    OIL&GAS    01-Aug-18   527.8       561.6       158    ₹5,350        +6.4%     334d  
  AXISBANK    48     PVT BNK    01-Apr-19   761.5       806.1       99     ₹4,416        +5.9%     91d   
  UPL         60     MFG        01-Apr-19   580.5       593.6       130    ₹1,701        +2.3%     91d   
  JUSTDIAL    45     IT         03-Jun-19   795.0       761.6       97     ₹-3,240       -4.2%     28d   
  IPCALAB     66     HEALTH     01-Apr-19   474.3       452.9       159    ₹-3,391       -4.5%     91d   
  DEEPAKNTR   74     MFG        03-Jun-19   309.9       293.0       249    ₹-4,202       -5.4%     28d   
  GUJGASLTD   73     OIL&GAS    03-Jun-19   177.2       162.9       435    ₹-6,229       -8.1%     28d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  VAIBHAVGBL  8      CON DUR    3.006    0.39   +26.3%    +38.6%    150.6       509    ₹76,642       +10.8%    
  TRENT       10     CONSUMP    2.828    0.45   +41.2%    +21.2%    450.5       170    ₹76,593       +9.2%     
  WIPRO       12     IT         2.747    0.18   +41.9%    +10.6%    129.5       592    ₹76,637       -2.1%     
  SIEMENS     13     ENERGY     2.836    0.62   +33.0%    +25.8%    754.7       101    ₹76,225       +5.3%     
  RELAXO      15     CON DUR    2.720    0.32   +26.3%    +18.2%    439.8       174    ₹76,523       +5.6%     
  ICICIBANK   19     PVT BNK    2.633    0.40   +50.5%    +11.4%    423.6       181    ₹76,664       +2.8%     
  GODREJPROP  21     REALTY     2.661    0.67   +44.5%    +34.6%    1,096.2     69     ₹75,641       +15.2%    

  HOLDS (21)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  3      FIN SVC    01-Jun-18   201.1       358.7       382    ₹60,197       +78.4%      +4.3%     
  BATAINDIA   6      CON DUR    01-Aug-18   878.9       1,351.4     94     ₹44,416       +53.8%      +2.5%     
  PGHL        2      HEALTH     01-Apr-19   2,796.3     3,522.0     27     ₹19,593       +26.0%      +7.7%     
  BAJAJFINSV  9      FIN SVC    01-Aug-18   696.2       848.7       119    ₹18,145       +21.9%      +2.6%     
  TITAN       7      CON DUR    01-Apr-19   1,094.0     1,292.1     69     ₹13,667       +18.1%      +2.8%     
  PIIND       4      MFG        01-Apr-19   1,014.5     1,170.1     74     ₹11,515       +15.3%      +3.5%     
  HDFCBANK    17     PVT BNK    01-Apr-19   534.0       577.7       141    ₹6,162        +8.2%       +2.4%     
  HDFC        18     PVT BNK    01-Apr-19   534.0       577.7       141    ₹6,162        +8.2%       +2.4%     
  SHREECEM    22     INFRA      02-May-19   19,375.8    20,903.6    3      ₹4,583        +7.9%       +1.7%     
  SRF         5      MFG        03-Jun-19   558.0       589.6       138    ₹4,361        +5.7%       +2.4%     
  MUTHOOTFIN  27     FIN SVC    01-Apr-19   537.4       563.4       140    ₹3,646        +4.8%       +0.2%     
  DCMSHRIRAM  11     MFG        03-Jun-19   485.0       507.8       159    ₹3,628        +4.7%       +4.8%     
  PNCINFRA    20     INFRA      03-Jun-19   196.5       201.1       392    ₹1,835        +2.4%       +3.1%     
  ICICIGI     25     FIN SVC    02-May-19   1,053.0     1,055.7     71     ₹188          +0.3%       -2.8%     
  ATUL        1      MFG        03-Jun-19   3,884.6     3,893.8     19     ₹176          +0.2%       +0.8%     
  GILLETTE    14     FMCG       02-May-19   6,727.6     6,690.7     11     ₹-406         -0.5%       +1.7%     
  JKCEMENT    28     MFG        03-Jun-19   1,010.0     971.6       76     ₹-2,918       -3.8%       +0.6%     
  ULTRACEMCO  35     INFRA      03-Jun-19   4,607.3     4,379.3     16     ₹-3,648       -4.9%       -1.0%     
  HEIDELBERG  33     MFG        03-Jun-19   163.6       154.3       471    ₹-4,400       -5.7%       +0.1%     
  HONAUT      16     MFG        03-Jun-19   26,282.6    24,614.7    2      ₹-3,336       -6.3%       +0.5%     
  BALRAMCHIN  40     FMCG       02-May-19   144.9       132.1       522    ₹-6,672       -8.8%       -2.0%     

  AFTER: Invested ₹2,273,554 | Cash ₹27,855 | Total ₹2,301,409 | Positions 28/30 | Slot ₹76,735

========================================================================
  REBALANCE #56  —  01 Aug 2019
  NAV: ₹2,119,282  |  Slot: ₹70,643  |  Cash: ₹27,855
========================================================================

  [REGIME OFF] Nifty 500 8,935.7 < SMA200 9,243.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (28)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  39     FIN SVC    01-Jun-18   201.1       313.0       382    ₹42,741       +55.6%      -3.3%     
  PGHL        1      HEALTH     01-Apr-19   2,796.3     3,897.3     27     ₹29,727       +39.4%      +4.1%     
  BATAINDIA   25     CON DUR    01-Aug-18   878.9       1,221.7     94     ₹32,215       +39.0%      -2.1%     
  PIIND       6      MFG        01-Apr-19   1,014.5     1,093.1     74     ₹5,812        +7.7%       -0.1%     
  ICICIGI     13     FIN SVC    02-May-19   1,053.0     1,101.5     71     ₹3,441        +4.6%       +2.9%     
  VAIBHAVGBL  2      CON DUR    01-Jul-19   150.6       153.8       509    ₹1,636        +2.1%       +4.2%     
  MUTHOOTFIN  16     FIN SVC    01-Apr-19   537.4       544.8       140    ₹1,035        +1.4%       +0.0%     
  BAJAJFINSV  111    FIN SVC    01-Aug-18   696.2       705.6       119    ₹1,116        +1.3%       -4.9%     
  SHREECEM    54     INFRA      02-May-19   19,375.8    19,507.9    3      ₹396          +0.7%       -4.4%     
  HDFCBANK    144    PVT BNK    01-Apr-19   534.0       517.5       141    ₹-2,319       -3.1%       -4.1%     
  HDFC        145    PVT BNK    01-Apr-19   534.0       517.5       141    ₹-2,319       -3.1%       -4.1%     
  WIPRO       72     IT         01-Jul-19   129.5       124.1       592    ₹-3,152       -4.1%       +1.1%     
  ICICIBANK   3      PVT BNK    01-Jul-19   423.6       403.4       181    ₹-3,655       -4.8%       -0.9%     
  RELAXO      88     CON DUR    01-Jul-19   439.8       418.3       174    ₹-3,740       -4.9%       +2.0%     
  GILLETTE    107    FMCG       02-May-19   6,727.6     6,389.0     11     ₹-3,724       -5.0%       +0.1%     
  TITAN       94     CON DUR    01-Apr-19   1,094.0     1,037.1     69     ₹-3,925       -5.2%       -5.1%     
  BALRAMCHIN  14     FMCG       02-May-19   144.9       135.0       522    ₹-5,157       -6.8%       -3.4%     
  TRENT       18     CONSUMP    01-Jul-19   450.5       418.6       170    ₹-5,428       -7.1%       -0.0%     
  SRF         5      MFG        03-Jun-19   558.0       515.0       138    ₹-5,924       -7.7%       -3.8%     
  PNCINFRA    10     INFRA      03-Jun-19   196.5       181.3       392    ₹-5,930       -7.7%       -4.0%     
  JKCEMENT    17     MFG        03-Jun-19   1,010.0     929.0       76     ₹-6,151       -8.0%       -1.6%     
  ATUL        15     MFG        03-Jun-19   3,884.6     3,533.6     19     ₹-6,668       -9.0%       -2.9%     
  HEIDELBERG  31     MFG        03-Jun-19   163.6       146.2       471    ₹-8,205       -10.6%      -5.3%     
  ULTRACEMCO  120    INFRA      03-Jun-19   4,607.3     4,104.5     16     ₹-8,045       -10.9%      -5.1%     
  HONAUT      63     MFG        03-Jun-19   26,282.6    22,664.6    2      ₹-7,236       -13.8%      -0.7%     
  GODREJPROP  22     REALTY     01-Jul-19   1,096.2     941.6       69     ₹-10,671      -14.1%      -0.5%     
  SIEMENS     97     ENERGY     01-Jul-19   754.7       635.0       101    ₹-12,086      -15.9%      -6.4%     
  DCMSHRIRAM  126    MFG        03-Jun-19   485.0       365.4       159    ₹-19,014      -24.7%      -15.6%    

  AFTER: Invested ₹2,091,427 | Cash ₹27,855 | Total ₹2,119,282 | Positions 28/30 | Slot ₹70,643

========================================================================
  REBALANCE #57  —  03 Sep 2019
  NAV: ₹2,108,955  |  Slot: ₹70,299  |  Cash: ₹27,855
========================================================================

  [REGIME OFF] Nifty 500 8,802.3 < SMA200 9,267.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (28)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BATAINDIA   1      CON DUR    01-Aug-18   878.9       1,444.0     94     ₹53,117       +64.3%      +5.2%     
  BAJFINANCE  65     FIN SVC    01-Jun-18   201.1       316.5       382    ₹44,092       +57.4%      -1.0%     
  PGHL        33     HEALTH     01-Apr-19   2,796.3     3,363.9     27     ₹15,326       +20.3%      -5.7%     
  PIIND       6      MFG        01-Apr-19   1,014.5     1,133.0     74     ₹8,768        +11.7%      +2.7%     
  ICICIGI     16     FIN SVC    02-May-19   1,053.0     1,138.3     71     ₹6,056        +8.1%       +0.6%     
  RELAXO      24     CON DUR    01-Jul-19   439.8       452.1       174    ₹2,147        +2.8%       +5.7%     
  MUTHOOTFIN  27     FIN SVC    01-Apr-19   537.4       541.4       140    ₹560          +0.7%       -1.7%     
  BAJAJFINSV  145    FIN SVC    01-Aug-18   696.2       700.1       119    ₹456          +0.5%       -1.8%     
  TRENT       11     CONSUMP    01-Jul-19   450.5       448.1       170    ₹-410         -0.5%       -1.6%     
  GILLETTE    44     FMCG       02-May-19   6,727.6     6,606.9     11     ₹-1,327       -1.8%       +3.6%     
  HDFCBANK    103    PVT BNK    01-Apr-19   534.0       515.0       141    ₹-2,676       -3.6%       -1.1%     
  HDFC        109    PVT BNK    01-Apr-19   534.0       515.0       141    ₹-2,676       -3.6%       -1.1%     
  SRF         36     MFG        03-Jun-19   558.0       532.9       138    ₹-3,456       -4.5%       -2.7%     
  JKCEMENT    46     MFG        03-Jun-19   1,010.0     961.5       76     ₹-3,679       -4.8%       -2.0%     
  TITAN       114    CON DUR    01-Apr-19   1,094.0     1,039.4     69     ₹-3,766       -5.0%       -2.6%     
  VAIBHAVGBL  28     CON DUR    01-Jul-19   150.6       141.9       509    ₹-4,392       -5.7%       -1.1%     
  HONAUT      64     MFG        03-Jun-19   26,282.6    24,236.5    2      ₹-4,092       -7.8%       +3.6%     
  HEIDELBERG  71     MFG        03-Jun-19   163.6       148.2       471    ₹-7,296       -9.5%       -3.0%     
  WIPRO       75     IT         01-Jul-19   129.5       116.4       592    ₹-7,704       -10.1%      -0.5%     
  SHREECEM    194    INFRA      02-May-19   19,375.8    17,370.6    3      ₹-6,016       -10.3% ⚠    -7.2%     
  ICICIBANK   67     PVT BNK    01-Jul-19   423.6       379.3       181    ₹-8,014       -10.5%      -4.4%     
  ATUL        112    MFG        03-Jun-19   3,884.6     3,411.3     19     ₹-8,993       -12.2%      -2.3%     
  SIEMENS     89     ENERGY     01-Jul-19   754.7       656.7       101    ₹-9,894       -13.0%      -0.9%     
  PNCINFRA    132    INFRA      03-Jun-19   196.5       168.2       392    ₹-11,087      -14.4%      -8.7%     
  BALRAMCHIN  39     FMCG       02-May-19   144.9       123.8       522    ₹-11,023      -14.6%      +1.5%     
  GODREJPROP  53     REALTY     01-Jul-19   1,096.2     900.0       69     ₹-13,538      -17.9%      -0.8%     
  ULTRACEMCO  205    INFRA      03-Jun-19   4,607.3     3,768.7     16     ₹-13,417      -18.2% ⚠    -5.7%     
  DCMSHRIRAM  177    MFG        03-Jun-19   485.0       342.7       159    ₹-22,624      -29.3% ⚠    -6.5%     
  ⚠  WAZ < 0 (momentum below universe mean): DCMSHRIRAM, SHREECEM, ULTRACEMCO

  AFTER: Invested ₹2,081,100 | Cash ₹27,855 | Total ₹2,108,955 | Positions 28/30 | Slot ₹70,299

========================================================================
  REBALANCE #58  —  01 Oct 2019
  NAV: ₹2,318,611  |  Slot: ₹77,287  |  Cash: ₹27,855
========================================================================

  [REGIME OFF] Nifty 500 9,236.4 < SMA200 9,278.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (28)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  25     FIN SVC    01-Jun-18   201.1       387.8       382    ₹71,324       +92.9%      +8.0%     
  BATAINDIA   2      CON DUR    01-Aug-18   878.9       1,630.0     94     ₹70,600       +85.5%      +6.5%     
  PGHL        37     HEALTH     01-Apr-19   2,796.3     3,733.5     27     ₹25,306       +33.5%      +1.2%     
  PIIND       8      MFG        01-Apr-19   1,014.5     1,235.2     74     ₹16,327       +21.7%      +0.5%     
  BAJAJFINSV  65     FIN SVC    01-Aug-18   696.2       842.7       119    ₹17,435       +21.0%      +7.9%     
  SIEMENS     9      ENERGY     01-Jul-19   754.7       875.4       101    ₹12,192       +16.0%      +13.8%    
  TITAN       27     CON DUR    01-Apr-19   1,094.0     1,256.2     69     ₹11,189       +14.8%      +6.3%     
  MUTHOOTFIN  30     FIN SVC    01-Apr-19   537.4       600.6       140    ₹8,845        +11.8%      +5.1%     
  RELAXO      24     CON DUR    01-Jul-19   439.8       489.3       174    ₹8,618        +11.3%      +3.1%     
  ICICIGI     35     FIN SVC    02-May-19   1,053.0     1,151.0     71     ₹6,956        +9.3%       +1.4%     
  HDFCBANK    32     PVT BNK    01-Apr-19   534.0       581.8       141    ₹6,739        +9.0%       +5.4%     
  HDFC        33     PVT BNK    01-Apr-19   534.0       581.8       141    ₹6,739        +9.0%       +5.4%     
  HONAUT      19     MFG        03-Jun-19   26,282.6    27,751.2    2      ₹2,937        +5.6%       +4.4%     
  TRENT       34     CONSUMP    01-Jul-19   450.5       475.7       170    ₹4,279        +5.6%       -0.2%     
  BALRAMCHIN  38     FMCG       02-May-19   144.9       149.5       522    ₹2,420        +3.2%       +4.2%     
  JKCEMENT    41     MFG        03-Jun-19   1,010.0     1,010.5     76     ₹42           +0.1%       -0.5%     
  ATUL        50     MFG        03-Jun-19   3,884.6     3,886.1     19     ₹29           +0.0%       +4.3%     
  VAIBHAVGBL  89     CON DUR    01-Jul-19   150.6       147.2       509    ₹-1,723       -2.2%       +0.4%     
  ICICIBANK   64     PVT BNK    01-Jul-19   423.6       410.7       181    ₹-2,334       -3.0%       +0.7%     
  SRF         101    MFG        03-Jun-19   558.0       524.9       138    ₹-4,556       -5.9%       -3.4%     
  PNCINFRA    129    INFRA      03-Jun-19   196.5       184.1       392    ₹-4,860       -6.3%       +0.5%     
  SHREECEM    209    INFRA      02-May-19   19,375.8    18,134.6    3      ₹-3,724       -6.4% ⚠     -2.1%     
  GILLETTE    161    FMCG       02-May-19   6,727.6     6,289.6     11     ₹-4,818       -6.5% ⚠     -0.8%     
  HEIDELBERG  105    MFG        03-Jun-19   163.6       148.6       471    ₹-7,087       -9.2%       -2.7%     
  GODREJPROP  67     REALTY     01-Jul-19   1,096.2     983.2       69     ₹-7,797       -10.3%      +0.6%     
  ULTRACEMCO  157    INFRA      03-Jun-19   4,607.3     4,085.9     16     ₹-8,342       -11.3%      +0.9%     
  WIPRO       258    IT         01-Jul-19   129.5       107.2       592    ₹-13,181      -17.2% ⚠    -4.1%     
  DCMSHRIRAM  220    MFG        03-Jun-19   485.0       362.6       159    ₹-19,458      -25.2% ⚠    -2.9%     
  ⚠  WAZ < 0 (momentum below universe mean): GILLETTE, SHREECEM, DCMSHRIRAM, WIPRO

  AFTER: Invested ₹2,290,756 | Cash ₹27,855 | Total ₹2,318,611 | Positions 28/30 | Slot ₹77,287

========================================================================
  REBALANCE #59  —  01 Nov 2019
  NAV: ₹2,409,292  |  Slot: ₹80,310  |  Cash: ₹27,855
========================================================================

  EXITS (16)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PGHL        138    HEALTH     01-Apr-19   2,796.3     3,796.3     27     ₹26,999       +35.8%    214d  
  BAJAJFINSV  46     FIN SVC    01-Aug-18   696.2       835.4       119    ₹16,562       +20.0%    457d  
  HDFC        65     PVT BNK    01-Apr-19   534.0       577.7       141    ₹6,164        +8.2%     214d  
  HDFCBANK    64     PVT BNK    01-Apr-19   534.0       577.7       141    ₹6,164        +8.2%     214d  
  BALRAMCHIN  104    FMCG       02-May-19   144.9       154.0       522    ₹4,790        +6.3%     183d  
  GILLETTE    59     FMCG       02-May-19   6,727.6     7,107.5     11     ₹4,179        +5.6%     183d  
  ICICIBANK   61     PVT BNK    01-Jul-19   423.6       447.1       181    ₹4,257        +5.6%     123d  
  SRF         63     MFG        03-Jun-19   558.0       567.9       138    ₹1,373        +1.8%     151d  
  SHREECEM    173    INFRA      02-May-19   19,375.8    19,509.7    3      ₹402          +0.7%     183d  
  VAIBHAVGBL  208    CON DUR    01-Jul-19   150.6       139.7       509    ₹-5,526       -7.2%     123d  
  WIPRO       202    IT         01-Jul-19   129.5       118.7       592    ₹-6,359       -8.3%     123d  
  GODREJPROP  78     REALTY     01-Jul-19   1,096.2     1,000.0     69     ₹-6,641       -8.8%     123d  
  HEIDELBERG  193    MFG        03-Jun-19   163.6       149.0       471    ₹-6,917       -9.0%     151d  
  ULTRACEMCO  224    INFRA      03-Jun-19   4,607.3     4,037.6     16     ₹-9,115       -12.4%    151d  
  PNCINFRA    213    INFRA      03-Jun-19   196.5       170.0       392    ₹-10,380      -13.5%    151d  
  DCMSHRIRAM  273    MFG        03-Jun-19   485.0       344.7       159    ₹-22,304      -28.9%    151d  

  ENTRIES (15)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MANAPPURAM  1      FIN SVC    3.923    0.18   +147.1%   +52.5%    152.5       526    ₹80,232       +11.8%    
  BERGEPAINT  2      CON DUR    3.794    0.24   +74.7%    +55.1%    411.5       195    ₹80,239       +6.3%     
  ABBOTINDIA  4      HEALTH     3.426    0.24   +60.5%    +36.5%    10,661.6    7      ₹74,631       +3.8%     
  BPCL        5      OIL&GAS    3.278    0.15   +92.7%    +54.7%    166.5       482    ₹80,276       +2.9%     
  ADANIGREEN  6      ENERGY     3.333    0.11   +128.6%   +81.4%    89.9        892    ₹80,235       +10.8%    
  HINDUNILVR  8      FMCG       3.119    0.13   +42.1%    +26.6%    1,949.5     41     ₹79,930       +4.4%     
  NAM-INDIA   10     FIN SVC    3.147    0.27   +131.0%   +56.5%    293.5       273    ₹80,120       +15.2%    
  SBILIFE     11     FIN SVC    3.088    -0.12  +78.7%    +25.5%    985.5       81     ₹79,824       +6.9%     
  COLPAL      13     FMCG       3.026    0.29   +44.3%    +32.2%    1,309.1     61     ₹79,856       +1.5%     
  GLAXO       14     HEALTH     2.992    0.21   +17.2%    +39.2%    1,392.4     57     ₹79,364       +11.3%    
  NAVINFLUOR  15     MFG        2.943    0.33   +38.5%    +53.3%    869.6       92     ₹80,007       +10.4%    
  IGL         17     OIL&GAS    2.786    0.15   +63.2%    +29.3%    175.0       458    ₹80,150       +5.3%     
  FINEORG     18     MFG        2.798    0.18   +72.6%    +32.4%    1,855.1     43     ₹79,770       +1.9%     
  HDFCLIFE    19     FIN SVC    2.724    0.11   +65.7%    +22.8%    601.3       133    ₹79,976       +1.6%     
  GODFRYPHLP  21     FMCG       2.666    0.37   +52.6%    +58.7%    330.6       242    ₹80,009       +6.0%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  20     FIN SVC    01-Jun-18   201.1       395.3       382    ₹74,198       +96.6%      +2.5%     
  BATAINDIA   3      CON DUR    01-Aug-18   878.9       1,634.0     94     ₹70,980       +85.9%      +0.1%     
  PIIND       12     MFG        01-Apr-19   1,014.5     1,362.2     74     ₹25,730       +34.3%      +2.8%     
  SIEMENS     7      ENERGY     01-Jul-19   754.7       948.2       101    ₹19,548       +25.6%      +4.9%     
  ICICIGI     29     FIN SVC    02-May-19   1,053.0     1,288.4     71     ₹16,709       +22.3%      +4.6%     
  RELAXO      9      CON DUR    01-Jul-19   439.8       534.0       174    ₹16,402       +21.4%      +4.5%     
  TRENT       16     CONSUMP    01-Jul-19   450.5       544.1       170    ₹15,903       +20.8%      +5.0%     
  MUTHOOTFIN  28     FIN SVC    01-Apr-19   537.4       633.0       140    ₹13,388       +17.8%      +4.7%     
  TITAN       26     CON DUR    01-Apr-19   1,094.0     1,277.0     69     ₹12,630       +16.7%      +0.5%     
  ATUL        31     MFG        03-Jun-19   3,884.6     4,234.9     19     ₹6,656        +9.0%       +4.3%     
  JKCEMENT    43     MFG        03-Jun-19   1,010.0     1,082.0     76     ₹5,477        +7.1%       +1.9%     
  HONAUT      22     MFG        03-Jun-19   26,282.6    28,037.0    2      ₹3,509        +6.7%       +0.7%     

  AFTER: Invested ₹2,368,214 | Cash ₹39,660 | Total ₹2,407,873 | Positions 27/30 | Slot ₹80,310

========================================================================
  REBALANCE #60  —  02 Dec 2019
  NAV: ₹2,407,049  |  Slot: ₹80,235  |  Cash: ₹39,660
========================================================================

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TRENT       44     CONSUMP    01-Jul-19   450.5       523.2       170    ₹12,347       +16.1%    154d  
  MUTHOOTFIN  76     FIN SVC    01-Apr-19   537.4       604.4       140    ₹9,382        +12.5%    245d  
  TITAN       133    CON DUR    01-Apr-19   1,094.0     1,131.8     69     ₹2,607        +3.5%     245d  
  HONAUT      60     MFG        03-Jun-19   26,282.6    26,924.4    2      ₹1,284        +2.4%     182d  
  ATUL        65     MFG        03-Jun-19   3,884.6     3,949.5     19     ₹1,233        +1.7%     182d  
  NAVINFLUOR  67     MFG        01-Nov-19   869.6       833.8       92     ₹-3,294       -4.1%     31d   
  HINDUNILVR  53     FMCG       01-Nov-19   1,949.5     1,827.7     41     ₹-4,993       -6.2%     31d   
  HDFCLIFE    69     FIN SVC    01-Nov-19   601.3       558.8       133    ₹-5,652       -7.1%     31d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  LALPATHLAB  5      HEALTH     3.260    0.18   +93.3%    +36.6%    760.0       105    ₹79,796       +2.5%     
  EPL         6      MFG        3.146    0.13   +77.3%    +66.9%    133.7       599    ₹80,113       +8.8%     
  ASTRAZEN    11     HEALTH     2.873    0.19   +68.8%    +51.9%    2,769.7     28     ₹77,553       +6.8%     
  MCX         14     FIN SVC    2.621    0.19   +67.3%    +28.9%    215.7       372    ₹80,233       +0.4%     
  ALKYLAMINE  15     MFG        2.592    0.12   +42.8%    +42.6%    411.5       194    ₹79,831       +5.4%     
  BAJAJFINSV  16     FIN SVC    2.560    0.28   +57.2%    +25.6%    891.6       89     ₹79,349       +0.1%     
  GUJGASLTD   17     OIL&GAS    2.594    0.08   +74.3%    +21.1%    206.7       388    ₹80,181       +8.8%     
  RELIANCE    18     OIL&GAS    2.476    0.16   +41.6%    +24.4%    706.5       113    ₹79,830       +4.8%     

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BAJFINANCE  28     FIN SVC    01-Jun-18   201.1       383.7       382    ₹69,767       +90.8%      -3.4%     
  BATAINDIA   27     CON DUR    01-Aug-18   878.9       1,526.7     94     ₹60,888       +73.7%      -2.5%     
  ADANIGREEN  1      ENERGY     01-Nov-19   89.9        131.3       892    ₹36,884       +46.0%      +21.4%    
  PIIND       4      MFG        01-Apr-19   1,014.5     1,478.3     74     ₹34,320       +45.7%      +5.3%     
  RELAXO      8      CON DUR    01-Jul-19   439.8       575.8       174    ₹23,673       +30.9%      +3.8%     
  ICICIGI     33     FIN SVC    02-May-19   1,053.0     1,314.4     71     ₹18,558       +24.8%      +2.1%     
  GODFRYPHLP  23     FMCG       01-Nov-19   330.6       410.8       242    ₹19,414       +24.3%      +13.5%    
  SIEMENS     22     ENERGY     01-Jul-19   754.7       841.1       101    ₹8,723        +11.4%      -3.9%     
  JKCEMENT    34     MFG        03-Jun-19   1,010.0     1,123.3     76     ₹8,610        +11.2%      +0.4%     
  ABBOTINDIA  2      HEALTH     01-Nov-19   10,661.6    11,580.2    7      ₹6,430        +8.6%       +2.8%     
  IGL         10     OIL&GAS    01-Nov-19   175.0       185.5       458    ₹4,799        +6.0%       +2.4%     
  GLAXO       3      HEALTH     01-Nov-19   1,392.4     1,389.4     57     ₹-171         -0.2%       +2.2%     
  NAM-INDIA   13     FIN SVC    01-Nov-19   293.5       290.5       273    ₹-822         -1.0%       -0.4%     
  BERGEPAINT  7      CON DUR    01-Nov-19   411.5       400.4       195    ₹-2,162       -2.7%       +1.3%     
  BPCL        9      OIL&GAS    01-Nov-19   166.5       160.3       482    ₹-2,991       -3.7%       -2.0%     
  FINEORG     21     MFG        01-Nov-19   1,855.1     1,768.5     43     ₹-3,723       -4.7%       -5.9%     
  COLPAL      30     FMCG       01-Nov-19   1,309.1     1,248.1     61     ₹-3,719       -4.7%       -3.8%     
  SBILIFE     20     FIN SVC    01-Nov-19   985.5       932.5       81     ₹-4,293       -5.4%       -1.4%     
  MANAPPURAM  12     FIN SVC    01-Nov-19   152.5       139.6       526    ₹-6,794       -8.5%       -1.9%     

  AFTER: Invested ₹2,397,765 | Cash ₹8,527 | Total ₹2,406,293 | Positions 27/30 | Slot ₹80,235

========================================================================
  REBALANCE #61  —  01 Jan 2020
  NAV: ₹2,509,725  |  Slot: ₹83,658  |  Cash: ₹8,527
========================================================================

  EXITS (11)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJFINANCE  44     FIN SVC    01-Jun-18   201.1       411.0       382    ₹80,188       +104.4%   579d  
  BATAINDIA   45     CON DUR    01-Aug-18   878.9       1,637.8     94     ₹71,337       +86.3%    518d  
  GODFRYPHLP  43     FMCG       01-Nov-19   330.6       383.6       242    ₹12,815       +16.0%    61d   
  SIEMENS     76     ENERGY     01-Jul-19   754.7       845.9       101    ₹9,213        +12.1%    184d  
  BAJAJFINSV  40     FIN SVC    02-Dec-19   891.6       934.4       89     ₹3,814        +4.8%     30d   
  FINEORG     36     MFG        01-Nov-19   1,855.1     1,872.5     43     ₹748          +0.9%     61d   
  LALPATHLAB  42     HEALTH     02-Dec-19   760.0       731.3       105    ₹-3,014       -3.8%     30d   
  COLPAL      176    FMCG       01-Nov-19   1,309.1     1,256.4     61     ₹-3,215       -4.0%     61d   
  RELIANCE    41     OIL&GAS    02-Dec-19   706.5       672.2       113    ₹-3,869       -4.8%     30d   
  BPCL        81     OIL&GAS    01-Nov-19   166.5       157.7       482    ₹-4,289       -5.3%     61d   
  ASTRAZEN    35     HEALTH     02-Dec-19   2,769.7     2,579.8     28     ₹-5,318       -6.9%     30d   

  ENTRIES (12)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  CRISIL      6      MNC        2.872    0.20   +24.4%    +43.9%    1,762.0     47     ₹82,812       +9.4%     
  NH          8      HEALTH     3.023    0.17   +64.2%    +37.8%    321.9       259    ₹83,377       +6.8%     
  DIXON       9      CON DUR    2.978    0.12   +86.6%    +32.5%    754.7       110    ₹83,020       +4.0%     
  APLAPOLLO   10     METAL      2.842    0.37   +56.2%    +37.1%    186.7       448    ₹83,642       +11.6%    
  AVANTIFEED  12     FMCG       2.774    0.75   +57.1%    +56.8%    564.6       148    ₹83,559       +9.5%     
  NAVINFLUOR  13     MFG        2.734    0.43   +42.4%    +33.6%    980.3       85     ₹83,323       +4.0%     
  IPCALAB     18     HEALTH     2.634    0.09   +46.7%    +23.3%    556.7       150    ₹83,512       +0.8%     
  JBCHEPHARM  19     HEALTH     2.582    -0.06  +50.3%    +20.8%    199.1       420    ₹83,601       +2.2%     
  COROMANDEL  20     MFG        2.554    -0.01  +20.4%    +28.8%    492.0       170    ₹83,639       +3.3%     
  BHARTIARTL  21     CONSUMP    2.559    -0.28  +55.3%    +29.8%    433.3       193    ₹83,630       +1.7%     
  GMMPFAUDLR  22     MFG        2.499    0.03   +53.8%    +26.1%    619.8       134    ₹83,057       +6.7%     
  SRF         23     MFG        2.500    -0.09  +55.7%    +25.0%    673.0       124    ₹83,447       +3.6%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  1      ENERGY     01-Nov-19   89.9        174.8       892    ₹75,642       +94.3%      +22.5%    
  PIIND       16     MFG        01-Apr-19   1,014.5     1,424.0     74     ₹30,301       +40.4%      -0.7%     
  RELAXO      4      CON DUR    01-Jul-19   439.8       603.5       174    ₹28,478       +37.2%      +1.3%     
  ICICIGI     33     FIN SVC    02-May-19   1,053.0     1,328.8     71     ₹19,576       +26.2%      +0.4%     
  GUJGASLTD   2      OIL&GAS    02-Dec-19   206.7       242.9       388    ₹14,060       +17.5%      +12.3%    
  EPL         11     MFG        02-Dec-19   133.7       155.3       599    ₹12,913       +16.1%      +7.9%     
  JKCEMENT    30     MFG        03-Jun-19   1,010.0     1,144.0     76     ₹10,188       +13.3%      +1.5%     
  ABBOTINDIA  3      HEALTH     01-Nov-19   10,661.6    12,056.1    7      ₹9,761        +13.1%      +1.3%     
  IGL         5      OIL&GAS    01-Nov-19   175.0       189.6       458    ₹6,699        +8.4%       +0.7%     
  MANAPPURAM  7      FIN SVC    01-Nov-19   152.5       157.7       526    ₹2,704        +3.4%       +3.6%     
  ALKYLAMINE  27     MFG        02-Dec-19   411.5       424.5       194    ₹2,518        +3.2%       +4.0%     
  MCX         29     FIN SVC    02-Dec-19   215.7       221.1       372    ₹2,004        +2.5%       +2.3%     
  BERGEPAINT  17     CON DUR    01-Nov-19   411.5       418.7       195    ₹1,412        +1.8%       +2.1%     
  NAM-INDIA   15     FIN SVC    01-Nov-19   293.5       292.5       273    ₹-259         -0.3%       +1.4%     
  SBILIFE     14     FIN SVC    01-Nov-19   985.5       964.6       81     ₹-1,689       -2.1%       +0.3%     
  GLAXO       31     HEALTH     01-Nov-19   1,392.4     1,359.1     57     ₹-1,894       -2.4%       -0.2%     

  AFTER: Invested ₹2,471,308 | Cash ₹37,229 | Total ₹2,508,537 | Positions 28/30 | Slot ₹83,658

========================================================================
  REBALANCE #62  —  03 Feb 2020
  NAV: ₹2,744,244  |  Slot: ₹91,475  |  Cash: ₹37,229
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ICICIGI     82     FIN SVC    02-May-19   1,053.0     1,247.4     71     ₹13,801       +18.5%    277d  
  MCX         77     FIN SVC    02-Dec-19   215.7       221.1       372    ₹2,004        +2.5%     63d   
  GLAXO       113    HEALTH     01-Nov-19   1,392.4     1,371.2     57     ₹-1,206       -1.5%     94d   
  SBILIFE     112    FIN SVC    01-Nov-19   985.5       901.7       81     ₹-6,790       -8.5%     94d   
  CRISIL      108    MNC        01-Jan-20   1,762.0     1,536.9     47     ₹-10,578      -12.8%    33d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  AUBANK      6      PVT BNK    3.425    0.36   +71.7%    +52.0%    519.4       176    ₹91,410       +9.5%     
  AMBER       7      CON DUR    3.422    0.59   +74.6%    +52.6%    1,532.1     59     ₹90,397       +12.6%    
  NESCO       12     CONSUMP    3.022    0.31   +63.8%    +31.3%    715.7       127    ₹90,894       +2.6%     
  FINEORG     14     MFG        2.963    0.35   +106.1%   +22.9%    2,317.9     39     ₹90,397       +7.2%     

  HOLDS (23)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  1      ENERGY     01-Nov-19   89.9        196.1       892    ₹94,730       +118.1%     +4.1%     
  RELAXO      4      CON DUR    01-Jul-19   439.8       706.9       174    ₹46,477       +60.7%      +5.2%     
  PIIND       24     MFG        01-Apr-19   1,014.5     1,520.1     74     ₹37,410       +49.8%      +3.5%     
  GMMPFAUDLR  2      MFG        01-Jan-20   619.8       923.4       134    ₹40,676       +49.0%      +21.3%    
  GUJGASLTD   5      OIL&GAS    02-Dec-19   206.7       269.8       388    ₹24,495       +30.5%      +2.3%     
  JKCEMENT    20     MFG        03-Jun-19   1,010.0     1,316.9     76     ₹23,324       +30.4%      +2.6%     
  IGL         8      OIL&GAS    01-Nov-19   175.0       227.3       458    ₹23,936       +29.9%      +6.7%     
  ALKYLAMINE  11     MFG        02-Dec-19   411.5       521.0       194    ₹21,242       +26.6%      +5.9%     
  DIXON       3      CON DUR    01-Jan-20   754.7       946.1       110    ₹21,048       +25.4%      +11.2%    
  COROMANDEL  10     MFG        01-Jan-20   492.0       585.9       170    ₹15,970       +19.1%      +6.6%     
  NAVINFLUOR  17     MFG        01-Jan-20   980.3       1,143.1     85     ₹13,841       +16.6%      +5.8%     
  EPL         26     MFG        02-Dec-19   133.7       155.1       599    ₹12,781       +16.0%      -0.2%     
  JBCHEPHARM  9      HEALTH     01-Jan-20   199.1       228.6       420    ₹12,419       +14.9%      +3.2%     
  BHARTIARTL  18     CONSUMP    01-Jan-20   433.3       487.6       193    ₹10,470       +12.5%      +4.0%     
  BERGEPAINT  21     CON DUR    01-Nov-19   411.5       461.5       195    ₹9,752        +12.2%      +4.1%     
  NH          16     HEALTH     01-Jan-20   321.9       356.2       259    ₹8,888        +10.7%      +1.9%     
  AVANTIFEED  25     FMCG       01-Jan-20   564.6       622.8       148    ₹8,611        +10.3%      +1.2%     
  SRF         13     MFG        01-Jan-20   673.0       741.4       124    ₹8,488        +10.2%      +4.2%     
  ABBOTINDIA  38     HEALTH     01-Nov-19   10,661.6    11,694.8    7      ₹7,232        +9.7%       +0.5%     
  MANAPPURAM  43     FIN SVC    01-Nov-19   152.5       163.8       526    ₹5,909        +7.4%       +1.0%     
  IPCALAB     33     HEALTH     01-Jan-20   556.7       583.3       150    ₹3,980        +4.8%       -1.9%     
  APLAPOLLO   22     METAL      01-Jan-20   186.7       189.6       448    ₹1,279        +1.5%       -1.4%     
  NAM-INDIA   48     FIN SVC    01-Nov-19   293.5       285.9       273    ₹-2,069       -2.6%       -1.8%     

  AFTER: Invested ₹2,675,881 | Cash ₹67,931 | Total ₹2,743,812 | Positions 27/30 | Slot ₹91,475

========================================================================
  REBALANCE #63  —  02 Mar 2020
  NAV: ₹2,676,975  |  Slot: ₹89,233  |  Cash: ₹67,931
========================================================================

  [REGIME OFF] Nifty 500 9,178.8 < SMA200 9,535.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (27)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  2      ENERGY     01-Nov-19   89.9        156.1       892    ₹58,961       +73.5%      -15.0%    
  RELAXO      15     CON DUR    01-Jul-19   439.8       679.2       174    ₹41,664       +54.4%      -7.6%     
  GMMPFAUDLR  1      MFG        01-Jan-20   619.8       946.0       134    ₹43,710       +52.6%      -4.3%     
  PIIND       52     MFG        01-Apr-19   1,014.5     1,496.3     74     ₹35,652       +47.5%      -0.7%     
  ALKYLAMINE  9      MFG        02-Dec-19   411.5       589.8       194    ₹34,587       +43.3%      -0.6%     
  NAVINFLUOR  3      MFG        01-Jan-20   980.3       1,346.9     85     ₹31,163       +37.4%      +7.1%     
  JKCEMENT    13     MFG        03-Jun-19   1,010.0     1,363.5     76     ₹26,868       +35.0%      -0.5%     
  ABBOTINDIA  5      HEALTH     01-Nov-19   10,661.6    14,123.8    7      ₹24,235       +32.5%      +2.8%     
  GUJGASLTD   8      OIL&GAS    02-Dec-19   206.7       265.9       388    ₹22,978       +28.7%      -2.5%     
  IPCALAB     20     HEALTH     01-Jan-20   556.7       704.0       150    ₹22,089       +26.4%      +6.4%     
  JBCHEPHARM  14     HEALTH     01-Jan-20   199.1       246.6       420    ₹19,990       +23.9%      -0.9%     
  COROMANDEL  24     MFG        01-Jan-20   492.0       572.6       170    ₹13,708       +16.4%      +0.1%     
  BHARTIARTL  39     CONSUMP    01-Jan-20   433.3       495.5       193    ₹12,001       +14.4%      -2.1%     
  SRF         25     MFG        01-Jan-20   673.0       765.8       124    ₹11,512       +13.8%      -2.5%     
  NAM-INDIA   26     FIN SVC    01-Nov-19   293.5       327.3       273    ₹9,224        +11.5%      -0.8%     
  AUBANK      10     PVT BNK    03-Feb-20   519.4       575.0       176    ₹9,789        +10.7%      +2.7%     
  IGL         67     OIL&GAS    01-Nov-19   175.0       192.8       458    ₹8,139        +10.2%      -7.9%     
  BERGEPAINT  18     CON DUR    01-Nov-19   411.5       452.2       195    ₹7,941        +9.9%       -1.8%     
  DIXON       48     CON DUR    01-Jan-20   754.7       784.3       110    ₹3,257        +3.9%       -7.6%     
  EPL         109    MFG        02-Dec-19   133.7       136.7       599    ₹1,792        +2.2%       -14.0%    
  NH          64     HEALTH     01-Jan-20   321.9       325.8       259    ₹1,010        +1.2%       -4.1%     
  APLAPOLLO   53     METAL      01-Jan-20   186.7       178.7       448    ₹-3,566       -4.3%       -8.8%     
  MANAPPURAM  95     FIN SVC    01-Nov-19   152.5       143.2       526    ₹-4,888       -6.1%       -6.3%     
  NESCO       36     CONSUMP    03-Feb-20   715.7       669.4       127    ₹-5,884       -6.5%       -5.1%     
  FINEORG     23     MFG        03-Feb-20   2,317.9     2,107.3     39     ₹-8,214       -9.1%       -4.1%     
  AMBER       11     CON DUR    03-Feb-20   1,532.1     1,373.3     59     ₹-9,374       -10.4%      -7.9%     
  AVANTIFEED  172    FMCG       01-Jan-20   564.6       400.5       148    ₹-24,293      -29.1% ⚠    -22.4%    
  ⚠  WAZ < 0 (momentum below universe mean): AVANTIFEED

  AFTER: Invested ₹2,609,044 | Cash ₹67,931 | Total ₹2,676,975 | Positions 27/30 | Slot ₹89,233

========================================================================
  REBALANCE #64  —  01 Apr 2020
  NAV: ₹2,194,144  |  Slot: ₹73,138  |  Cash: ₹67,931
========================================================================

  [REGIME OFF] Nifty 500 6,761.9 < SMA200 9,328.1 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (27)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  1      ENERGY     01-Nov-19   89.9        152.9       892    ₹56,151       +70.0%      +3.3%     
  ABBOTINDIA  2      HEALTH     01-Nov-19   10,661.6    14,325.7    7      ₹25,648       +34.4%      +6.4%     
  GMMPFAUDLR  3      MFG        01-Jan-20   619.8       826.2       134    ₹27,659       +33.3%      -0.4%     
  RELAXO      12     CON DUR    01-Jul-19   439.8       574.3       174    ₹23,409       +30.6%      -4.6%     
  IPCALAB     5      HEALTH     01-Jan-20   556.7       686.1       150    ₹19,409       +23.2%      +4.0%     
  NAVINFLUOR  7      MFG        01-Jan-20   980.3       1,180.3     85     ₹17,004       +20.4%      -2.7%     
  JBCHEPHARM  6      HEALTH     01-Jan-20   199.1       238.8       420    ₹16,697       +20.0%      -1.9%     
  PIIND       57     MFG        01-Apr-19   1,014.5     1,175.0     74     ₹11,876       +15.8%      -3.7%     
  ALKYLAMINE  17     MFG        02-Dec-19   411.5       440.8       194    ₹5,683        +7.1%       -9.8%     
  GUJGASLTD   20     OIL&GAS    02-Dec-19   206.7       216.2       388    ₹3,700        +4.6%       -6.3%     
  EPL         33     MFG        02-Dec-19   133.7       135.8       599    ₹1,212        +1.5%       -2.5%     
  COROMANDEL  24     MFG        01-Jan-20   492.0       497.7       170    ₹965          +1.2%       -3.2%     
  IGL         29     OIL&GAS    01-Nov-19   175.0       174.2       458    ₹-388         -0.5%       +2.5%     
  BERGEPAINT  13     CON DUR    01-Nov-19   411.5       392.5       195    ₹-3,696       -4.6%       +0.5%     
  DIXON       25     CON DUR    01-Jan-20   754.7       713.0       110    ₹-4,586       -5.5%       -1.5%     
  BHARTIARTL  23     CONSUMP    01-Jan-20   433.3       402.8       193    ₹-5,895       -7.0%       -8.0%     
  JKCEMENT    77     MFG        03-Jun-19   1,010.0     901.1       76     ₹-8,271       -10.8%      -13.6%    
  AMBER       11     CON DUR    03-Feb-20   1,532.1     1,267.9     59     ₹-15,591      -17.2%      +1.2%     
  FINEORG     15     MFG        03-Feb-20   2,317.9     1,858.4     39     ₹-17,918      -19.8%      -3.8%     
  SRF         67     MFG        01-Jan-20   673.0       520.3       124    ₹-18,925      -22.7%      -14.5%    
  NH          69     HEALTH     01-Jan-20   321.9       247.5       259    ₹-19,285      -23.1%      -5.8%     
  NAM-INDIA   54     FIN SVC    01-Nov-19   293.5       215.1       273    ₹-21,386      -26.7%      -6.1%     
  APLAPOLLO   143    METAL      01-Jan-20   186.7       121.6       448    ₹-29,187      -34.9%      -12.6%    
  NESCO       97     CONSUMP    03-Feb-20   715.7       451.9       127    ₹-33,506      -36.9%      -12.4%    
  MANAPPURAM  162    FIN SVC    01-Nov-19   152.5       83.7        526    ₹-36,220      -45.1%      -16.9%    
  AVANTIFEED  218    FMCG       01-Jan-20   564.6       266.8       148    ₹-44,078      -52.8% ⚠    -14.8%    
  AUBANK      161    PVT BNK    03-Feb-20   519.4       239.5       176    ₹-49,262      -53.9% ⚠    -31.6%    
  ⚠  WAZ < 0 (momentum below universe mean): AUBANK, AVANTIFEED

  AFTER: Invested ₹2,126,213 | Cash ₹67,931 | Total ₹2,194,144 | Positions 27/30 | Slot ₹73,138

========================================================================
  REBALANCE #65  —  04 May 2020
  NAV: ₹2,578,632  |  Slot: ₹85,954  |  Cash: ₹67,931
========================================================================

  [REGIME OFF] Nifty 500 7,596.9 < SMA200 9,130.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (27)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  1      ENERGY     01-Nov-19   89.9        209.6       892    ₹106,683      +133.0%     +10.6%    
  GMMPFAUDLR  3      MFG        01-Jan-20   619.8       1,162.7     134    ₹72,749       +87.6%      +9.1%     
  ALKYLAMINE  6      MFG        02-Dec-19   411.5       689.4       194    ₹53,911       +67.5%      +11.0%    
  ABBOTINDIA  2      HEALTH     01-Nov-19   10,661.6    16,694.5    7      ₹42,230       +56.6%      +8.5%     
  NAVINFLUOR  4      MFG        01-Jan-20   980.3       1,487.6     85     ₹43,122       +51.8%      +3.8%     
  PIIND       24     MFG        01-Apr-19   1,014.5     1,489.1     74     ₹35,117       +46.8%      +5.5%     
  IPCALAB     5      HEALTH     01-Jan-20   556.7       792.6       150    ₹35,372       +42.4%      +4.5%     
  RELAXO      52     CON DUR    01-Jul-19   439.8       597.4       174    ₹27,420       +35.8%      -0.8%     
  JBCHEPHARM  13     HEALTH     01-Jan-20   199.1       259.9       420    ₹25,565       +30.6%      +1.6%     
  DIXON       15     CON DUR    01-Jan-20   754.7       915.4       110    ₹17,669       +21.3%      +12.4%    
  IGL         33     OIL&GAS    01-Nov-19   175.0       209.4       458    ₹15,777       +19.7%      +5.4%     
  BHARTIARTL  18     CONSUMP    01-Jan-20   433.3       509.0       193    ₹14,612       +17.5%      +8.1%     
  EPL         38     MFG        02-Dec-19   133.7       155.3       599    ₹12,913       +16.1%      +3.0%     
  GUJGASLTD   45     OIL&GAS    02-Dec-19   206.7       239.5       388    ₹12,728       +15.9%      -0.2%     
  COROMANDEL  40     MFG        01-Jan-20   492.0       529.7       170    ₹6,407        +7.7%       +2.7%     
  JKCEMENT    79     MFG        03-Jun-19   1,010.0     1,073.3     76     ₹4,818        +6.3%       +0.6%     
  SRF         27     MFG        01-Jan-20   673.0       713.5       124    ₹5,027        +6.0%       +6.1%     
  BERGEPAINT  42     CON DUR    01-Nov-19   411.5       390.6       195    ₹-4,068       -5.1%       -4.8%     
  FINEORG     36     MFG        03-Feb-20   2,317.9     1,942.5     39     ₹-14,639      -16.2%      -1.9%     
  NH          88     HEALTH     01-Jan-20   321.9       267.8       259    ₹-14,026      -16.8%      -2.0%     
  MANAPPURAM  120    FIN SVC    01-Nov-19   152.5       108.2       526    ₹-23,308      -29.1%      +5.5%     
  AMBER       74     CON DUR    03-Feb-20   1,532.1     1,080.1     59     ₹-26,671      -29.5%      -3.4%     
  APLAPOLLO   187    METAL      01-Jan-20   186.7       128.5       448    ₹-26,096      -31.2% ⚠    +1.0%     
  NAM-INDIA   91     FIN SVC    01-Nov-19   293.5       196.3       273    ₹-26,527      -33.1%      -8.3%     
  AVANTIFEED  146    FMCG       01-Jan-20   564.6       377.6       148    ₹-27,681      -33.1%      +2.5%     
  NESCO       219    CONSUMP    03-Feb-20   715.7       422.3       127    ₹-37,257      -41.0% ⚠    -6.1%     
  AUBANK      252    PVT BNK    03-Feb-20   519.4       257.2       176    ₹-46,136      -50.5% ⚠    -6.4%     
  ⚠  WAZ < 0 (momentum below universe mean): APLAPOLLO, NESCO, AUBANK

  AFTER: Invested ₹2,510,701 | Cash ₹67,931 | Total ₹2,578,632 | Positions 27/30 | Slot ₹85,954

========================================================================
  REBALANCE #66  —  01 Jun 2020
  NAV: ₹2,771,789  |  Slot: ₹92,393  |  Cash: ₹67,931
========================================================================

  [REGIME OFF] Nifty 500 8,020.1 < SMA200 8,960.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (27)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  1      ENERGY     01-Nov-19   89.9        260.8       892    ₹152,354      +189.9%     +13.0%    
  GMMPFAUDLR  2      MFG        01-Jan-20   619.8       1,272.7     134    ₹87,479       +105.3%     +6.6%     
  ALKYLAMINE  7      MFG        02-Dec-19   411.5       782.5       194    ₹71,976       +90.2%      +10.0%    
  JBCHEPHARM  3      HEALTH     01-Jan-20   199.1       324.7       420    ₹52,792       +63.1%      +8.5%     
  RELAXO      27     CON DUR    01-Jul-19   439.8       698.2       174    ₹44,957       +58.8%      +11.4%    
  NAVINFLUOR  4      MFG        01-Jan-20   980.3       1,541.6     85     ₹47,716       +57.3%      +5.3%     
  PIIND       31     MFG        01-Apr-19   1,014.5     1,552.3     74     ₹39,795       +53.0%      +4.1%     
  ABBOTINDIA  5      HEALTH     01-Nov-19   10,661.6    15,437.7    7      ₹33,432       +44.8%      -0.9%     
  IPCALAB     18     HEALTH     01-Jan-20   556.7       749.6       150    ₹28,922       +34.6%      -2.1%     
  DIXON       10     CON DUR    01-Jan-20   754.7       1,008.1     110    ₹27,872       +33.6%      +13.3%    
  BHARTIARTL  95     CONSUMP    01-Jan-20   433.3       534.4       193    ₹19,510       +23.3%      +1.2%     
  COROMANDEL  15     MFG        01-Jan-20   492.0       605.6       170    ₹19,316       +23.1%      +4.7%     
  IGL         26     OIL&GAS    01-Nov-19   175.0       209.8       458    ₹15,950       +19.9%      +2.0%     
  JKCEMENT    100    MFG        03-Jun-19   1,010.0     1,180.3     76     ₹12,945       +16.9%      +7.7%     
  GUJGASLTD   67     OIL&GAS    02-Dec-19   206.7       233.9       388    ₹10,582       +13.2%      +0.9%     
  EPL         68     MFG        02-Dec-19   133.7       150.8       599    ₹10,225       +12.8%      +0.0%     
  SRF         69     MFG        01-Jan-20   673.0       726.6       124    ₹6,646        +8.0%       +6.3%     
  BERGEPAINT  47     CON DUR    01-Nov-19   411.5       397.9       195    ₹-2,651       -3.3%       +4.6%     
  NH          62     HEALTH     01-Jan-20   321.9       292.8       259    ₹-7,547       -9.1%       +9.4%     
  AMBER       39     CON DUR    03-Feb-20   1,532.1     1,338.6     59     ₹-11,422      -12.6%      +13.6%    
  FINEORG     65     MFG        03-Feb-20   2,317.9     1,926.1     39     ₹-15,281      -16.9%      -0.0%     
  APLAPOLLO   154    METAL      01-Jan-20   186.7       153.4       448    ₹-14,912      -17.8% ⚠    +15.0%    
  MANAPPURAM  130    FIN SVC    01-Nov-19   152.5       117.8       526    ₹-18,290      -22.8%      +9.3%     
  NAM-INDIA   135    FIN SVC    01-Nov-19   293.5       224.5       273    ₹-18,821      -23.5%      +10.0%    
  AVANTIFEED  90     FMCG       01-Jan-20   564.6       414.4       148    ₹-22,225      -26.6%      +8.0%     
  NESCO       250    CONSUMP    03-Feb-20   715.7       424.8       127    ₹-36,949      -40.7% ⚠    +4.0%     
  AUBANK      356    PVT BNK    03-Feb-20   519.4       204.0       176    ₹-55,506      -60.7% ⚠    -3.8%     
  ⚠  WAZ < 0 (momentum below universe mean): APLAPOLLO, NESCO, AUBANK

  AFTER: Invested ₹2,703,858 | Cash ₹67,931 | Total ₹2,771,789 | Positions 27/30 | Slot ₹92,393

========================================================================
  REBALANCE #67  —  01 Jul 2020
  NAV: ₹2,981,182  |  Slot: ₹99,373  |  Cash: ₹67,931
========================================================================

  [REGIME OFF] Nifty 500 8,554.7 < SMA200 8,895.9 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (27)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  2      ENERGY     01-Nov-19   89.9        341.0       892    ₹223,892      +279.0%     -5.7%     
  GMMPFAUDLR  6      MFG        01-Jan-20   619.8       1,382.7     134    ₹102,225      +123.1%     +0.9%     
  ALKYLAMINE  8      MFG        02-Dec-19   411.5       841.0       194    ₹83,331       +104.4%     +4.5%     
  NAVINFLUOR  22     MFG        01-Jan-20   980.3       1,657.5     85     ₹57,562       +69.1%      +6.3%     
  JBCHEPHARM  19     HEALTH     01-Jan-20   199.1       333.5       420    ₹56,471       +67.5%      +1.7%     
  DIXON       7      CON DUR    01-Jan-20   754.7       1,172.2     110    ₹45,926       +55.3%      +8.4%     
  GUJGASLTD   26     OIL&GAS    02-Dec-19   206.7       311.5       388    ₹40,664       +50.7%      +11.1%    
  PIIND       57     MFG        01-Apr-19   1,014.5     1,508.1     74     ₹36,525       +48.7%      -0.9%     
  IPCALAB     34     HEALTH     01-Jan-20   556.7       809.5       150    ₹37,913       +45.4%      +1.6%     
  COROMANDEL  13     MFG        01-Jan-20   492.0       700.0       170    ₹35,357       +42.3%      +5.6%     
  RELAXO      81     CON DUR    01-Jul-19   439.8       617.6       174    ₹30,934       +40.4%      -3.0%     
  ABBOTINDIA  45     HEALTH     01-Nov-19   10,661.6    14,456.3    7      ₹26,563       +35.6%      -2.0%     
  JKCEMENT    43     MFG        03-Jun-19   1,010.0     1,352.0     76     ₹25,996       +33.9%      +6.8%     
  BHARTIARTL  368    CONSUMP    01-Jan-20   433.3       535.5       193    ₹19,713       +23.6%      -0.4%     
  EPL         135    MFG        02-Dec-19   133.7       158.4       599    ₹14,758       +18.4%      -0.2%     
  IGL         91     OIL&GAS    01-Nov-19   175.0       195.6       458    ₹9,446        +11.8%      -3.5%     
  SRF         106    MFG        01-Jan-20   673.0       705.2       124    ₹3,992        +4.8%       -0.3%     
  BERGEPAINT  98     CON DUR    01-Nov-19   411.5       401.3       195    ₹-1,978       -2.5%       -1.4%     
  AMBER       71     CON DUR    03-Feb-20   1,532.1     1,442.2     59     ₹-5,307       -5.9%       -0.7%     
  MANAPPURAM  92     FIN SVC    01-Nov-19   152.5       141.8       526    ₹-5,638       -7.0%       +7.5%     
  NAM-INDIA   127    FIN SVC    01-Nov-19   293.5       262.7       273    ₹-8,397       -10.5%      +8.1%     
  APLAPOLLO   167    METAL      01-Jan-20   186.7       159.8       448    ₹-12,051      -14.4% ⚠    +3.3%     
  AVANTIFEED  52     FMCG       01-Jan-20   564.6       466.3       148    ₹-14,549      -17.4%      +5.1%     
  NH          237    HEALTH     01-Jan-20   321.9       261.4       259    ₹-15,680      -18.8% ⚠    -3.8%     
  FINEORG     212    MFG        03-Feb-20   2,317.9     1,838.1     39     ₹-18,712      -20.7% ⚠    -3.0%     
  NESCO       327    CONSUMP    03-Feb-20   715.7       415.0       127    ₹-38,189      -42.0% ⚠    -2.9%     
  AUBANK      278    PVT BNK    03-Feb-20   519.4       277.9       176    ₹-42,507      -46.5% ⚠    +8.9%     
  ⚠  WAZ < 0 (momentum below universe mean): APLAPOLLO, FINEORG, NH, AUBANK, NESCO

  AFTER: Invested ₹2,913,251 | Cash ₹67,931 | Total ₹2,981,182 | Positions 27/30 | Slot ₹99,373

========================================================================
  REBALANCE #68  —  03 Aug 2020
  NAV: ₹3,168,256  |  Slot: ₹105,609  |  Cash: ₹67,931
========================================================================

  EXITS (15)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PIIND       64     MFG        01-Apr-19   1,014.5     1,809.4     74     ₹58,820       +78.3%    490d  
  ABBOTINDIA  133    HEALTH     01-Nov-19   10,661.6    14,688.6    7      ₹28,188       +37.8%    276d  
  RELAXO      166    CON DUR    01-Jul-19   439.8       584.3       174    ₹25,154       +32.9%    399d  
  BHARTIARTL  390    CONSUMP    01-Jan-20   433.3       523.2       193    ₹17,351       +20.7%    215d  
  SRF         156    MFG        01-Jan-20   673.0       751.3       124    ₹9,714        +11.6%    215d  
  BERGEPAINT  76     CON DUR    01-Nov-19   411.5       426.3       195    ₹2,898        +3.6%     276d  
  IGL         292    OIL&GAS    01-Nov-19   175.0       175.4       458    ₹194          +0.2%     276d  
  APLAPOLLO   71     METAL      01-Jan-20   186.7       186.2       448    ₹-234         -0.3%     215d  
  FINEORG     106    MFG        03-Feb-20   2,317.9     2,183.3     39     ₹-5,249       -5.8%     182d  
  MANAPPURAM  126    FIN SVC    01-Nov-19   152.5       142.6       526    ₹-5,213       -6.5%     276d  
  NH          175    HEALTH     01-Jan-20   321.9       292.2       259    ₹-7,688       -9.2%     215d  
  AVANTIFEED  144    FMCG       01-Jan-20   564.6       433.8       148    ₹-19,351      -23.2%    215d  
  NAM-INDIA   199    FIN SVC    01-Nov-19   293.5       220.6       273    ₹-19,900      -24.8%    276d  
  AUBANK      104    PVT BNK    03-Feb-20   519.4       354.9       176    ₹-28,939      -31.7%    182d  
  NESCO       286    CONSUMP    03-Feb-20   715.7       422.1       127    ₹-37,288      -41.0%    182d  

  ENTRIES (12)
  [52w filter blocked 2: ALOKINDS(-37.6%), ZENSARTECH(-25.7%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  AARTIDRUGS  2      HEALTH     5.471    0.61   +261.3%   +146.9%   429.0       246    ₹105,532      +14.0%    
  LAURUSLABS  4      HEALTH     4.383    0.60   +197.3%   +104.3%   197.1       535    ₹105,473      +39.6%    
  PERSISTENT  5      IT         4.068    0.41   +71.8%    +108.7%   466.7       226    ₹105,476      +21.7%    
  GRANULES    6      HEALTH     3.851    0.81   +250.9%   +88.6%    293.4       359    ₹105,347      +13.5%    
  MASTEK      8      IT         3.442    0.64   +62.6%    +168.0%   613.2       172    ₹105,479      +35.5%    
  POLYMED     9      HEALTH     3.235    0.67   +135.0%   +97.7%    408.3       258    ₹105,343      +10.2%    
  DHANUKA     10     FMCG       3.041    0.70   +119.2%   +84.9%    781.7       135    ₹105,527      +1.9%     
  JUBLPHARMA  11     HEALTH     3.444    0.79   +100.3%   +127.5%   618.2       170    ₹105,091      +16.9%    
  HAL         12     DEFENCE    2.643    0.34   +40.8%    +77.5%    399.6       264    ₹105,498      +1.5%     
  CDSL        13     FIN SVC    2.586    0.67   +77.5%    +58.2%    159.5       662    ₹105,606      +5.4%     
  MUTHOOTFIN  16     FIN SVC    2.446    1.12   +115.6%   +64.0%    1,172.9     90     ₹105,561      +4.0%     
  MPHASIS     17     IT         2.432    0.54   +30.2%    +66.8%    1,024.9     103    ₹105,569      +9.8%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  1      ENERGY     01-Nov-19   89.9        339.8       892    ₹222,866      +277.8%     -2.2%     
  GMMPFAUDLR  21     MFG        01-Jan-20   619.8       1,355.4     134    ₹98,565       +118.7%     -0.2%     
  ALKYLAMINE  18     MFG        02-Dec-19   411.5       893.0       194    ₹93,420       +117.0%     +1.4%     
  DIXON       6      CON DUR    01-Jan-20   754.7       1,590.2     110    ₹91,904       +110.7%     +14.6%    
  NAVINFLUOR  19     MFG        01-Jan-20   980.3       1,726.7     85     ₹63,447       +76.1%      +1.1%     
  JBCHEPHARM  26     HEALTH     01-Jan-20   199.1       344.7       420    ₹61,153       +73.1%      +1.5%     
  IPCALAB     38     HEALTH     01-Jan-20   556.7       927.0       150    ₹55,533       +66.5%      +7.8%     
  EPL         48     MFG        02-Dec-19   133.7       215.5       599    ₹48,967       +61.1%      +21.6%    
  COROMANDEL  14     MFG        01-Jan-20   492.0       737.5       170    ₹41,736       +49.9%      +1.3%     
  JKCEMENT    39     MFG        03-Jun-19   1,010.0     1,471.4     76     ₹35,067       +45.7%      +3.2%     
  GUJGASLTD   42     OIL&GAS    02-Dec-19   206.7       291.7       388    ₹33,005       +41.2%      +3.6%     
  AMBER       15     CON DUR    03-Feb-20   1,532.1     1,722.2     59     ₹11,213       +12.4%      +14.4%    

  AFTER: Invested ₹3,110,043 | Cash ₹56,710 | Total ₹3,166,753 | Positions 24/30 | Slot ₹105,609

========================================================================
  REBALANCE #69  —  01 Sep 2020
  NAV: ₹3,573,432  |  Slot: ₹119,114  |  Cash: ₹56,710
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  JBCHEPHARM  59     HEALTH     01-Jan-20   199.1       360.0       420    ₹67,580       +80.8%    244d  
  JKCEMENT    54     MFG        03-Jun-19   1,010.0     1,465.0     76     ₹34,584       +45.1%    456d  
  GUJGASLTD   98     OIL&GAS    02-Dec-19   206.7       299.4       388    ₹36,002       +44.9%    274d  
  MPHASIS     73     IT         03-Aug-20   1,024.9     1,009.9     103    ₹-1,554       -1.5%     29d   
  JUBLPHARMA  62     HEALTH     03-Aug-20   618.2       563.4       170    ₹-9,305       -8.9%     29d   
  MUTHOOTFIN  58     FIN SVC    03-Aug-20   1,172.9     1,060.1     90     ₹-10,150      -9.6%     29d   

  ENTRIES (6)
  [52w filter blocked 1: CYIENT(-22.1%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  CGPOWER     4      ENERGY     4.810    0.34   +120.7%   +238.1%   21.0        5661   ₹119,113      +29.1%    
  BSOFT       10     IT         3.075    0.99   +179.0%   +110.7%   157.4       756    ₹118,964      +6.8%     
  ADANIENT    12     METAL      2.712    0.98   +115.6%   +89.9%    280.3       424    ₹118,844      +16.4%    
  EMAMILTD    14     FMCG       2.442    0.69   +26.9%    +93.7%    323.5       368    ₹119,038      +9.1%     
  UFLEX       15     MFG        2.437    0.89   +59.6%    +75.8%    320.2       372    ₹119,108      +0.1%     
  DIVISLAB    17     HEALTH     2.410    0.48   +107.1%   +33.5%    3,134.3     38     ₹119,103      +5.3%     

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  1      ENERGY     01-Nov-19   89.9        494.6       892    ₹360,992      +449.9%     +22.0%    
  ALKYLAMINE  5      MFG        02-Dec-19   411.5       1,234.1     194    ₹159,580      +199.9%     +6.2%     
  GMMPFAUDLR  16     MFG        01-Jan-20   619.8       1,788.8     134    ₹156,645      +188.6%     -2.6%     
  DIXON       7      CON DUR    01-Jan-20   754.7       1,655.4     110    ₹99,078       +119.3%     +3.6%     
  NAVINFLUOR  29     MFG        01-Jan-20   980.3       1,921.0     85     ₹79,959       +96.0%      -2.3%     
  EPL         8      MFG        02-Dec-19   133.7       253.2       599    ₹71,545       +89.3%      +4.3%     
  IPCALAB     24     HEALTH     01-Jan-20   556.7       989.1       150    ₹64,851       +77.7%      +2.4%     
  AARTIDRUGS  2      HEALTH     03-Aug-20   429.0       683.7       246    ₹62,667       +59.4%      +8.0%     
  COROMANDEL  31     MFG        01-Jan-20   492.0       699.9       170    ₹35,346       +42.3%      -4.6%     
  CDSL        13     FIN SVC    03-Aug-20   159.5       192.7       662    ₹21,964       +20.8%      +7.5%     
  AMBER       50     CON DUR    03-Feb-20   1,532.1     1,790.8     59     ₹15,257       +16.9%      +2.5%     
  LAURUSLABS  3      HEALTH     03-Aug-20   197.1       219.9       535    ₹12,180       +11.5%      +5.3%     
  GRANULES    6      HEALTH     03-Aug-20   293.4       318.1       359    ₹8,844        +8.4%       +5.3%     
  MASTEK      9      IT         03-Aug-20   613.2       660.6       172    ₹8,153        +7.7%       +2.1%     
  HAL         35     DEFENCE    03-Aug-20   399.6       398.7       264    ₹-230         -0.2%       -16.3%    
  PERSISTENT  11     IT         03-Aug-20   466.7       460.0       226    ₹-1,508       -1.4%       -2.0%     
  POLYMED     34     HEALTH     03-Aug-20   408.3       396.8       258    ₹-2,975       -2.8%       -2.6%     
  DHANUKA     27     FMCG       03-Aug-20   781.7       737.6       135    ₹-5,953       -5.6%       -5.7%     

  AFTER: Invested ₹3,556,974 | Cash ₹15,609 | Total ₹3,572,584 | Positions 24/30 | Slot ₹119,114

========================================================================
  REBALANCE #70  —  01 Oct 2020
  NAV: ₹3,993,095  |  Slot: ₹133,103  |  Cash: ₹15,609
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  GMMPFAUDLR  89     MFG        01-Jan-20   619.8       1,270.6     134    ₹87,207       +105.0%   274d  
  COROMANDEL  53     MFG        01-Jan-20   492.0       749.7       170    ₹43,801       +52.4%    274d  
  DHANUKA     58     FMCG       03-Aug-20   781.7       733.9       135    ₹-6,454       -6.1%     59d   
  HAL         270    DEFENCE    03-Aug-20   399.6       357.3       264    ₹-11,180      -10.6%    59d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ADVENZYMES  11     HEALTH     2.788    0.63   +100.4%   +96.7%    316.9       419    ₹132,788      +26.1%    
  JBCHEPHARM  13     HEALTH     2.731    0.64   +187.2%   +37.9%    468.5       284    ₹133,062      +6.0%     
  FSL         14     IT         2.628    0.99   +59.1%    +88.9%    63.0        2113   ₹133,094      +3.7%     

  HOLDS (20)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  1      ENERGY     01-Nov-19   89.9        747.1       892    ₹586,178      +730.6%     +18.2%    
  ALKYLAMINE  9      MFG        02-Dec-19   411.5       1,267.0     194    ₹165,959      +207.9%     +2.1%     
  DIXON       10     CON DUR    01-Jan-20   754.7       1,766.2     110    ₹111,261      +134.0%     -0.7%     
  NAVINFLUOR  27     MFG        01-Jan-20   980.3       2,095.0     85     ₹94,752       +113.7%     +3.6%     
  IPCALAB     17     HEALTH     01-Jan-20   556.7       1,106.3     150    ₹82,435       +98.7%      +7.1%     
  AARTIDRUGS  2      HEALTH     03-Aug-20   429.0       795.5       246    ₹90,152       +85.4%      +10.9%    
  EPL         31     MFG        02-Dec-19   133.7       223.0       599    ₹53,493       +66.8%      -2.8%     
  CDSL        8      FIN SVC    03-Aug-20   159.5       230.8       662    ₹47,169       +44.7%      +5.1%     
  LAURUSLABS  3      HEALTH     03-Aug-20   197.1       274.5       535    ₹41,397       +39.2%      +4.9%     
  PERSISTENT  4      IT         03-Aug-20   466.7       623.6       226    ₹35,449       +33.6%      +10.5%    
  AMBER       29     CON DUR    03-Feb-20   1,532.1     2,043.8     59     ₹30,190       +33.4%      +4.4%     
  MASTEK      7      IT         03-Aug-20   613.2       817.2       172    ₹35,072       +33.3%      +4.6%     
  GRANULES    5      HEALTH     03-Aug-20   293.4       379.2       359    ₹30,798       +29.2%      +5.9%     
  BSOFT       6      IT         01-Sep-20   157.4       181.7       756    ₹18,432       +15.5%      +4.7%     
  CGPOWER     19     ENERGY     01-Sep-20   21.0        24.3        5661   ₹18,175       +15.3%      +8.2%     
  POLYMED     23     HEALTH     03-Aug-20   408.3       462.1       258    ₹13,891       +13.2%      +1.2%     
  ADANIENT    12     METAL      01-Sep-20   280.3       307.3       424    ₹11,436       +9.6%       +8.0%     
  UFLEX       38     MFG        01-Sep-20   320.2       311.6       372    ₹-3,195       -2.7%       -0.2%     
  EMAMILTD    47     FMCG       01-Sep-20   323.5       314.3       368    ₹-3,384       -2.8%       -2.4%     
  DIVISLAB    16     HEALTH     01-Sep-20   3,134.3     2,973.1     38     ₹-6,126       -5.1%       -1.9%     

  AFTER: Invested ₹3,885,336 | Cash ₹107,286 | Total ₹3,992,622 | Positions 23/30 | Slot ₹133,103

========================================================================
  REBALANCE #71  —  02 Nov 2020
  NAV: ₹4,053,930  |  Slot: ₹135,131  |  Cash: ₹107,286
========================================================================

  EXITS (2)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  UFLEX       78     MFG        01-Sep-20   320.2       315.1       372    ₹-1,895       -1.6%     62d   
  FSL         66     IT         01-Oct-20   63.0        59.9        2113   ₹-6,447       -4.8%     32d   

  ENTRIES (2)
  [52w filter blocked 1: ALOKINDS(-63.2%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BLUEDART    5      INFRA      3.570    0.50   +60.9%    +94.5%    3,775.6     35     ₹132,147      +19.3%    
  THYROCARE   7      HEALTH     3.172    0.57   +111.4%   +71.7%    335.9       402    ₹135,046      +8.8%     

  HOLDS (21)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  1      ENERGY     01-Nov-19   89.9        859.2       892    ₹686,171      +855.2%     +16.0%    
  ALKYLAMINE  13     MFG        02-Dec-19   411.5       1,126.4     194    ₹138,692      +173.7%     -4.9%     
  DIXON       12     CON DUR    01-Jan-20   754.7       1,897.4     110    ₹125,697      +151.4%     +2.9%     
  NAVINFLUOR  19     MFG        01-Jan-20   980.3       2,138.5     85     ₹98,449       +118.2%     +3.2%     
  IPCALAB     20     HEALTH     01-Jan-20   556.7       1,130.2     150    ₹86,013       +103.0%     +4.7%     
  EPL         57     MFG        02-Dec-19   133.7       224.7       599    ₹54,505       +68.0%      -0.9%     
  AARTIDRUGS  4      HEALTH     03-Aug-20   429.0       643.3       246    ₹52,724       +50.0%      -11.4%    
  LAURUSLABS  3      HEALTH     03-Aug-20   197.1       294.6       535    ₹52,137       +49.4%      -5.0%     
  CGPOWER     2      ENERGY     01-Sep-20   21.0        30.0        5661   ₹50,609       +42.5%      +15.1%    
  CDSL        16     FIN SVC    03-Aug-20   159.5       222.1       662    ₹41,435       +39.2%      +0.3%     
  AMBER       34     CON DUR    03-Feb-20   1,532.1     2,100.8     59     ₹33,550       +37.1%      -2.5%     
  MASTEK      18     IT         03-Aug-20   613.2       835.7       172    ₹38,263       +36.3%      +5.7%     
  GRANULES    17     HEALTH     03-Aug-20   293.4       363.1       359    ₹25,015       +23.7%      -2.8%     
  ADANIENT    8      METAL      01-Sep-20   280.3       340.1       424    ₹25,367       +21.3%      +7.6%     
  PERSISTENT  23     IT         03-Aug-20   466.7       539.9       226    ₹16,549       +15.7%      -6.3%     
  POLYMED     30     HEALTH     03-Aug-20   408.3       469.7       258    ₹15,844       +15.0%      -1.9%     
  BSOFT       15     IT         01-Sep-20   157.4       163.0       756    ₹4,273        +3.6%       -6.9%     
  JBCHEPHARM  6      HEALTH     01-Oct-20   468.5       470.4       284    ₹530          +0.4%       -0.8%     
  EMAMILTD    28     FMCG       01-Sep-20   323.5       314.1       368    ₹-3,433       -2.9%       -2.1%     
  DIVISLAB    27     HEALTH     01-Sep-20   3,134.3     2,957.3     38     ₹-6,725       -5.6%       -2.2%     
  ADVENZYMES  25     HEALTH     01-Oct-20   316.9       287.1       419    ₹-12,479      -9.4%       -3.9%     

  AFTER: Invested ₹3,969,977 | Cash ₹83,636 | Total ₹4,053,613 | Positions 23/30 | Slot ₹135,131

========================================================================
  REBALANCE #72  —  01 Dec 2020
  NAV: ₹4,761,185  |  Slot: ₹158,706  |  Cash: ₹83,636
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  IPCALAB     59     HEALTH     01-Jan-20   556.7       1,120.5     150    ₹84,558       +101.3%   335d  
  EPL         214    MFG        02-Dec-19   133.7       230.2       599    ₹57,778       +72.1%    365d  
  EMAMILTD    85     FMCG       01-Sep-20   323.5       401.4       368    ₹28,695       +24.1%    91d   
  THYROCARE   167    HEALTH     02-Nov-20   335.9       316.0       402    ₹-8,027       -5.9%     29d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RATNAMANI   10     METAL      2.847    0.10   +74.0%    +43.1%    1,095.9     144    ₹157,808      +11.3%    
  KAJARIACER  13     CON DUR    2.542    -0.04  +30.2%    +54.6%    640.3       247    ₹158,155      +10.2%    
  TATAELXSI   14     IT         2.671    0.02   +107.4%   +48.4%    1,527.0     103    ₹157,282      +5.9%     
  JKCEMENT    15     MFG        2.626    -0.08  +79.8%    +37.8%    2,018.6     78     ₹157,449      +7.9%     

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  1      ENERGY     01-Nov-19   89.9        1,134.7     892    ₹931,917      +1161.5%    +10.4%    
  ALKYLAMINE  8      MFG        02-Dec-19   411.5       1,546.5     194    ₹220,182      +275.8%     +13.3%    
  DIXON       5      CON DUR    01-Jan-20   754.7       2,301.9     110    ₹170,186      +205.0%     +9.6%     
  NAVINFLUOR  7      MFG        01-Jan-20   980.3       2,638.5     85     ₹140,951      +169.2%     +6.2%     
  CGPOWER     3      ENERGY     01-Sep-20   21.0        42.7        5661   ₹122,748      +103.1%     +20.4%    
  AARTIDRUGS  6      HEALTH     03-Aug-20   429.0       695.2       246    ₹65,482       +62.0%      +2.0%     
  LAURUSLABS  2      HEALTH     03-Aug-20   197.1       317.1       535    ₹64,202       +60.9%      +9.9%     
  AMBER       21     CON DUR    03-Feb-20   1,532.1     2,439.1     59     ₹53,510       +59.2%      +8.4%     
  ADANIENT    19     METAL      01-Sep-20   280.3       420.7       424    ₹59,528       +50.1%      +10.8%    
  MASTEK      28     IT         03-Aug-20   613.2       912.0       172    ₹51,383       +48.7%      +0.2%     
  CDSL        26     FIN SVC    03-Aug-20   159.5       235.2       662    ₹50,114       +47.5%      +2.5%     
  GRANULES    9      HEALTH     03-Aug-20   293.4       420.4       359    ₹45,573       +43.3%      +8.2%     
  PERSISTENT  24     IT         03-Aug-20   466.7       577.6       226    ₹25,061       +23.8%      +2.7%     
  POLYMED     38     HEALTH     03-Aug-20   408.3       484.7       258    ₹19,713       +18.7%      +0.6%     
  BSOFT       40     IT         01-Sep-20   157.4       181.5       756    ₹18,283       +15.4%      +3.4%     
  DIVISLAB    33     HEALTH     01-Sep-20   3,134.3     3,512.3     38     ₹14,363       +12.1%      +5.7%     
  ADVENZYMES  11     HEALTH     01-Oct-20   316.9       341.3       419    ₹10,198       +7.7%       +9.1%     
  BLUEDART    4      INFRA      02-Nov-20   3,775.6     3,844.5     35     ₹2,412        +1.8%       +1.4%     
  JBCHEPHARM  12     HEALTH     01-Oct-20   468.5       475.4       284    ₹1,947        +1.5%       +3.3%     

  AFTER: Invested ₹4,727,530 | Cash ₹32,906 | Total ₹4,760,437 | Positions 23/30 | Slot ₹158,706

========================================================================
  REBALANCE #73  —  01 Jan 2021
  NAV: ₹4,918,175  |  Slot: ₹163,939  |  Cash: ₹32,906
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  AMBER       85     CON DUR    03-Feb-20   1,532.1     2,386.3     59     ₹50,395       +55.7%    333d  
  POLYMED     75     HEALTH     03-Aug-20   408.3       504.1       258    ₹24,717       +23.5%    151d  
  GRANULES    73     HEALTH     03-Aug-20   293.4       352.9       359    ₹21,348       +20.3%    151d  
  BLUEDART    57     INFRA      02-Nov-20   3,775.6     3,980.7     35     ₹7,178        +5.4%     60d   
  KAJARIACER  67     CON DUR    01-Dec-20   640.3       673.5       247    ₹8,209        +5.2%     31d   
  ADVENZYMES  176    HEALTH     01-Oct-20   316.9       318.9       419    ₹813          +0.6%     92d   
  RATNAMANI   58     METAL      01-Dec-20   1,095.9     1,038.7     144    ₹-8,236       -5.2%     31d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SUZLON      2      ENERGY     4.331    0.13   +262.2%   +131.0%   6.2         26647  ₹163,933      +33.2%    
  HBLENGINE   3      MFG        4.170    0.13   +155.5%   +143.9%   38.8        4227   ₹163,928      +24.6%    
  VAKRANGEE   7      IT         3.160    0.18   +39.1%    +126.9%   60.6        2703   ₹163,897      +19.0%    
  VAIBHAVGBL  8      CON DUR    2.826    -0.14  +223.7%   +34.1%    461.9       354    ₹163,517      +14.4%    
  SAIL        9      METAL      3.206    -0.15  +73.7%    +118.2%   63.2        2592   ₹163,926      +22.8%    
  TATACHEM    11     MFG        2.882    0.07   +69.1%    +59.1%    440.7       372    ₹163,939      +2.3%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  1      ENERGY     01-Nov-19   89.9        1,065.9     892    ₹870,592      +1085.0%    +2.0%     
  ALKYLAMINE  14     MFG        02-Dec-19   411.5       1,512.7     194    ₹213,633      +267.6%     +1.6%     
  DIXON       5      CON DUR    01-Jan-20   754.7       2,718.5     110    ₹216,011      +260.2%     +6.9%     
  NAVINFLUOR  29     MFG        01-Jan-20   980.3       2,594.4     85     ₹137,202      +164.7%     +2.6%     
  CGPOWER     6      ENERGY     01-Sep-20   21.0        44.1        5661   ₹130,577      +109.6%     +3.7%     
  MASTEK      21     IT         03-Aug-20   613.2       1,160.0     172    ₹94,035       +89.2%      +10.7%    
  LAURUSLABS  4      HEALTH     03-Aug-20   197.1       347.1       535    ₹80,229       +76.1%      +4.7%     
  ADANIENT    13     METAL      01-Sep-20   280.3       489.7       424    ₹88,806       +74.7%      +6.8%     
  AARTIDRUGS  10     HEALTH     03-Aug-20   429.0       724.7       246    ₹72,747       +68.9%      +1.4%     
  CDSL        53     FIN SVC    03-Aug-20   159.5       253.0       662    ₹61,865       +58.6%      +2.8%     
  PERSISTENT  34     IT         03-Aug-20   466.7       716.3       226    ₹56,397       +53.5%      +8.6%     
  BSOFT       17     IT         01-Sep-20   157.4       234.1       756    ₹58,018       +48.8%      +10.2%    
  DIVISLAB    19     HEALTH     01-Sep-20   3,134.3     3,733.7     38     ₹22,779       +19.1%      +3.5%     
  TATAELXSI   18     IT         01-Dec-20   1,527.0     1,751.4     103    ₹23,115       +14.7%      +8.4%     
  JBCHEPHARM  54     HEALTH     01-Oct-20   468.5       489.0       284    ₹5,807        +4.4%       +1.0%     
  JKCEMENT    55     MFG        01-Dec-20   2,018.6     1,880.8     78     ₹-10,744      -6.8%       -0.7%     

  AFTER: Invested ₹4,882,001 | Cash ₹35,007 | Total ₹4,917,008 | Positions 22/30 | Slot ₹163,939

========================================================================
  REBALANCE #74  —  01 Feb 2021
  NAV: ₹4,988,556  |  Slot: ₹166,285  |  Cash: ₹35,007
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  NAVINFLUOR  138    MFG        01-Jan-20   980.3       2,307.2     85     ₹112,789      +135.4%   397d  
  CDSL        155    FIN SVC    03-Aug-20   159.5       232.5       662    ₹48,334       +45.8%    182d  
  DIVISLAB    71     HEALTH     01-Sep-20   3,134.3     3,359.8     38     ₹8,570        +7.2%     153d  
  JKCEMENT    106    MFG        01-Dec-20   2,018.6     2,116.9     78     ₹7,666        +4.9%     62d   
  JBCHEPHARM  130    HEALTH     01-Oct-20   468.5       466.7       284    ₹-524         -0.4%     123d  
  SAIL        69     METAL      01-Jan-21   63.2        54.0        2592   ₹-23,984      -14.6%    31d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IIFL        10     FIN SVC    2.703    0.31   +9.8%     +135.7%   168.4       987    ₹166,196      +41.6%    
  TRIDENT     11     CONSUMP    2.690    0.21   +112.8%   +97.9%    13.1        12678  ₹166,282      +5.1%     
  IFBIND      12     CON DUR    2.641    0.11   +105.8%   +93.4%    1,363.0     121    ₹164,923      +4.6%     
  TMPV        14     AUTO       2.721    0.32   +58.9%    +110.8%   272.5       610    ₹166,253      +12.6%    
  JKTYRE      17     AUTO       2.515    0.11   +58.5%    +94.2%    123.5       1346   ₹166,164      +21.3%    

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  1      ENERGY     01-Nov-19   89.9        1,024.2     892    ₹833,396      +1038.7%    +0.6%     
  ALKYLAMINE  4      MFG        02-Dec-19   411.5       1,934.4     194    ₹295,437      +370.1%     +5.9%     
  DIXON       3      CON DUR    01-Jan-20   754.7       3,029.4     110    ₹250,219      +301.4%     +6.0%     
  ADANIENT    13     METAL      01-Sep-20   280.3       535.7       424    ₹108,275      +91.1%      +4.7%     
  CGPOWER     16     ENERGY     01-Sep-20   21.0        40.0        5661   ₹107,090      +89.9%      -2.1%     
  MASTEK      26     IT         03-Aug-20   613.2       1,111.5     172    ₹85,693       +81.2%      -1.2%     
  TATAELXSI   2      IT         01-Dec-20   1,527.0     2,765.8     103    ₹127,595      +81.1%      +21.6%    
  LAURUSLABS  5      HEALTH     03-Aug-20   197.1       343.9       535    ₹78,495       +74.4%      -2.1%     
  PERSISTENT  15     IT         03-Aug-20   466.7       739.7       226    ₹61,704       +58.5%      +0.5%     
  AARTIDRUGS  8      HEALTH     03-Aug-20   429.0       668.4       246    ₹58,884       +55.8%      -3.5%     
  BSOFT       9      IT         01-Sep-20   157.4       234.7       756    ₹58,480       +49.2%      -0.4%     
  VAIBHAVGBL  7      CON DUR    01-Jan-21   461.9       508.2       354    ₹16,399       +10.0%      +8.2%     
  TATACHEM    24     MFG        01-Jan-21   440.7       453.6       372    ₹4,816        +2.9%       -1.5%     
  SUZLON      6      ENERGY     01-Jan-21   6.2         6.1         26647  ₹-2,447       -1.5%       +3.8%     
  HBLENGINE   23     MFG        01-Jan-21   38.8        35.0        4227   ₹-16,084      -9.8%       +0.2%     
  VAKRANGEE   38     IT         01-Jan-21   60.6        50.2        2703   ₹-28,189      -17.2%      -8.6%     

  AFTER: Invested ₹4,868,046 | Cash ₹119,525 | Total ₹4,987,571 | Positions 21/30 | Slot ₹166,285

========================================================================
  REBALANCE #75  —  01 Mar 2021
  NAV: ₹5,618,709  |  Slot: ₹187,290  |  Cash: ₹119,525
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MASTEK      59     IT         03-Aug-20   613.2       1,127.4     172    ₹88,441       +83.8%    210d  
  AARTIDRUGS  63     HEALTH     03-Aug-20   429.0       623.0       246    ₹47,727       +45.2%    210d  
  BSOFT       112    IT         01-Sep-20   157.4       211.7       756    ₹41,045       +34.5%    181d  
  JKTYRE      49     AUTO       01-Feb-21   123.5       120.3       1346   ₹-4,289       -2.6%     28d   
  VAKRANGEE   185    IT         01-Jan-21   60.6        48.8        2703   ₹-31,931      -19.5%    59d   

  ENTRIES (4)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MMTC        5      METAL      3.883    0.14   +157.7%   +155.6%   47.8        3918   ₹187,280      +44.3%    
  DEEPAKNTR   9      MFG        3.125    -0.11  +214.9%   +80.8%    1,537.2     121    ₹186,001      +25.9%    
  RCF         11     MFG        2.923    0.01   +137.3%   +109.0%   81.0        2313   ₹187,279      +51.2%    
  POONAWALLA  12     FIN SVC    3.361    0.50   +152.8%   +159.8%   120.2       1557   ₹187,224      +24.0%    

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  2      ENERGY     01-Nov-19   89.9        1,141.0     892    ₹937,537      +1168.5%    +3.1%     
  DIXON       3      CON DUR    01-Jan-20   754.7       3,888.2     110    ₹344,685      +415.2%     +5.8%     
  ALKYLAMINE  41     MFG        02-Dec-19   411.5       1,983.9     194    ₹305,036      +382.1%     +0.7%     
  ADANIENT    4      METAL      01-Sep-20   280.3       849.1       424    ₹241,156      +202.9%     +15.5%    
  CGPOWER     1      ENERGY     01-Sep-20   21.0        56.2        5661   ₹198,802      +166.9%     +13.1%    
  LAURUSLABS  10     HEALTH     03-Aug-20   197.1       357.6       535    ₹85,865       +81.4%      +1.0%     
  PERSISTENT  17     IT         03-Aug-20   466.7       813.4       226    ₹78,341       +74.3%      +1.2%     
  IIFL        7      FIN SVC    01-Feb-21   168.4       275.0       987    ₹105,269      +63.3%      +23.1%    
  TATAELXSI   14     IT         01-Dec-20   1,527.0     2,458.6     103    ₹95,952       +61.0%      -2.8%     
  TATACHEM    6      MFG        01-Jan-21   440.7       682.6       372    ₹90,005       +54.9%      +19.7%    
  VAIBHAVGBL  8      CON DUR    01-Jan-21   461.9       645.7       354    ₹65,055       +39.8%      +15.9%    
  TMPV        30     AUTO       01-Feb-21   272.5       319.7       610    ₹28,734       +17.3%      +4.5%     
  HBLENGINE   23     MFG        01-Jan-21   38.8        38.8        4227   ₹0            +0.0%       +8.8%     
  TRIDENT     31     CONSUMP    01-Feb-21   13.1        12.5        12678  ₹-8,055       -4.8%       -1.1%     
  IFBIND      37     CON DUR    01-Feb-21   1,363.0     1,206.8     121    ₹-18,900      -11.5%      -5.4%     
  SUZLON      34     ENERGY     01-Jan-21   6.2         5.4         26647  ₹-19,574      -11.9%      +1.5%     

  AFTER: Invested ₹5,445,941 | Cash ₹171,879 | Total ₹5,617,821 | Positions 20/30 | Slot ₹187,290

========================================================================
  REBALANCE #76  —  01 Apr 2021
  NAV: ₹5,831,341  |  Slot: ₹194,378  |  Cash: ₹171,879
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TRIDENT     65     CONSUMP    01-Feb-21   13.1        12.9        12678  ₹-2,877       -1.7%     59d   
  HBLENGINE   271    MFG        01-Jan-21   38.8        33.6        4227   ₹-21,857      -13.3%    90d   
  IFBIND      164    CON DUR    01-Feb-21   1,363.0     1,096.8     121    ₹-32,210      -19.5%    59d   
  SUZLON      354    ENERGY     01-Jan-21   6.2         4.7         26647  ₹-39,148      -23.9%    90d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  INTELLECT   1      IT         6.564    0.31   +1379.3%  +130.6%   718.6       270    ₹194,018      +20.9%    
  TANLA       2      IT         6.067    -0.04  +1824.6%  +20.4%    795.6       244    ₹194,122      -2.5%     
  BALAMINES   7      MFG        3.307    -0.01  +614.9%   +91.1%    1,754.4     110    ₹192,985      +4.6%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  14     ENERGY     01-Nov-19   89.9        1,160.1     892    ₹954,529      +1189.7%    -1.8%     
  ALKYLAMINE  21     MFG        02-Dec-19   411.5       2,249.1     194    ₹356,486      +446.5%     +6.3%     
  DIXON       20     CON DUR    01-Jan-20   754.7       3,580.2     110    ₹310,808      +374.4%     -5.8%     
  ADANIENT    3      METAL      01-Sep-20   280.3       1,104.0     424    ₹349,261      +293.9%     +16.5%    
  CGPOWER     4      ENERGY     01-Sep-20   21.0        67.9        5661   ₹265,349      +222.8%     +9.6%     
  PERSISTENT  30     IT         03-Aug-20   466.7       941.6       226    ₹107,333      +101.8%     +6.1%     
  LAURUSLABS  46     HEALTH     03-Aug-20   197.1       359.2       535    ₹86,707       +82.2%      +2.0%     
  TATAELXSI   15     IT         01-Dec-20   1,527.0     2,599.1     103    ₹110,424      +70.2%      +2.7%     
  IIFL        6      FIN SVC    01-Feb-21   168.4       284.8       987    ₹114,929      +69.2%      +2.4%     
  TATACHEM    12     MFG        01-Jan-21   440.7       715.4       372    ₹102,182      +62.3%      +5.5%     
  VAIBHAVGBL  9      CON DUR    01-Jan-21   461.9       727.2       354    ₹93,908       +57.4%      +4.1%     
  TMPV        26     AUTO       01-Feb-21   272.5       299.7       610    ₹16,551       +10.0%      +0.1%     
  DEEPAKNTR   11     MFG        01-Mar-21   1,537.2     1,618.8     121    ₹9,872        +5.3%       +7.1%     
  MMTC        37     METAL      01-Mar-21   47.8        46.5        3918   ₹-4,898       -2.6%       +8.7%     
  POONAWALLA  5      FIN SVC    01-Mar-21   120.2       114.3       1557   ₹-9,238       -4.9%       +0.5%     
  RCF         43     MFG        01-Mar-21   81.0        70.3        2313   ₹-24,683      -13.2%      +2.6%     

  AFTER: Invested ₹5,677,612 | Cash ₹153,038 | Total ₹5,830,651 | Positions 19/30 | Slot ₹194,378

========================================================================
  REBALANCE #77  —  03 May 2021
  NAV: ₹6,300,934  |  Slot: ₹210,031  |  Cash: ₹153,038
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TMPV        118    AUTO       01-Feb-21   272.5       285.5       610    ₹7,874        +4.7%     91d   
  MMTC        110    METAL      01-Mar-21   47.8        42.8        3918   ₹-19,786      -10.6%    63d   
  RCF         114    MFG        01-Mar-21   81.0        64.8        2313   ₹-37,332      -19.9%    63d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JSWSTEEL    7      METAL      3.937    0.22   +345.4%   +92.2%    684.6       306    ₹209,501      +15.6%    
  PRAJIND     8      ENERGY     3.308    0.57   +294.2%   +119.0%   235.1       893    ₹209,945      +13.2%    
  POLYMED     9      HEALTH     3.280    0.02   +335.8%   +92.9%    983.8       213    ₹209,554      +6.0%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIGREEN  50     ENERGY     01-Nov-19   89.9        1,036.5     892    ₹844,323      +1052.3%    -4.5%     
  ALKYLAMINE  16     MFG        02-Dec-19   411.5       3,294.2     194    ₹559,236      +700.5%     +24.0%    
  DIXON       14     CON DUR    01-Jan-20   754.7       4,241.7     110    ₹383,568      +462.0%     +8.0%     
  ADANIENT    1      METAL      01-Sep-20   280.3       1,251.9     424    ₹411,981      +346.7%     +10.2%    
  CGPOWER     4      ENERGY     01-Sep-20   21.0        72.2        5661   ₹289,395      +243.0%     +5.8%     
  LAURUSLABS  33     HEALTH     03-Aug-20   197.1       469.3       535    ₹145,602      +138.0%     +9.0%     
  PERSISTENT  15     IT         03-Aug-20   466.7       1,036.6     226    ₹128,791      +122.1%     +9.6%     
  TATAELXSI   25     IT         01-Dec-20   1,527.0     3,314.9     103    ₹184,154      +117.1%     +13.3%    
  VAIBHAVGBL  21     CON DUR    01-Jan-21   461.9       781.1       354    ₹112,992      +69.1%      +4.3%     
  TATACHEM    30     MFG        01-Jan-21   440.7       719.1       372    ₹103,566      +63.2%      +3.2%     
  IIFL        27     FIN SVC    01-Feb-21   168.4       245.9       987    ₹76,531       +46.0%      -5.1%     
  BALAMINES   5      MFG        01-Apr-21   1,754.4     2,553.1     110    ₹87,862       +45.5%      +22.5%    
  DEEPAKNTR   12     MFG        01-Mar-21   1,537.2     1,871.8     121    ₹40,483       +21.8%      +12.8%    
  TANLA       3      IT         01-Apr-21   795.6       817.6       244    ₹5,363        +2.8%       -1.8%     
  INTELLECT   2      IT         01-Apr-21   718.6       691.1       270    ₹-7,408       -3.8%       +2.5%     
  POONAWALLA  6      FIN SVC    01-Mar-21   120.2       115.5       1557   ₹-7,390       -3.9%       +0.4%     

  AFTER: Invested ₹6,285,327 | Cash ₹14,860 | Total ₹6,300,187 | Positions 19/30 | Slot ₹210,031

========================================================================
  REBALANCE #78  —  01 Jun 2021
  NAV: ₹6,822,029  |  Slot: ₹227,401  |  Cash: ₹14,860
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIGREEN  62     ENERGY     01-Nov-19   89.9        1,271.1     892    ₹1,053,541    +1313.1%  578d  
  DIXON       73     CON DUR    01-Jan-20   754.7       4,098.5     110    ₹367,812      +443.0%   517d  
  VAIBHAVGBL  70     CON DUR    01-Jan-21   461.9       755.0       354    ₹103,757      +63.5%    151d  
  IIFL        168    FIN SVC    01-Feb-21   168.4       253.2       987    ₹83,703       +50.4%    120d  
  TATACHEM    189    MFG        01-Jan-21   440.7       648.6       372    ₹77,350       +47.2%    151d  

  ENTRIES (10)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ATGL        1      OIL&GAS    6.576    0.57   +1126.6%  +188.0%   1,438.2     158    ₹227,239      +10.6%    
  ADANIENSOL  3      ENERGY     4.841    0.48   +762.2%   +101.6%   1,499.4     151    ₹226,417      +13.2%    
  PRINCEPIPE  6      MFG        4.143    0.84   +817.7%   +74.9%    703.3       323    ₹227,162      +7.7%     
  HIKAL       9      HEALTH     3.562    0.49   +230.7%   +137.6%   375.1       606    ₹227,319      +6.6%     
  MASTEK      12     IT         3.043    0.56   +634.3%   +61.1%    1,858.7     122    ₹226,762      +5.9%     
  TRIVENI     17     FMCG       3.011    0.45   +267.5%   +88.4%    140.3       1620   ₹227,366      +6.8%     
  LUXIND      18     CON DUR    2.761    0.29   +228.9%   +68.2%    2,984.9     76     ₹226,851      +24.8%    
  KPITTECH    19     IT         2.737    0.45   +377.7%   +65.6%    229.1       992    ₹227,219      +3.6%     
  CDSL        20     FIN SVC    2.699    0.39   +294.5%   +55.9%    459.4       495    ₹227,399      +10.6%    
  CGCL        21     FIN SVC    2.643    0.15   +277.6%   +58.4%    517.0       439    ₹226,956      +19.1%    

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ALKYLAMINE  16     MFG        02-Dec-19   411.5       3,513.2     194    ₹601,724      +753.7%     +3.0%     
  ADANIENT    4      METAL      01-Sep-20   280.3       1,412.2     424    ₹479,923      +403.8%     +9.7%     
  CGPOWER     2      ENERGY     01-Sep-20   21.0        82.9        5661   ₹350,350      +294.1%     +0.1%     
  LAURUSLABS  14     HEALTH     03-Aug-20   197.1       524.5       535    ₹175,126      +166.0%     +7.1%     
  PERSISTENT  13     IT         03-Aug-20   466.7       1,202.4     226    ₹166,260      +157.6%     +5.6%     
  TATAELXSI   22     IT         01-Dec-20   1,527.0     3,384.3     103    ₹191,296      +121.6%     +1.8%     
  BALAMINES   15     MFG        01-Apr-21   1,754.4     2,686.7     110    ₹102,548      +53.1%      +4.1%     
  PRAJIND     7      ENERGY     03-May-21   235.1       311.3       893    ₹68,053       +32.4%      +2.1%     
  POONAWALLA  11     FIN SVC    01-Mar-21   120.2       141.4       1557   ₹32,872       +17.6%      +8.3%     
  DEEPAKNTR   47     MFG        01-Mar-21   1,537.2     1,730.5     121    ₹23,385       +12.6%      -0.6%     
  INTELLECT   5      IT         01-Apr-21   718.6       733.7       270    ₹4,089        +2.1%       -0.1%     
  TANLA       8      IT         01-Apr-21   795.6       812.3       244    ₹4,079        +2.1%       -1.9%     
  POLYMED     36     HEALTH     03-May-21   983.8       999.6       213    ₹3,352        +1.6%       -0.9%     
  JSWSTEEL    10     METAL      03-May-21   684.6       657.4       306    ₹-8,328       -4.0%       +0.4%     

  AFTER: Invested ₹6,734,789 | Cash ₹84,544 | Total ₹6,819,333 | Positions 24/30 | Slot ₹227,401

========================================================================
  REBALANCE #79  —  01 Jul 2021
  NAV: ₹7,189,831  |  Slot: ₹239,661  |  Cash: ₹84,544
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DEEPAKNTR   54     MFG        01-Mar-21   1,537.2     1,850.4     121    ₹37,896       +20.4%    122d  
  POLYMED     60     HEALTH     03-May-21   983.8       1,004.6     213    ₹4,427        +2.1%     59d   
  ADANIENSOL  89     ENERGY     01-Jun-21   1,499.4     1,006.5     151    ₹-74,435      -32.9%    30d   

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  VENKEYS     8      FMCG       3.572    0.39   +228.4%   +132.8%   3,551.4     67     ₹237,947      +17.4%    
  FSL         12     IT         3.312    0.41   +455.0%   +73.3%    172.7       1387   ₹239,600      +14.6%    

  HOLDS (21)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ALKYLAMINE  20     MFG        02-Dec-19   411.5       3,616.0     194    ₹621,679      +778.7%     +3.3%     
  ADANIENT    3      METAL      01-Sep-20   280.3       1,486.9     424    ₹511,614      +430.5%     -0.3%     
  CGPOWER     7      ENERGY     01-Sep-20   21.0        78.4        5661   ₹324,626      +272.5%     -1.5%     
  LAURUSLABS  2      HEALTH     03-Aug-20   197.1       656.5       535    ₹245,737      +233.0%     +7.4%     
  PERSISTENT  17     IT         03-Aug-20   466.7       1,427.2     226    ₹217,064      +205.8%     +12.5%    
  TATAELXSI   11     IT         01-Dec-20   1,527.0     4,030.3     103    ₹257,836      +163.9%     +11.3%    
  BALAMINES   18     MFG        01-Apr-21   1,754.4     2,753.0     110    ₹109,847      +56.9%      +5.6%     
  PRAJIND     10     ENERGY     03-May-21   235.1       349.8       893    ₹102,463      +48.8%      +1.5%     
  TRIVENI     9      FMCG       01-Jun-21   140.3       185.8       1620   ₹73,658       +32.4%      +8.5%     
  HIKAL       1      HEALTH     01-Jun-21   375.1       485.4       606    ₹66,838       +29.4%      +6.8%     
  LUXIND      4      CON DUR    01-Jun-21   2,984.9     3,743.7     76     ₹57,670       +25.4%      +10.8%    
  POONAWALLA  36     FIN SVC    01-Mar-21   120.2       147.4       1557   ₹42,341       +22.6%      -0.5%     
  MASTEK      13     IT         01-Jun-21   1,858.7     2,139.6     122    ₹34,268       +15.1%      +3.3%     
  KPITTECH    32     IT         01-Jun-21   229.1       255.6       992    ₹26,368       +11.6%      +7.8%     
  PRINCEPIPE  6      MFG        01-Jun-21   703.3       726.5       323    ₹7,489        +3.3%       +5.1%     
  CDSL        24     FIN SVC    01-Jun-21   459.4       467.3       495    ₹3,899        +1.7%       +1.3%     
  INTELLECT   28     IT         01-Apr-21   718.6       718.5       270    ₹-13          -0.0%       -1.2%     
  CGCL        38     FIN SVC    01-Jun-21   517.0       499.5       439    ₹-7,665       -3.4%       +1.5%     
  JSWSTEEL    19     METAL      03-May-21   684.6       644.2       306    ₹-12,369      -5.9%       -1.4%     
  TANLA       5      IT         01-Apr-21   795.6       747.9       244    ₹-11,635      -6.0%       -3.1%     
  ATGL        43     OIL&GAS    01-Jun-21   1,438.2     967.5       158    ₹-74,376      -32.7%      -23.5%    

  AFTER: Invested ₹6,992,973 | Cash ₹196,291 | Total ₹7,189,264 | Positions 23/30 | Slot ₹239,661

========================================================================
  REBALANCE #80  —  02 Aug 2021
  NAV: ₹7,660,537  |  Slot: ₹255,351  |  Cash: ₹196,291
========================================================================

  EXITS (5)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TRIVENI     115    FMCG       01-Jun-21   140.3       175.4       1620   ₹56,841       +25.0%    62d   
  JSWSTEEL    85     METAL      03-May-21   684.6       713.8       306    ₹8,907        +4.3%     91d   
  INTELLECT   87     IT         01-Apr-21   718.6       724.8       270    ₹1,686        +0.9%     123d  
  CGCL        57     FIN SVC    01-Jun-21   517.0       513.4       439    ₹-1,595       -0.7%     62d   
  ATGL        146    OIL&GAS    01-Jun-21   1,438.2     887.1       158    ₹-87,085      -38.3%    62d   

  ENTRIES (4)
  [52w filter blocked 1: HFCL(-22.5%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JSWENERGY   2      ENERGY     4.306    0.34   +434.7%   +122.1%   237.3       1075   ₹255,138      +14.7%    
  ECLERX      6      IT         3.313    0.27   +384.0%   +82.3%    768.2       332    ₹255,029      +8.8%     
  BSE         7      FIN SVC    3.258    0.45   +151.3%   +106.8%   133.3       1916   ₹255,350      +10.0%    
  ICIL        10     CONSUMP    2.966    0.35   +326.2%   +98.9%    261.4       976    ₹255,139      +21.0%    

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ALKYLAMINE  38     MFG        02-Dec-19   411.5       4,316.7     194    ₹757,607      +949.0%     +9.1%     
  ADANIENT    8      METAL      01-Sep-20   280.3       1,435.0     424    ₹489,594      +412.0%     +1.4%     
  CGPOWER     3      ENERGY     01-Sep-20   21.0        78.1        5661   ₹323,228      +271.4%     +1.3%     
  LAURUSLABS  31     HEALTH     03-Aug-20   197.1       644.6       535    ₹239,414      +227.0%     +1.3%     
  PERSISTENT  24     IT         03-Aug-20   466.7       1,515.7     226    ₹237,062      +224.8%     +6.2%     
  TATAELXSI   36     IT         01-Dec-20   1,527.0     4,010.4     103    ₹255,787      +162.6%     +0.6%     
  BALAMINES   35     MFG        01-Apr-21   1,754.4     3,261.3     110    ₹165,764      +85.9%      +9.2%     
  PRAJIND     13     ENERGY     03-May-21   235.1       357.6       893    ₹109,396      +52.1%      +2.0%     
  CDSL        9      FIN SVC    01-Jun-21   459.4       665.3       495    ₹101,915      +44.8%      +8.7%     
  HIKAL       15     HEALTH     01-Jun-21   375.1       539.1       606    ₹99,380       +43.7%      +5.3%     
  POONAWALLA  17     FIN SVC    01-Mar-21   120.2       170.7       1557   ₹78,600       +42.0%      +7.0%     
  LUXIND      1      CON DUR    01-Jun-21   2,984.9     4,167.5     76     ₹89,879       +39.6%      +7.8%     
  MASTEK      11     IT         01-Jun-21   1,858.7     2,537.1     122    ₹82,759       +36.5%      +6.3%     
  KPITTECH    26     IT         01-Jun-21   229.1       288.2       992    ₹58,676       +25.8%      +8.5%     
  TANLA       5      IT         01-Apr-21   795.6       915.3       244    ₹29,212       +15.0%      +6.6%     
  FSL         21     IT         01-Jul-21   172.7       178.7       1387   ₹8,296        +3.5%       -1.5%     
  PRINCEPIPE  14     MFG        01-Jun-21   703.3       681.9       323    ₹-6,898       -3.0%       -0.8%     
  VENKEYS     12     FMCG       01-Jul-21   3,551.4     3,106.5     67     ₹-29,814      -12.5%      -1.6%     

  AFTER: Invested ₹7,421,067 | Cash ₹238,258 | Total ₹7,659,325 | Positions 22/30 | Slot ₹255,351

========================================================================
  REBALANCE #81  —  01 Sep 2021
  NAV: ₹7,760,887  |  Slot: ₹258,696  |  Cash: ₹238,258
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ALKYLAMINE  84     MFG        02-Dec-19   411.5       4,110.2     194    ₹717,556      +898.8%   639d  
  CDSL        58     FIN SVC    01-Jun-21   459.4       563.2       495    ₹51,388       +22.6%    92d   
  TANLA       93     IT         01-Apr-21   795.6       836.8       244    ₹10,055       +5.2%     153d  
  BSE         60     FIN SVC    02-Aug-21   133.3       126.6       1916   ₹-12,849      -5.0%     30d   
  FSL         81     IT         01-Jul-21   172.7       161.5       1387   ₹-15,657      -6.5%     62d   
  PRINCEPIPE  118    MFG        01-Jun-21   703.3       644.5       323    ₹-18,986      -8.4%     92d   
  VENKEYS     185    FMCG       01-Jul-21   3,551.4     2,810.3     67     ₹-49,654      -20.9%    62d   

  ENTRIES (9)
  [52w filter blocked 1: HGS(-22.8%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ATGL        2      OIL&GAS    3.423    0.63   +717.5%   -4.3%     1,508.6     171    ₹257,977      +29.6%    
  APOLLOHOSP  3      HEALTH     3.392    0.44   +206.1%   +54.0%    4,977.0     51     ₹253,829      +10.4%    
  SRF         5      MFG        3.258    0.62   +135.4%   +52.3%    1,958.2     132    ₹258,483      +9.1%     
  ADANIENSOL  8      ENERGY     3.046    0.65   +496.9%   +6.5%     1,659.6     155    ₹257,238      +33.6%    
  TECHM       10     IT         2.905    0.49   +108.3%   +45.0%    1,205.2     214    ₹257,906      +4.8%     
  KNRCON      12     INFRA      2.820    0.67   +145.4%   +51.8%    330.9       781    ₹258,399      +10.9%    
  BAJAJFINSV  14     FIN SVC    2.709    0.93   +157.1%   +42.0%    1,673.8     154    ₹257,771      +9.1%     
  MPHASIS     16     IT         2.686    0.37   +140.1%   +48.6%    2,538.1     101    ₹256,352      +2.6%     
  CARBORUNIV  18     MFG        2.630    0.54   +198.6%   +39.4%    801.9       322    ₹258,220      +10.1%    

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ADANIENT    9      METAL      01-Sep-20   280.3       1,561.7     424    ₹543,322      +457.2%     +5.9%     
  CGPOWER     42     ENERGY     01-Sep-20   21.0        84.8        5661   ₹360,696      +302.8%     +3.7%     
  PERSISTENT  11     IT         03-Aug-20   466.7       1,610.3     226    ₹258,451      +245.0%     +4.4%     
  LAURUSLABS  46     HEALTH     03-Aug-20   197.1       652.7       535    ₹243,735      +231.1%     -1.0%     
  TATAELXSI   4      IT         01-Dec-20   1,527.0     4,604.6     103    ₹316,992      +201.5%     +5.1%     
  BALAMINES   6      MFG        01-Apr-21   1,754.4     4,097.9     110    ₹257,779      +133.6%     +16.7%    
  HIKAL       13     HEALTH     01-Jun-21   375.1       613.3       606    ₹144,352      +63.5%      +0.4%     
  KPITTECH    20     IT         01-Jun-21   229.1       322.6       992    ₹92,841       +40.9%      +2.6%     
  POONAWALLA  54     FIN SVC    01-Mar-21   120.2       169.0       1557   ₹75,983       +40.6%      -2.9%     
  MASTEK      17     IT         01-Jun-21   1,858.7     2,600.4     122    ₹90,485       +39.9%      +6.4%     
  PRAJIND     48     ENERGY     03-May-21   235.1       325.2       893    ₹80,443       +38.3%      -0.2%     
  LUXIND      22     CON DUR    01-Jun-21   2,984.9     4,017.3     76     ₹78,462       +34.6%      +0.7%     
  JSWENERGY   1      ENERGY     02-Aug-21   237.3       248.8       1075   ₹12,364       +4.8%       +6.5%     
  ECLERX      7      IT         02-Aug-21   768.2       731.5       332    ₹-12,164      -4.8%       -0.6%     
  ICIL        36     CONSUMP    02-Aug-21   261.4       226.4       976    ₹-34,174      -13.4%      -2.5%     

  AFTER: Invested ₹7,695,541 | Cash ₹62,596 | Total ₹7,758,137 | Positions 24/30 | Slot ₹258,696

========================================================================
  REBALANCE #82  —  01 Oct 2021
  NAV: ₹8,213,310  |  Slot: ₹273,777  |  Cash: ₹62,596
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ADANIENT    17     METAL      01-Sep-20   280.3       1,455.5     424    ₹498,267      +419.3%   395d  
  LAURUSLABS  229    HEALTH     03-Aug-20   197.1       609.9       535    ₹220,813      +209.4%   424d  
  HIKAL       90     HEALTH     01-Jun-21   375.1       566.4       606    ₹115,924      +51.0%    122d  
  KPITTECH    64     IT         01-Jun-21   229.1       330.6       992    ₹100,699      +44.3%    122d  
  LUXIND      163    CON DUR    01-Jun-21   2,984.9     3,603.0     76     ₹46,976       +20.7%    122d  
  ECLERX      89     IT         02-Aug-21   768.2       734.2       332    ₹-11,260      -4.4%     60d   
  TECHM       53     IT         01-Sep-21   1,205.2     1,150.6     214    ₹-11,681      -4.5%     30d   
  APOLLOHOSP  101    HEALTH     01-Sep-21   4,977.0     4,404.5     51     ₹-29,198      -11.5%    30d   
  KNRCON      85     INFRA      01-Sep-21   330.9       288.8       781    ₹-32,879      -12.7%    30d   

  ENTRIES (10)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TRIDENT     8      CONSUMP    3.233    0.63   +348.4%   +79.8%    27.3        10038  ₹273,752      +18.1%    
  OIL         9      OIL&GAS    3.225    0.55   +196.4%   +54.8%    132.1       2073   ₹273,745      +16.1%    
  LINDEINDIA  10     MFG        3.144    0.36   +263.3%   +53.5%    2,601.0     105    ₹273,108      +1.6%     
  GUJALKALI   12     MFG        2.869    0.61   +128.5%   +84.9%    676.2       404    ₹273,201      +29.9%    
  LTTS        13     IT         2.796    0.81   +194.6%   +61.4%    4,375.8     62     ₹271,301      +2.8%     
  SOBHA       15     REALTY     2.603    1.14   +237.9%   +66.4%    768.8       356    ₹273,685      +3.1%     
  BAJAJHLDNG  16     FIN SVC    2.537    0.47   +99.7%    +34.8%    4,433.3     61     ₹270,431      +4.4%     
  GODREJPROP  17     REALTY     2.495    0.95   +161.4%   +61.2%    2,233.3     122    ₹272,463      +14.7%    
  PRESTIGE    19     REALTY     2.466    0.77   +97.5%    +69.3%    477.2       573    ₹273,439      +8.8%     
  TATAPOWER   20     ENERGY     2.442    1.15   +212.2%   +34.9%    158.3       1729   ₹273,685      +15.5%    

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  CGPOWER     2      ENERGY     01-Sep-20   21.0        122.7       5661   ₹575,715      +483.3%     +20.6%    
  PERSISTENT  32     IT         03-Aug-20   466.7       1,760.2     226    ₹292,321      +277.1%     +2.2%     
  TATAELXSI   6      IT         01-Dec-20   1,527.0     5,491.9     103    ₹408,384      +259.7%     +6.9%     
  BALAMINES   5      MFG        01-Apr-21   1,754.4     4,471.8     110    ₹298,916      +154.9%     +2.0%     
  JSWENERGY   1      ENERGY     02-Aug-21   237.3       375.5       1075   ₹148,472      +58.2%      +16.4%    
  MASTEK      30     IT         01-Jun-21   1,858.7     2,935.1     122    ₹131,323      +57.9%      +2.8%     
  PRAJIND     41     ENERGY     03-May-21   235.1       334.8       893    ₹89,041       +42.4%      +1.2%     
  POONAWALLA  52     FIN SVC    01-Mar-21   120.2       162.0       1557   ₹65,051       +34.7%      -4.0%     
  SRF         7      MFG        01-Sep-21   1,958.2     2,186.9     132    ₹30,190       +11.7%      +3.6%     
  MPHASIS     19     IT         01-Sep-21   2,538.1     2,770.0     101    ₹23,421       +9.1%       -1.8%     
  CARBORUNIV  14     MFG        01-Sep-21   801.9       858.7       322    ₹18,277       +7.1%       +2.0%     
  ICIL        34     CONSUMP    02-Aug-21   261.4       269.1       976    ₹7,524        +2.9%       +2.2%     
  BAJAJFINSV  11     FIN SVC    01-Sep-21   1,673.8     1,712.9     154    ₹6,012        +2.3%       -0.1%     
  ADANIENSOL  3      ENERGY     01-Sep-21   1,659.6     1,577.8     155    ₹-12,671      -4.9%       -1.1%     
  ATGL        4      OIL&GAS    01-Sep-21   1,508.6     1,421.5     171    ₹-14,898      -5.8%       +3.2%     

  AFTER: Invested ₹8,050,994 | Cash ₹159,076 | Total ₹8,210,069 | Positions 25/30 | Slot ₹273,777

========================================================================
  REBALANCE #83  —  01 Nov 2021
  NAV: ₹8,377,763  |  Slot: ₹279,259  |  Cash: ₹159,076
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BALAMINES   69     MFG        01-Apr-21   1,754.4     3,232.9     110    ₹162,634      +84.3%    214d  
  MASTEK      71     IT         01-Jun-21   1,858.7     2,683.6     122    ₹100,636      +44.4%    153d  
  TATAPOWER   5      ENERGY     01-Oct-21   158.3       215.5       1729   ₹98,944       +36.2%    31d   
  SOBHA       51     REALTY     01-Oct-21   768.8       817.5       356    ₹17,332       +6.3%     31d   
  BAJAJFINSV  15     FIN SVC    01-Sep-21   1,673.8     1,751.6     154    ₹11,968       +4.6%     61d   
  PRESTIGE    124    REALTY     01-Oct-21   477.2       432.3       573    ₹-25,726      -9.4%     31d   
  ICIL        347    CONSUMP    02-Aug-21   261.4       230.7       976    ₹-29,986      -11.8%    91d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  UNITDSPR    6      FMCG       3.128    0.58   +88.6%    +51.9%    953.3       292    ₹278,372      +11.5%    
  TRITURBINE  9      ENERGY     2.850    0.67   +192.7%   +66.0%    195.1       1431   ₹279,180      +17.0%    
  CENTURYPLY  10     CON DUR    2.761    0.67   +200.1%   +36.0%    569.6       490    ₹279,122      +9.0%     
  TITAN       11     CON DUR    2.716    0.87   +98.4%    +40.3%    2,375.1     117    ₹277,890      +1.6%     
  GRINDWELL   12     MFG        2.697    0.85   +204.4%   +33.7%    1,626.3     171    ₹278,098      +9.9%     
  BRIGADE     13     REALTY     2.626    0.81   +175.0%   +47.2%    355.1       786    ₹279,074      +7.3%     
  TANLA       14     IT         2.655    0.31   +322.4%   +24.7%    1,107.0     252    ₹278,961      +12.4%    
  SOLARINDS   17     DEFENCE    2.574    0.65   +135.7%   +42.3%    2,402.7     116    ₹278,712      +0.6%     

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  CGPOWER     1      ENERGY     01-Sep-20   21.0        147.1       5661   ₹713,842      +599.3%     +11.1%    
  PERSISTENT  9      IT         03-Aug-20   466.7       1,957.1     226    ₹336,832      +319.3%     +2.9%     
  TATAELXSI   10     IT         01-Dec-20   1,527.0     5,673.7     103    ₹427,106      +271.6%     +0.8%     
  POONAWALLA  62     FIN SVC    01-Mar-21   120.2       170.1       1557   ₹77,676       +41.5%      +5.7%     
  JSWENERGY   6      ENERGY     02-Aug-21   237.3       333.6       1075   ₹103,470      +40.6%      -5.9%     
  PRAJIND     47     ENERGY     03-May-21   235.1       327.7       893    ₹82,667       +39.4%      +2.2%     
  TRIDENT     3      CONSUMP    01-Oct-21   27.3        37.3        10038  ₹100,268      +36.6%      +9.2%     
  MPHASIS     22     IT         01-Sep-21   2,538.1     3,039.8     101    ₹50,666       +19.8%      +2.1%     
  SRF         32     MFG        01-Sep-21   1,958.2     2,092.4     132    ₹17,717       +6.9%       -3.8%     
  ADANIENSOL  2      ENERGY     01-Sep-21   1,659.6     1,760.6     155    ₹15,647       +6.1%       +0.5%     
  LTTS        30     IT         01-Oct-21   4,375.8     4,629.8     62     ₹15,750       +5.8%       +4.4%     
  GODREJPROP  38     REALTY     01-Oct-21   2,233.3     2,345.3     122    ₹13,664       +5.0%       +2.4%     
  CARBORUNIV  20     MFG        01-Sep-21   801.9       820.4       322    ₹5,939        +2.3%       -3.0%     
  GUJALKALI   24     MFG        01-Oct-21   676.2       687.0       404    ₹4,336        +1.6%       +2.2%     
  BAJAJHLDNG  25     FIN SVC    01-Oct-21   4,433.3     4,457.9     61     ₹1,504        +0.6%       +0.6%     
  ATGL        4      OIL&GAS    01-Sep-21   1,508.6     1,432.8     171    ₹-12,977      -5.0%       +0.6%     
  OIL         17     OIL&GAS    01-Oct-21   132.1       119.0       2073   ₹-27,041      -9.9%       +0.2%     
  LINDEINDIA  29     MFG        01-Oct-21   2,601.0     2,333.4     105    ₹-28,096      -10.3%      -4.1%     

  AFTER: Invested ₹8,358,829 | Cash ₹16,286 | Total ₹8,375,115 | Positions 26/30 | Slot ₹279,259

========================================================================
  REBALANCE #84  —  01 Dec 2021
  NAV: ₹8,514,581  |  Slot: ₹283,819  |  Cash: ₹16,286
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  CARBORUNIV  58     MFG        01-Sep-21   801.9       875.4       322    ₹23,662       +9.2%     91d   
  MPHASIS     71     IT         01-Sep-21   2,538.1     2,756.6     101    ₹22,067       +8.6%     91d   
  SRF         118    MFG        01-Sep-21   1,958.2     1,988.5     132    ₹3,998        +1.5%     91d   
  UNITDSPR    64     FMCG       01-Nov-21   953.3       855.9       292    ₹-28,463      -10.2%    30d   
  GODREJPROP  69     REALTY     01-Oct-21   2,233.3     1,966.5     122    ₹-32,550      -11.9%    61d   
  GUJALKALI   75     MFG        01-Oct-21   676.2       552.9       404    ₹-49,830      -18.2%    61d   

  ENTRIES (5)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  KPITTECH    3      IT         3.840    0.81   +380.7%   +50.9%    495.4       572    ₹283,357      +16.7%    
  SFL         8      CON DUR    3.141    0.54   +141.5%   +38.6%    1,616.7     175    ₹282,918      +7.0%     
  KPRMILL     12     MFG        2.941    0.84   +223.2%   +48.8%    500.8       566    ₹283,427      +1.9%     
  KEI         15     MFG        2.724    0.63   +199.7%   +46.3%    1,126.2     252    ₹283,815      +9.5%     
  BSE         17     FIN SVC    2.701    0.81   +193.8%   +38.0%    175.3       1619   ₹283,815      +9.5%     

  HOLDS (20)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  CGPOWER     2      ENERGY     01-Sep-20   21.0        149.0       5661   ₹724,187      +608.0%     -0.2%     
  PERSISTENT  6      IT         03-Aug-20   466.7       2,032.5     226    ₹353,870      +335.5%     +2.7%     
  TATAELXSI   10     IT         01-Dec-20   1,527.0     5,587.5     103    ₹418,229      +265.9%     -3.0%     
  TRIDENT     1      CONSUMP    01-Oct-21   27.3        46.2        10038  ₹190,484      +69.6%      +5.9%     
  POONAWALLA  35     FIN SVC    01-Mar-21   120.2       186.2       1557   ₹102,696      +54.9%      +1.8%     
  PRAJIND     42     ENERGY     03-May-21   235.1       315.3       893    ₹71,587       +34.1%      -2.7%     
  JSWENERGY   8      ENERGY     02-Aug-21   237.3       294.0       1075   ₹60,929       +23.9%      -4.2%     
  TANLA       11     IT         01-Nov-21   1,107.0     1,364.8     252    ₹64,967       +23.3%      +6.4%     
  SOLARINDS   5      DEFENCE    01-Nov-21   2,402.7     2,813.8     116    ₹47,686       +17.1%      +6.1%     
  LTTS        12     IT         01-Oct-21   4,375.8     5,055.3     62     ₹42,127       +15.5%      +2.6%     
  BAJAJHLDNG  26     FIN SVC    01-Oct-21   4,433.3     4,853.5     61     ₹25,634       +9.5%       +4.8%     
  ADANIENSOL  14     ENERGY     01-Sep-21   1,659.6     1,797.2     155    ₹21,320       +8.3%       -4.2%     
  ATGL        19     OIL&GAS    01-Sep-21   1,508.6     1,623.7     171    ₹19,679       +7.6%       +1.9%     
  BRIGADE     16     REALTY     01-Nov-21   355.1       369.1       786    ₹11,023       +3.9%       +2.9%     
  CENTURYPLY  7      CON DUR    01-Nov-21   569.6       578.0       490    ₹4,090        +1.5%       -6.5%     
  GRINDWELL   21     MFG        01-Nov-21   1,626.3     1,629.3     171    ₹515          +0.2%       -1.2%     
  TITAN       37     CON DUR    01-Nov-21   2,375.1     2,329.6     117    ₹-5,324       -1.9%       -1.9%     
  LINDEINDIA  52     MFG        01-Oct-21   2,601.0     2,548.7     105    ₹-5,492       -2.0%       +3.9%     
  TRITURBINE  24     ENERGY     01-Nov-21   195.1       185.4       1431   ₹-13,853      -5.0%       +2.1%     
  OIL         30     OIL&GAS    01-Oct-21   132.1       115.3       2073   ₹-34,804      -12.7%      +2.0%     

  AFTER: Invested ₹8,379,653 | Cash ₹133,246 | Total ₹8,512,898 | Positions 25/30 | Slot ₹283,819

========================================================================
  REBALANCE #85  —  03 Jan 2022
  NAV: ₹9,275,694  |  Slot: ₹309,190  |  Cash: ₹133,246
========================================================================

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PRAJIND     69     ENERGY     03-May-21   235.1       323.6       893    ₹78,988       +37.6%    245d  
  TITAN       53     CON DUR    01-Nov-21   2,375.1     2,491.2     117    ₹13,575       +4.9%     63d   
  BRIGADE     62     REALTY     01-Nov-21   355.1       366.6       786    ₹9,069        +3.2%     63d   
  KEI         56     MFG        01-Dec-21   1,126.2     1,131.2     252    ₹1,251        +0.4%     33d   
  SOLARINDS   58     DEFENCE    01-Nov-21   2,402.7     2,373.7     116    ₹-3,360       -1.2%     63d   
  LINDEINDIA  93     MFG        01-Oct-21   2,601.0     2,491.4     105    ₹-11,515      -4.2%     94d   
  TRITURBINE  75     ENERGY     01-Nov-21   195.1       186.4       1431   ₹-12,459      -4.5%     63d   
  OIL         257    OIL&GAS    01-Oct-21   132.1       106.9       2073   ₹-52,138      -19.0%    94d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  UNOMINDA    6      AUTO       3.846    0.96   +199.5%   +66.9%    595.1       519    ₹308,866      +8.6%     
  RAJESHEXPO  10     CON DUR    3.034    0.30   +75.1%    +45.7%    851.1       363    ₹308,956      +13.4%    
  RADICO      12     FMCG       2.917    0.66   +173.0%   +36.1%    1,210.1     255    ₹308,569      +5.7%     
  BSOFT       14     IT         2.849    0.77   +131.3%   +43.2%    538.0       574    ₹308,804      +10.1%    
  TECHM       15     IT         2.843    0.84   +95.4%    +31.5%    1,512.5     204    ₹308,548      +4.9%     
  TIINDIA     17     AUTO       2.813    0.79   +137.0%   +36.2%    1,877.1     164    ₹307,841      +10.9%    
  ECLERX      18     IT         2.758    0.33   +212.0%   +26.3%    927.6       333    ₹308,887      +17.1%    

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  CGPOWER     3      ENERGY     01-Sep-20   21.0        193.7       5661   ₹977,513      +820.7%     +9.2%     
  PERSISTENT  7      IT         03-Aug-20   466.7       2,353.9     226    ₹426,496      +404.4%     +6.7%     
  TATAELXSI   33     IT         01-Dec-20   1,527.0     5,597.2     103    ₹419,227      +266.5%     +2.5%     
  TRIDENT     1      CONSUMP    01-Oct-21   27.3        50.4        10038  ₹232,303      +84.9%      +3.1%     
  POONAWALLA  8      FIN SVC    01-Mar-21   120.2       218.3       1557   ₹152,658      +81.5%      +7.0%     
  TANLA       5      IT         01-Nov-21   1,107.0     1,714.7     252    ₹153,138      +54.9%      +3.2%     
  KPRMILL     4      MFG        01-Dec-21   500.8       690.4       566    ₹107,311      +37.9%      +14.4%    
  LTTS        34     IT         01-Oct-21   4,375.8     5,407.4     62     ₹63,958       +23.6%      +5.3%     
  JSWENERGY   42     ENERGY     02-Aug-21   237.3       291.4       1075   ₹58,153       +22.8%      -1.0%     
  KPITTECH    2      IT         01-Dec-21   495.4       594.7       572    ₹56,800       +20.0%      +13.7%    
  GRINDWELL   13     MFG        01-Nov-21   1,626.3     1,893.2     171    ₹45,638       +16.4%      +6.8%     
  BSE         9      FIN SVC    01-Dec-21   175.3       202.3       1619   ₹43,657       +15.4%      +0.4%     
  ATGL        11     OIL&GAS    01-Sep-21   1,508.6     1,741.2     171    ₹39,776       +15.4%      +0.4%     
  BAJAJHLDNG  40     FIN SVC    01-Oct-21   4,433.3     5,006.4     61     ₹34,961       +12.9%      +3.1%     
  CENTURYPLY  27     CON DUR    01-Nov-21   569.6       602.0       490    ₹15,875       +5.7%       +0.9%     
  ADANIENSOL  30     ENERGY     01-Sep-21   1,659.6     1,731.1     155    ₹11,082       +4.3%       -2.8%     
  SFL         16     CON DUR    01-Dec-21   1,616.7     1,618.9     175    ₹389          +0.1%       +1.5%     

  AFTER: Invested ₹9,124,040 | Cash ₹149,089 | Total ₹9,273,129 | Positions 24/30 | Slot ₹309,190

========================================================================
  REBALANCE #86  —  01 Feb 2022
  NAV: ₹9,183,168  |  Slot: ₹306,106  |  Cash: ₹149,089
========================================================================

  EXITS (4)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  LTTS        129    IT         01-Oct-21   4,375.8     4,442.6     62     ₹4,142        +1.5%     123d  
  RADICO      108    FMCG       03-Jan-22   1,210.1     1,052.6     255    ₹-40,153      -13.0%    29d   
  TECHM       137    IT         03-Jan-22   1,512.5     1,276.0     204    ₹-48,241      -15.6%    29d   
  BSOFT       61     IT         03-Jan-22   538.0       447.9       574    ₹-51,717      -16.7%    29d   

  ENTRIES (3)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SHARDACROP  4      FMCG       3.902    0.45   +109.2%   +88.9%    567.5       539    ₹305,900      +32.2%    
  ELGIEQUIP   9      MFG        3.369    0.50   +119.4%   +70.6%    336.5       909    ₹305,838      +2.1%     
  THERMAX     10     ENERGY     3.171    0.45   +118.0%   +54.1%    2,030.4     150    ₹304,561      +5.1%     

  HOLDS (20)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  CGPOWER     6      ENERGY     01-Sep-20   21.0        171.8       5661   ₹853,646      +716.7%     -4.3%     
  PERSISTENT  18     IT         03-Aug-20   466.7       2,183.0     226    ₹387,888      +367.8%     +3.2%     
  TATAELXSI   11     IT         01-Dec-20   1,527.0     7,089.3     103    ₹572,920      +364.3%     +10.5%    
  POONAWALLA  2      FIN SVC    01-Mar-21   120.2       278.8       1557   ₹246,809      +131.8%     +5.8%     
  TRIDENT     3      CONSUMP    01-Oct-21   27.3        56.5        10038  ₹292,917      +107.0%     +0.8%     
  TANLA       13     IT         01-Nov-21   1,107.0     1,643.0     252    ₹135,069      +48.4%      -2.1%     
  KPRMILL     5      MFG        01-Dec-21   500.8       673.7       566    ₹97,913       +34.5%      +0.6%     
  JSWENERGY   22     ENERGY     02-Aug-21   237.3       300.0       1075   ₹67,321       +26.4%      +0.7%     
  ATGL        7      OIL&GAS    01-Sep-21   1,508.6     1,861.9     171    ₹60,411       +23.4%      +3.3%     
  KPITTECH    1      IT         01-Dec-21   495.4       609.1       572    ₹65,050       +23.0%      -3.5%     
  BSE         8      FIN SVC    01-Dec-21   175.3       211.4       1619   ₹58,424       +20.6%      +0.2%     
  ADANIENSOL  14     ENERGY     01-Sep-21   1,659.6     1,986.9     155    ₹50,732       +19.7%      +1.7%     
  GRINDWELL   25     MFG        01-Nov-21   1,626.3     1,873.7     171    ₹42,308       +15.2%      +1.7%     
  BAJAJHLDNG  52     FIN SVC    01-Oct-21   4,433.3     4,889.1     61     ₹27,803       +10.3%      -0.4%     
  SFL         21     CON DUR    01-Dec-21   1,616.7     1,700.6     175    ₹14,687       +5.2%       +0.7%     
  CENTURYPLY  55     CON DUR    01-Nov-21   569.6       598.2       490    ₹13,976       +5.0%       -2.7%     
  RAJESHEXPO  33     CON DUR    03-Jan-22   851.1       825.3       363    ₹-9,386       -3.0%       -0.6%     
  UNOMINDA    17     AUTO       03-Jan-22   595.1       550.3       519    ₹-23,252      -7.5%       +0.7%     
  TIINDIA     37     AUTO       03-Jan-22   1,877.1     1,676.5     164    ₹-32,901      -10.7%      -3.0%     
  ECLERX      41     IT         03-Jan-22   927.6       815.3       333    ₹-37,390      -12.1%      -4.0%     

  AFTER: Invested ₹8,889,125 | Cash ₹292,955 | Total ₹9,182,080 | Positions 23/30 | Slot ₹306,106

========================================================================
  REBALANCE #87  —  02 Mar 2022
  NAV: ₹8,475,943  |  Slot: ₹282,531  |  Cash: ₹292,955
========================================================================

  [REGIME OFF] Nifty 500 14,199.8 < SMA200 14,510.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (23)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  CGPOWER     10     ENERGY     01-Sep-20   21.0        167.6       5661   ₹829,880      +696.7%     -1.8%     
  TATAELXSI   16     IT         01-Dec-20   1,527.0     6,234.6     103    ₹484,878      +308.3%     -3.4%     
  PERSISTENT  39     IT         03-Aug-20   466.7       1,886.5     226    ₹320,862      +304.2%     -3.7%     
  POONAWALLA  27     FIN SVC    01-Mar-21   120.2       236.5       1557   ₹181,065      +96.7%      -3.0%     
  TRIDENT     9      CONSUMP    01-Oct-21   27.3        47.0        10038  ₹198,472      +72.5%      -7.7%     
  JSWENERGY   1      ENERGY     02-Aug-21   237.3       330.6       1075   ₹100,222      +39.3%      +3.8%     
  ADANIENSOL  8      ENERGY     01-Sep-21   1,659.6     2,241.1     155    ₹90,125       +35.0%      +11.1%    
  TANLA       103    IT         01-Nov-21   1,107.0     1,437.9     252    ₹83,389       +29.9%      -3.9%     
  BSE         4      FIN SVC    01-Dec-21   175.3       213.2       1619   ₹61,397       +21.6%      -3.3%     
  KPRMILL     5      MFG        01-Dec-21   500.8       602.4       566    ₹57,523       +20.3%      -5.2%     
  KPITTECH    2      IT         01-Dec-21   495.4       552.7       572    ₹32,785       +11.6%      -3.2%     
  ATGL        18     OIL&GAS    01-Sep-21   1,508.6     1,665.5     171    ₹26,816       +10.4%      -1.1%     
  BAJAJHLDNG  104    FIN SVC    01-Oct-21   4,433.3     4,774.2     61     ₹20,796       +7.7%       -2.2%     
  SFL         56     CON DUR    01-Dec-21   1,616.7     1,693.6     175    ₹13,457       +4.8%       +2.5%     
  CENTURYPLY  48     CON DUR    01-Nov-21   569.6       591.3       490    ₹10,591       +3.8%       -1.7%     
  ELGIEQUIP   23     MFG        01-Feb-22   336.5       346.9       909    ₹9,508        +3.1%       -1.0%     
  GRINDWELL   73     MFG        01-Nov-21   1,626.3     1,555.6     171    ₹-12,099      -4.4%       -4.2%     
  SHARDACROP  6      FMCG       01-Feb-22   567.5       504.8       539    ₹-33,790      -11.0%      -2.5%     
  THERMAX     75     ENERGY     01-Feb-22   2,030.4     1,802.4     150    ₹-34,203      -11.2%      -0.4%     
  TIINDIA     174    AUTO       03-Jan-22   1,877.1     1,508.8     164    ₹-60,396      -19.6%      -6.9%     
  RAJESHEXPO  200    CON DUR    03-Jan-22   851.1       682.8       363    ₹-61,099      -19.8%      -12.5%    
  UNOMINDA    76     AUTO       03-Jan-22   595.1       463.1       519    ₹-68,535      -22.2%      -5.0%     
  ECLERX      46     IT         03-Jan-22   927.6       708.9       333    ₹-72,835      -23.6%      -6.3%     

  AFTER: Invested ₹8,182,989 | Cash ₹292,955 | Total ₹8,475,943 | Positions 23/30 | Slot ₹282,531

========================================================================
  REBALANCE #88  —  01 Apr 2022
  NAV: ₹9,423,466  |  Slot: ₹314,116  |  Cash: ₹292,955
========================================================================

  EXITS (13)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  CGPOWER     74     ENERGY     01-Sep-20   21.0        186.1       5661   ₹934,174      +784.3%   577d  
  POONAWALLA  51     FIN SVC    01-Mar-21   120.2       276.5       1557   ₹243,268      +129.9%   396d  
  BSE         4      FIN SVC    01-Dec-21   175.3       297.8       1619   ₹198,245      +69.8%    121d  
  JSWENERGY   29     ENERGY     02-Aug-21   237.3       301.4       1075   ₹68,840       +27.0%    242d  
  TANLA       264    IT         01-Nov-21   1,107.0     1,383.9     252    ₹69,772       +25.0%    151d  
  GRINDWELL   116    MFG        01-Nov-21   1,626.3     1,754.9     171    ₹21,997       +7.9%     151d  
  SHARDACROP  —      OTHER      01-Feb-22   567.5       598.8       539    ₹16,869       +5.5%     59d   
  THERMAX     107    ENERGY     01-Feb-22   2,030.4     1,920.6     150    ₹-16,476      -5.4%     59d   
  ELGIEQUIP   165    MFG        01-Feb-22   336.5       290.1       909    ₹-42,115      -13.8%    59d   
  TIINDIA     211    AUTO       03-Jan-22   1,877.1     1,607.0     164    ₹-44,297      -14.4%    88d   
  ECLERX      108    IT         03-Jan-22   927.6       784.4       333    ₹-47,685      -15.4%    88d   
  RAJESHEXPO  199    CON DUR    03-Jan-22   851.1       687.4       363    ₹-59,432      -19.2%    88d   
  UNOMINDA    289    AUTO       03-Jan-22   595.1       460.4       519    ₹-69,895      -22.6%    88d   

  ENTRIES (16)
  [52w filter blocked 2: RTNINDIA(-33.1%), RENUKA(-23.9%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  BCG         1      IT         8.360    0.33   +2544.3%  -7.7%     101.3       3100   ₹314,094      +23.4%    
  GNFC        2      MFG        3.879    1.15   +213.2%   +97.3%    766.1       410    ₹314,100      +15.4%    
  BDL         5      DEFENCE    2.713    0.72   +73.8%    +47.1%    277.3       1132   ₹313,943      +6.6%     
  GAEL        6      FMCG       2.659    1.13   +107.0%   +61.7%    130.7       2402   ₹314,034      +8.9%     
  GUJALKALI   8      MFG        2.468    1.07   +186.7%   +45.6%    868.3       361    ₹313,453      +19.2%    
  GSFC        9      MFG        2.561    1.06   +119.6%   +44.2%    155.7       2017   ₹314,033      +17.2%    
  RATNAMANI   10     METAL      2.342    0.40   +39.6%    +31.7%    1,673.9     187    ₹313,012      +7.8%     
  POLYPLEX    11     MFG        2.707    0.66   +223.5%   +34.3%    2,280.6     137    ₹312,444      +13.0%    
  HAL         12     DEFENCE    2.241    0.84   +59.8%    +28.3%    723.8       433    ₹313,407      +7.9%     
  COALINDIA   13     ENERGY     2.204    0.81   +62.7%    +31.2%    137.5       2284   ₹314,054      +2.7%     
  VIPIND      14     CON DUR    2.234    0.89   +108.4%   +37.8%    734.0       427    ₹313,424      +9.2%     
  GAIL        16     OIL&GAS    2.114    0.74   +35.0%    +31.3%    93.1        3372   ₹314,090      +8.8%     
  SOLARINDS   18     DEFENCE    2.017    0.85   +123.9%   +21.2%    2,851.5     110    ₹313,664      +10.3%    
  ITC         20     FMCG       1.991    0.81   +26.1%    +20.1%    204.2       1537   ₹313,911      +3.9%     
  CUMMINSIND  21     INFRA      1.943    0.76   +35.1%    +22.9%    1,075.4     292    ₹314,030      +7.3%     
  SRF         22     MFG        1.958    1.19   +150.4%   +9.3%     2,590.2     121    ₹313,413      +3.3%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   3      IT         01-Dec-20   1,527.0     8,464.4     103    ₹714,556      +454.3%     +12.8%    
  PERSISTENT  63     IT         03-Aug-20   466.7       2,292.8     226    ₹412,690      +391.3%     +4.7%     
  TRIDENT     26     CONSUMP    01-Oct-21   27.3        50.3        10038  ₹230,893      +84.3%      +1.2%     
  ATGL        27     OIL&GAS    01-Sep-21   1,508.6     2,246.2     171    ₹126,124      +48.9%      +16.1%    
  ADANIENSOL  12     ENERGY     01-Sep-21   1,659.6     2,421.4     155    ₹118,087      +45.9%      +4.1%     
  CENTURYPLY  32     CON DUR    01-Nov-21   569.6       705.2       490    ₹66,445       +23.8%      +6.1%     
  KPRMILL     78     MFG        01-Dec-21   500.8       613.5       566    ₹63,788       +22.5%      -0.8%     
  KPITTECH    42     IT         01-Dec-21   495.4       596.0       572    ₹57,558       +20.3%      +3.6%     
  BAJAJHLDNG  77     FIN SVC    01-Oct-21   4,433.3     5,052.3     61     ₹37,758       +14.0%      +6.5%     
  SFL         69     CON DUR    01-Dec-21   1,616.7     1,760.5     175    ₹25,165       +8.9%       +3.2%     

  AFTER: Invested ₹9,323,152 | Cash ₹94,353 | Total ₹9,417,506 | Positions 26/30 | Slot ₹314,116

========================================================================
  REBALANCE #89  —  02 May 2022
  NAV: ₹9,280,321  |  Slot: ₹309,344  |  Cash: ₹94,353
========================================================================

  [REGIME OFF] Nifty 500 14,736.3 < SMA200 14,788.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (26)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   39     IT         01-Dec-20   1,527.0     7,286.3     103    ₹593,207      +377.2%     -4.3%     
  PERSISTENT  68     IT         03-Aug-20   466.7       2,013.4     226    ₹349,553      +331.4%     -3.1%     
  TRIDENT     53     CONSUMP    01-Oct-21   27.3        48.4        10038  ₹212,568      +77.6%      -3.3%     
  ADANIENSOL  12     ENERGY     01-Sep-21   1,659.6     2,804.4     155    ₹177,452      +69.0%      +6.0%     
  ATGL        27     OIL&GAS    01-Sep-21   1,508.6     2,484.6     171    ₹166,882      +64.7%      +3.7%     
  BDL         11     DEFENCE    01-Apr-22   277.3       348.6       1132   ₹80,722       +25.7%      -0.1%     
  KPRMILL     83     MFG        01-Dec-21   500.8       611.7       566    ₹62,790       +22.2%      -4.0%     
  GAEL        13     FMCG       01-Apr-22   130.7       157.8       2402   ₹65,114       +20.7%      +1.5%     
  CENTURYPLY  67     CON DUR    01-Nov-21   569.6       650.1       490    ₹39,419       +14.1%      +0.6%     
  BAJAJHLDNG  88     FIN SVC    01-Oct-21   4,433.3     4,983.5     61     ₹33,565       +12.4%      -1.5%     
  SFL         69     CON DUR    01-Dec-21   1,616.7     1,760.6     175    ₹25,191       +8.9%       -3.0%     
  POLYPLEX    3      MFG        01-Apr-22   2,280.6     2,471.2     137    ₹26,116       +8.4%       +2.9%     
  GUJALKALI   18     MFG        01-Apr-22   868.3       923.8       361    ₹20,052       +6.4%       +7.4%     
  KPITTECH    108    IT         01-Dec-21   495.4       526.6       572    ₹17,876       +6.3%       -4.6%     
  ITC         19     FMCG       01-Apr-22   204.2       212.2       1537   ₹12,207       +3.9%       +1.8%     
  HAL         54     DEFENCE    01-Apr-22   723.8       748.0       433    ₹10,465       +3.3%       -1.7%     
  SOLARINDS   22     DEFENCE    01-Apr-22   2,851.5     2,900.5     110    ₹5,394        +1.7%       +2.3%     
  COALINDIA   47     ENERGY     01-Apr-22   137.5       138.6       2284   ₹2,531        +0.8%       -1.2%     
  GSFC        21     MFG        01-Apr-22   155.7       153.0       2017   ₹-5,503       -1.8%       +0.1%     
  GAIL        73     OIL&GAS    01-Apr-22   93.1        90.6        3372   ₹-8,722       -2.8%       -1.1%     
  GNFC        5      MFG        01-Apr-22   766.1       739.7       410    ₹-10,831      -3.4%       -0.4%     
  SRF         65     MFG        01-Apr-22   2,590.2     2,474.6     121    ₹-13,992      -4.5%       -1.6%     
  RATNAMANI   36     METAL      01-Apr-22   1,673.9     1,556.8     187    ₹-21,896      -7.0%       -0.2%     
  VIPIND      40     CON DUR    01-Apr-22   734.0       672.3       427    ₹-26,358      -8.4%       -1.4%     
  CUMMINSIND  81     INFRA      01-Apr-22   1,075.4     975.4       292    ₹-29,201      -9.3%       -2.6%     
  BCG         1      IT         01-Apr-22   101.3       79.2        3100   ₹-68,722      -21.9%      -7.9%     

  AFTER: Invested ₹9,185,967 | Cash ₹94,353 | Total ₹9,280,321 | Positions 26/30 | Slot ₹309,344

========================================================================
  REBALANCE #90  —  01 Jun 2022
  NAV: ₹8,906,662  |  Slot: ₹296,889  |  Cash: ₹94,353
========================================================================

  [REGIME OFF] Nifty 500 14,082.9 < SMA200 14,803.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (26)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   10     IT         01-Dec-20   1,527.0     8,173.1     103    ₹684,546      +435.2%     +5.6%     
  PERSISTENT  133    IT         03-Aug-20   466.7       1,815.3     226    ₹304,772      +288.9%     -0.8%     
  TRIDENT     33     CONSUMP    01-Oct-21   27.3        44.2        10038  ₹169,810      +62.0%      -2.0%     
  ATGL        14     OIL&GAS    01-Sep-21   1,508.6     2,322.5     171    ₹139,170      +53.9%      -2.8%     
  BDL         2      DEFENCE    01-Apr-22   277.3       396.7       1132   ₹135,107      +43.0%      +10.9%    
  HAL         4      DEFENCE    01-Apr-22   723.8       899.0       433    ₹75,879       +24.2%      +9.7%     
  GAEL        6      FMCG       01-Apr-22   130.7       161.4       2402   ₹73,736       +23.5%      +2.8%     
  ADANIENSOL  181    ENERGY     01-Sep-21   1,659.6     1,957.9     155    ₹46,237       +18.0%      -13.5%    
  KPRMILL     93     MFG        01-Dec-21   500.8       580.1       566    ₹44,909       +15.8%      -0.5%     
  ITC         11     FMCG       01-Apr-22   204.2       224.5       1537   ₹31,203       +9.9%       +3.2%     
  KPITTECH    65     IT         01-Dec-21   495.4       532.8       572    ₹21,379       +7.5%       +6.3%     
  BAJAJHLDNG  116    FIN SVC    01-Oct-21   4,433.3     4,726.9     61     ₹17,910       +6.6%       +0.4%     
  COALINDIA   43     ENERGY     01-Apr-22   137.5       144.8       2284   ₹16,702       +5.3%       +6.1%     
  POLYPLEX    5      MFG        01-Apr-22   2,280.6     2,351.3     137    ₹9,691        +3.1%       +7.3%     
  RATNAMANI   40     METAL      01-Apr-22   1,673.9     1,708.2     187    ₹6,415        +2.0%       +4.9%     
  CENTURYPLY  174    CON DUR    01-Nov-21   569.6       561.6       490    ₹-3,944       -1.4%       +1.6%     
  GSFC        22     MFG        01-Apr-22   155.7       152.6       2017   ₹-6,315       -2.0%       +6.9%     
  SFL         189    CON DUR    01-Dec-21   1,616.7     1,491.2     175    ₹-21,954      -7.8% ⚠     -5.3%     
  GAIL        94     OIL&GAS    01-Apr-22   93.1        85.6        3372   ₹-25,294      -8.1%       -1.6%     
  SOLARINDS   38     DEFENCE    01-Apr-22   2,851.5     2,611.8     110    ₹-26,371      -8.4%       -3.9%     
  SRF         69     MFG        01-Apr-22   2,590.2     2,368.5     121    ₹-26,825      -8.6%       +1.5%     
  CUMMINSIND  81     INFRA      01-Apr-22   1,075.4     970.9       292    ₹-30,513      -9.7%       +0.8%     
  GUJALKALI   32     MFG        01-Apr-22   868.3       749.2       361    ₹-42,986      -13.7%      -3.0%     
  VIPIND      145    CON DUR    01-Apr-22   734.0       598.8       427    ₹-57,726      -18.4%      +1.8%     
  GNFC        50     MFG        01-Apr-22   766.1       603.2       410    ₹-66,788      -21.3%      +0.0%     
  BCG         1      IT         01-Apr-22   101.3       60.5        3100   ₹-126,529     -40.3%      -8.2%     
  ⚠  WAZ < 0 (momentum below universe mean): SFL

  AFTER: Invested ₹8,812,308 | Cash ₹94,353 | Total ₹8,906,662 | Positions 26/30 | Slot ₹296,889

========================================================================
  REBALANCE #91  —  01 Jul 2022
  NAV: ₹8,131,621  |  Slot: ₹271,054  |  Cash: ₹94,353
========================================================================

  [REGIME OFF] Nifty 500 13,394.4 < SMA200 14,710.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (26)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   19     IT         01-Dec-20   1,527.0     7,781.4     103    ₹644,202      +409.6%     +1.4%     
  PERSISTENT  265    IT         03-Aug-20   466.7       1,608.9     226    ₹258,126      +244.7% ⚠   -3.5%     
  ATGL        13     OIL&GAS    01-Sep-21   1,508.6     2,385.5     171    ₹149,944      +58.1%      +2.0%     
  ADANIENSOL  18     ENERGY     01-Sep-21   1,659.6     2,400.6     155    ₹114,863      +44.7%      +9.0%     
  TRIDENT     39     CONSUMP    01-Oct-21   27.3        35.7        10038  ₹84,292       +30.8%      -4.8%     
  BDL         10     DEFENCE    01-Apr-22   277.3       325.4       1132   ₹54,440       +17.3%      -9.3%     
  ITC         7      FMCG       01-Apr-22   204.2       234.9       1537   ₹47,072       +15.0%      +5.4%     
  HAL         8      DEFENCE    01-Apr-22   723.8       824.4       433    ₹43,568       +13.9%      -3.3%     
  GAEL        51     FMCG       01-Apr-22   130.7       136.2       2402   ₹13,201       +4.2%       -1.0%     
  KPRMILL     80     MFG        01-Dec-21   500.8       497.8       566    ₹-1,663       -0.6%       -5.5%     
  KPITTECH    44     IT         01-Dec-21   495.4       489.3       572    ₹-3,478       -1.2%       -0.3%     
  COALINDIA   60     ENERGY     01-Apr-22   137.5       135.4       2284   ₹-4,893       -1.6%       -1.1%     
  BAJAJHLDNG  94     FIN SVC    01-Oct-21   4,433.3     4,299.7     61     ₹-8,150       -3.0%       -2.5%     
  SOLARINDS   30     DEFENCE    01-Apr-22   2,851.5     2,713.6     110    ₹-15,167      -4.8%       -0.9%     
  RATNAMANI   131    METAL      01-Apr-22   1,673.9     1,542.1     187    ₹-24,643      -7.9%       -4.7%     
  CUMMINSIND  118    INFRA      01-Apr-22   1,075.4     970.8       292    ₹-30,555      -9.7%       +2.3%     
  POLYPLEX    41     MFG        01-Apr-22   2,280.6     2,055.3     137    ₹-30,867      -9.9%       -0.7%     
  CENTURYPLY  245    CON DUR    01-Nov-21   569.6       509.8       490    ₹-29,315      -10.5% ⚠    -2.1%     
  VIPIND      83     CON DUR    01-Apr-22   734.0       611.8       427    ₹-52,188      -16.7%      +2.0%     
  SFL         246    CON DUR    01-Dec-21   1,616.7     1,335.7     175    ₹-49,175      -17.4% ⚠    -2.6%     
  SRF         98     MFG        01-Apr-22   2,590.2     2,135.5     121    ₹-55,012      -17.6%      -4.4%     
  GAIL        284    OIL&GAS    01-Apr-22   93.1        76.1        3372   ₹-57,565      -18.3% ⚠    -5.1%     
  GSFC        184    MFG        01-Apr-22   155.7       116.5       2017   ₹-79,027      -25.2%      -7.1%     
  GUJALKALI   91     MFG        01-Apr-22   868.3       631.9       361    ₹-85,343      -27.2%      -4.3%     
  GNFC        134    MFG        01-Apr-22   766.1       516.0       410    ₹-102,558     -32.7%      -3.7%     
  BCG         46     IT         01-Apr-22   101.3       32.6        3100   ₹-212,932     -67.8%      -22.2%    
  ⚠  WAZ < 0 (momentum below universe mean): SFL, CENTURYPLY, PERSISTENT, GAIL

  AFTER: Invested ₹8,037,267 | Cash ₹94,353 | Total ₹8,131,621 | Positions 26/30 | Slot ₹271,054

========================================================================
  REBALANCE #92  —  01 Aug 2022
  NAV: ₹9,343,364  |  Slot: ₹311,445  |  Cash: ₹94,353
========================================================================

  EXITS (16)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  PERSISTENT  256    IT         03-Aug-20   466.7       1,780.3     226    ₹296,864      +281.5%   728d  
  TRIDENT     109    CONSUMP    01-Oct-21   27.3        37.1        10038  ₹98,389       +35.9%    304d  
  GAEL        138    FMCG       01-Apr-22   130.7       152.8       2402   ₹52,879       +16.8%    122d  
  KPRMILL     153    MFG        01-Dec-21   500.8       568.8       566    ₹38,533       +13.6%    243d  
  BAJAJHLDNG  101    FIN SVC    01-Oct-21   4,433.3     4,946.0     61     ₹31,276       +11.6%    304d  
  CENTURYPLY  161    CON DUR    01-Nov-21   569.6       586.5       490    ₹8,254        +3.0%     273d  
  POLYPLEX    89     MFG        01-Apr-22   2,280.6     2,299.7     137    ₹2,615        +0.8%     122d  
  GSFC        149    MFG        01-Apr-22   155.7       149.5       2017   ₹-12,449      -4.0%     122d  
  SOLARINDS   128    DEFENCE    01-Apr-22   2,851.5     2,684.1     110    ₹-18,412      -5.9%     122d  
  SRF         71     MFG        01-Apr-22   2,590.2     2,427.5     121    ₹-19,685      -6.3%     122d  
  SFL         276    CON DUR    01-Dec-21   1,616.7     1,486.9     175    ₹-22,706      -8.0%     243d  
  GAIL        237    OIL&GAS    01-Apr-22   93.1        85.2        3372   ₹-26,628      -8.5%     122d  
  GNFC        91     MFG        01-Apr-22   766.1       668.3       410    ₹-40,102      -12.8%    122d  
  VIPIND      146    CON DUR    01-Apr-22   734.0       619.6       427    ₹-48,869      -15.6%    122d  
  GUJALKALI   214    MFG        01-Apr-22   868.3       705.8       361    ₹-58,667      -18.7%    122d  
  BCG         227    IT         01-Apr-22   101.3       47.6        3100   ₹-166,656     -53.1%    122d  

  ENTRIES (15)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TVSMOTOR    2      AUTO       3.885    0.78   +62.0%    +47.7%    911.5       341    ₹310,818      +7.4%     
  CGPOWER     4      ENERGY     3.753    0.86   +183.4%   +24.5%    222.0       1402   ₹311,269      +6.7%     
  M&M         7      AUTO       3.771    0.97   +71.6%    +39.5%    1,193.6     260    ₹310,336      +8.2%     
  SIEMENS     8      ENERGY     2.757    0.88   +42.9%    +25.4%    1,598.0     194    ₹310,010      +3.8%     
  AIAENG      9      MFG        2.624    0.50   +25.3%    +30.1%    2,434.5     127    ₹309,182      +5.5%     
  VBL         10     FMCG       2.999    0.63   +83.6%    +27.4%    183.1       1700   ₹311,304      +7.8%     
  BEL         12     DEFENCE    2.553    0.89   +52.8%    +22.9%    90.6        3438   ₹311,363      +10.0%    
  FINEORG     13     MFG        2.625    0.87   +87.6%    +25.3%    5,475.9     56     ₹306,652      +7.1%     
  BLUEDART    14     INFRA      2.953    0.74   +58.4%    +26.6%    8,621.4     36     ₹310,370      +6.6%     
  SHOPERSTOP  16     CONSUMP    2.995    0.98   +151.3%   +25.8%    608.6       511    ₹310,995      +11.7%    
  EICHERMOT   17     AUTO       2.262    0.96   +21.8%    +24.4%    2,962.8     105    ₹311,096      +2.7%     
  MRF         18     MFG        2.251    0.67   +9.2%     +21.2%    86,534.8    3      ₹259,605      +9.6%     
  SBILIFE     19     FIN SVC    2.191    0.87   +16.1%    +21.8%    1,303.6     238    ₹310,251      +11.7%    
  MARUTI      21     AUTO       2.198    0.89   +26.5%    +21.7%    8,684.2     35     ₹303,948      +4.4%     
  HINDUNILVR  22     FMCG       2.157    0.69   +11.5%    +20.5%    2,413.3     129    ₹311,321      +2.4%     

  HOLDS (10)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TATAELXSI   19     IT         01-Dec-20   1,527.0     8,324.1     103    ₹700,100      +445.1%     +5.1%     
  ATGL        1      OIL&GAS    01-Sep-21   1,508.6     3,213.2     171    ₹291,487      +113.0%     +13.8%    
  ADANIENSOL  2      ENERGY     01-Sep-21   1,659.6     3,261.8     155    ₹248,333      +96.5%      +13.6%    
  BDL         18     DEFENCE    01-Apr-22   277.3       400.5       1132   ₹139,469      +44.4%      +11.2%    
  HAL         6      DEFENCE    01-Apr-22   723.8       963.6       433    ₹103,811      +33.1%      +8.3%     
  ITC         7      FMCG       01-Apr-22   204.2       254.0       1537   ₹76,461       +24.4%      +3.9%     
  COALINDIA   29     ENERGY     01-Apr-22   137.5       157.1       2284   ₹44,877       +14.3%      +7.2%     
  KPITTECH    57     IT         01-Dec-21   495.4       535.7       572    ₹23,060       +8.1%       +5.0%     
  CUMMINSIND  12     INFRA      01-Apr-22   1,075.4     1,157.0     292    ₹23,800       +7.6%       +6.1%     
  RATNAMANI   60     METAL      01-Apr-22   1,673.9     1,742.8     187    ₹12,888       +4.1%       +5.6%     

  AFTER: Invested ₹9,101,016 | Cash ₹236,887 | Total ₹9,337,903 | Positions 25/30 | Slot ₹311,445

========================================================================
  REBALANCE #93  —  01 Sep 2022
  NAV: ₹9,910,429  |  Slot: ₹330,348  |  Cash: ₹236,887
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TATAELXSI   93     IT         01-Dec-20   1,527.0     8,610.2     103    ₹729,566      +463.9%   639d  
  BDL         73     DEFENCE    01-Apr-22   277.3       405.3       1132   ₹144,824      +46.1%    153d  
  KPITTECH    113    IT         01-Dec-21   495.4       560.7       572    ₹37,337       +13.2%    274d  
  RATNAMANI   155    METAL      01-Apr-22   1,673.9     1,789.5     187    ₹21,633       +6.9%     153d  
  AIAENG      74     MFG        01-Aug-22   2,434.5     2,582.5     127    ₹18,791       +6.1%     31d   
  MARUTI      77     AUTO       01-Aug-22   8,684.2     8,761.9     35     ₹2,718        +0.9%     31d   
  HINDUNILVR  139    FMCG       01-Aug-22   2,413.3     2,425.4     129    ₹1,561        +0.5%     31d   
  BLUEDART    49     INFRA      01-Aug-22   8,621.4     8,626.0     36     ₹167          +0.1%     31d   
  SBILIFE     124    FIN SVC    01-Aug-22   1,303.6     1,287.0     238    ₹-3,957       -1.3%     31d   
  MRF         119    MFG        01-Aug-22   86,534.8    84,735.9    3      ₹-5,397       -2.1%     31d   

  ENTRIES (12)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  KARURVYSYA  4      PVT BNK    3.360    1.12   +80.1%    +60.4%    57.0        5791   ₹330,317      +11.0%    
  RELINFRA    5      INFRA      3.566    0.92   +185.5%   +79.4%    193.9       1704   ₹330,320      +38.7%    
  GRINDWELL   8      MFG        2.654    0.78   +78.4%    +29.8%    2,168.7     152    ₹329,636      +6.8%     
  MAZDOCK     10     DEFENCE    2.722    1.01   +77.9%    +41.9%    193.0       1711   ₹330,194      +23.0%    
  FEDERALBNK  11     PVT BNK    2.646    1.20   +54.6%    +35.3%    116.5       2836   ₹330,293      +6.9%     
  SOLARINDS   14     DEFENCE    2.526    0.56   +96.2%    +26.9%    3,315.3     99     ₹328,216      +2.9%     
  TIINDIA     15     AUTO       2.651    0.83   +78.8%    +41.4%    2,251.6     146    ₹328,726      +4.5%     
  GESHIP      16     INFRA      3.214    0.63   +71.3%    +48.1%    500.3       660    ₹330,183      +8.9%     
  PIDILITIND  17     MFG        2.379    0.78   +27.1%    +28.4%    1,387.6     238    ₹330,256      +6.8%     
  ELGIEQUIP   18     MFG        2.541    0.69   +155.3%   +30.9%    487.1       678    ₹330,283      +8.9%     
  KRBL        19     FMCG       2.294    0.65   +38.8%    +46.4%    317.2       1041   ₹330,198      +15.3%    
  PAGEIND     20     MFG        2.279    0.87   +67.6%    +14.8%    48,180.5    6      ₹289,083      +3.0%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        2      OIL&GAS    01-Sep-21   1,508.6     3,634.3     171    ₹363,482      +140.9%     +7.4%     
  ADANIENSOL  1      ENERGY     01-Sep-21   1,659.6     3,880.4     155    ₹344,232      +133.8%     +8.5%     
  HAL         21     DEFENCE    01-Apr-22   723.8       1,098.4     433    ₹162,181      +51.7%      +5.1%     
  ITC         16     FMCG       01-Apr-22   204.2       262.3       1537   ₹89,283       +28.4%      +2.0%     
  COALINDIA   26     ENERGY     01-Apr-22   137.5       172.5       2284   ₹79,861       +25.4%      +4.6%     
  BEL         10     DEFENCE    01-Aug-22   90.6        102.5       3438   ₹40,916       +13.1%      +9.4%     
  VBL         9      FMCG       01-Aug-22   183.1       206.6       1700   ₹39,902       +12.8%      +5.3%     
  EICHERMOT   41     AUTO       01-Aug-22   2,962.8     3,294.7     105    ₹34,843       +11.2%      +3.5%     
  FINEORG     30     MFG        01-Aug-22   5,475.9     6,045.1     56     ₹31,874       +10.4%      +1.1%     
  TVSMOTOR    4      AUTO       01-Aug-22   911.5       998.5       341    ₹29,668       +9.5%       +6.7%     
  CUMMINSIND  59     INFRA      01-Apr-22   1,075.4     1,166.6     292    ₹26,629       +8.5%       +1.9%     
  M&M         12     AUTO       01-Aug-22   1,193.6     1,265.2     260    ₹18,629       +6.0%       +4.5%     
  SIEMENS     46     ENERGY     01-Aug-22   1,598.0     1,691.8     194    ₹18,204       +5.9%       +3.0%     
  SHOPERSTOP  11     CONSUMP    01-Aug-22   608.6       622.3       511    ₹7,026        +2.3%       +2.8%     
  CGPOWER     7      ENERGY     01-Aug-22   222.0       224.0       1402   ₹2,839        +0.9%       -0.1%     

  AFTER: Invested ₹9,771,734 | Cash ₹134,043 | Total ₹9,905,777 | Positions 27/30 | Slot ₹330,348

========================================================================
  REBALANCE #94  —  03 Oct 2022
  NAV: ₹9,798,363  |  Slot: ₹326,612  |  Cash: ₹134,043
========================================================================

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COALINDIA   79     ENERGY     01-Apr-22   137.5       161.0       2284   ₹53,605       +17.1%    185d  
  CUMMINSIND  90     INFRA      01-Apr-22   1,075.4     1,129.1     292    ₹15,663       +5.0%     185d  
  M&M         47     AUTO       01-Aug-22   1,193.6     1,206.7     260    ₹3,410        +1.1%     63d   
  SIEMENS     92     ENERGY     01-Aug-22   1,598.0     1,568.0     194    ₹-5,826       -1.9%     63d   
  FEDERALBNK  41     PVT BNK    01-Sep-22   116.5       114.1       2836   ₹-6,814       -2.1%     32d   
  PIDILITIND  67     MFG        01-Sep-22   1,387.6     1,306.9     238    ₹-19,210      -5.8%     32d   
  GESHIP      29     INFRA      01-Sep-22   500.3       467.2       660    ₹-21,808      -6.6%     32d   
  RELINFRA    85     INFRA      01-Sep-22   193.9       135.4       1704   ₹-99,514      -30.1%    32d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TATAINVEST  2      FIN SVC    3.866    0.95   +87.1%    +73.3%    224.3       1455   ₹326,423      +3.4%     
  COCHINSHIP  3      DEFENCE    3.731    0.52   +42.0%    +56.8%    230.3       1417   ₹326,402      +17.8%    
  RITES       6      INFRA      3.184    0.47   +32.6%    +47.5%    149.3       2188   ₹326,597      +11.1%    
  TRITURBINE  11     ENERGY     2.905    0.80   +82.3%    +65.7%    250.8       1302   ₹326,546      +12.3%    
  CEATLTD     13     AUTO       2.797    0.64   +18.6%    +67.3%    1,527.0     213    ₹325,258      +3.0%     
  BAJAJHLDNG  14     FIN SVC    2.635    0.83   +38.5%    +44.7%    6,221.9     52     ₹323,538      +1.1%     
  BDL         15     DEFENCE    2.748    0.80   +134.8%   +31.4%    427.7       763    ₹326,344      +1.1%     
  RHIM        16     METAL      2.597    0.62   +91.4%    +33.6%    661.6       493    ₹326,175      +4.6%     

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        24     OIL&GAS    01-Sep-21   1,508.6     3,096.6     171    ₹271,533      +105.3%     -10.2%    
  ADANIENSOL  30     ENERGY     01-Sep-21   1,659.6     3,123.6     155    ₹226,928      +88.2%      -15.1%    
  HAL         13     DEFENCE    01-Apr-22   723.8       1,092.9     433    ₹159,806      +51.0%      -3.3%     
  ITC         40     FMCG       01-Apr-22   204.2       267.9       1537   ₹97,915       +31.2%      -1.9%     
  MAZDOCK     1      DEFENCE    01-Sep-22   193.0       247.0       1711   ₹92,496       +28.0%      +17.6%    
  SHOPERSTOP  2      CONSUMP    01-Aug-22   608.6       750.9       511    ₹72,715       +23.4%      +5.7%     
  FINEORG     14     MFG        01-Aug-22   5,475.9     6,593.5     56     ₹62,584       +20.4%      -2.5%     
  SOLARINDS   9      DEFENCE    01-Sep-22   3,315.3     3,959.1     99     ₹63,732       +19.4%      +8.9%     
  TIINDIA     8      AUTO       01-Sep-22   2,251.6     2,688.0     146    ₹63,720       +19.4%      +3.6%     
  KRBL        16     FMCG       01-Sep-22   317.2       368.4       1041   ₹53,302       +16.1%      +7.7%     
  VBL         18     FMCG       01-Aug-22   183.1       212.1       1700   ₹49,337       +15.8%      +0.2%     
  EICHERMOT   59     AUTO       01-Aug-22   2,962.8     3,344.6     105    ₹40,091       +12.9%      -2.5%     
  KARURVYSYA  4      PVT BNK    01-Sep-22   57.0        63.3        5791   ₹36,521       +11.1%      +0.2%     
  TVSMOTOR    26     AUTO       01-Aug-22   911.5       979.2       341    ₹23,094       +7.4%       -2.8%     
  BEL         35     DEFENCE    01-Aug-22   90.6        94.6        3438   ₹13,983       +4.5%       -5.2%     
  CGPOWER     27     ENERGY     01-Aug-22   222.0       226.8       1402   ₹6,717        +2.2%       -3.2%     
  PAGEIND     36     MFG        01-Sep-22   48,180.5    47,181.7    6      ₹-5,993       -2.1%       -0.9%     
  GRINDWELL   42     MFG        01-Sep-22   2,168.7     2,042.5     152    ₹-19,177      -5.8%       -1.9%     
  ELGIEQUIP   50     MFG        01-Sep-22   487.1       418.3       678    ₹-46,656      -14.1%      -7.7%     

  AFTER: Invested ₹9,782,616 | Cash ₹12,651 | Total ₹9,795,268 | Positions 27/30 | Slot ₹326,612

========================================================================
  REBALANCE #95  —  01 Nov 2022
  NAV: ₹10,560,320  |  Slot: ₹352,011  |  Cash: ₹12,651
========================================================================

  EXITS (3)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  FINEORG     108    MFG        01-Aug-22   5,475.9     5,612.0     56     ₹7,620        +2.5%     92d   
  PAGEIND     101    MFG        01-Sep-22   48,180.5    47,879.2    6      ₹-1,807       -0.6%     61d   
  CEATLTD     85     AUTO       03-Oct-22   1,527.0     1,515.1     213    ₹-2,541       -0.8%     29d   

  ENTRIES (2)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  AMBUJACEM   8      INFRA      2.889    0.86   +46.1%    +46.7%    534.7       658    ₹351,804      +5.5%     
  KPITTECH    13     IT         2.760    1.08   +122.9%   +34.3%    705.3       499    ₹351,934      +4.8%     

  HOLDS (24)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ATGL        11     OIL&GAS    01-Sep-21   1,508.6     3,676.2     171    ₹370,646      +143.7%     +9.2%     
  ADANIENSOL  67     ENERGY     01-Sep-21   1,659.6     3,342.4     155    ₹260,842      +101.4%     +0.9%     
  HAL         8      DEFENCE    01-Apr-22   723.8       1,207.2     433    ₹209,314      +66.8%      +3.3%     
  MAZDOCK     1      DEFENCE    01-Sep-22   193.0       307.1       1711   ₹195,266      +59.1%      +6.2%     
  ITC         19     FMCG       01-Apr-22   204.2       288.7       1537   ₹129,780      +41.3%      +2.5%     
  KARURVYSYA  3      PVT BNK    01-Sep-22   57.0        80.1        5791   ₹133,755      +40.5%      +9.8%     
  SHOPERSTOP  13     CONSUMP    01-Aug-22   608.6       762.4       511    ₹78,592       +25.3%      +1.3%     
  EICHERMOT   28     AUTO       01-Aug-22   2,962.8     3,668.2     105    ₹74,065       +23.8%      +4.3%     
  TIINDIA     17     AUTO       01-Sep-22   2,251.6     2,785.9     146    ₹78,018       +23.7%      +2.9%     
  TVSMOTOR    7      AUTO       01-Aug-22   911.5       1,119.4     341    ₹70,891       +22.8%      +2.6%     
  VBL         15     FMCG       01-Aug-22   183.1       219.4       1700   ₹61,738       +19.8%      +5.2%     
  SOLARINDS   10     DEFENCE    01-Sep-22   3,315.3     3,942.4     99     ₹62,079       +18.9%      +2.0%     
  KRBL        24     FMCG       01-Sep-22   317.2       375.6       1041   ₹60,770       +18.4%      -0.9%     
  CGPOWER     42     ENERGY     01-Aug-22   222.0       258.9       1402   ₹51,728       +16.6%      +3.5%     
  BEL         31     DEFENCE    01-Aug-22   90.6        104.3       3438   ₹47,195       +15.2%      +3.1%     
  RITES       5      INFRA      03-Oct-22   149.3       169.5       2188   ₹44,263       +13.6%      +5.3%     
  COCHINSHIP  2      DEFENCE    03-Oct-22   230.3       261.2       1417   ₹43,675       +13.4%      +6.8%     
  BDL         20     DEFENCE    03-Oct-22   427.7       472.9       763    ₹34,445       +10.6%      +4.1%     
  TRITURBINE  14     ENERGY     03-Oct-22   250.8       272.5       1302   ₹28,245       +8.6%       +2.8%     
  BAJAJHLDNG  30     FIN SVC    03-Oct-22   6,221.9     6,419.1     52     ₹10,255       +3.2%       +2.0%     
  TATAINVEST  6      FIN SVC    03-Oct-22   224.3       229.8       1455   ₹7,883        +2.4%       +1.4%     
  ELGIEQUIP   22     MFG        01-Sep-22   487.1       484.9       678    ₹-1,543       -0.5%       +4.1%     
  GRINDWELL   52     MFG        01-Sep-22   2,168.7     2,093.1     152    ₹-11,484      -3.5%       +3.2%     
  RHIM        38     METAL      03-Oct-22   661.6       629.2       493    ₹-15,990      -4.9%       -6.2%     

  AFTER: Invested ₹10,327,142 | Cash ₹232,343 | Total ₹10,559,484 | Positions 26/30 | Slot ₹352,011

========================================================================
  REBALANCE #96  —  01 Dec 2022
  NAV: ₹10,914,723  |  Slot: ₹363,824  |  Cash: ₹232,343
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ATGL        55     OIL&GAS    01-Sep-21   1,508.6     3,607.5     171    ₹358,897      +139.1%   456d  
  ADANIENSOL  289    ENERGY     01-Sep-21   1,659.6     2,834.2     155    ₹182,071      +70.8%    456d  
  EICHERMOT   91     AUTO       01-Aug-22   2,962.8     3,319.6     105    ₹37,464       +12.0%    122d  
  KPITTECH    57     IT         01-Nov-22   705.3       700.5       499    ₹-2,379       -0.7%     30d   
  ELGIEQUIP   109    MFG        01-Sep-22   487.1       481.1       678    ₹-4,092       -1.2%     91d   
  BAJAJHLDNG  76     FIN SVC    03-Oct-22   6,221.9     6,038.0     52     ₹-9,561       -3.0%     59d   
  GRINDWELL   271    MFG        01-Sep-22   2,168.7     1,902.9     152    ₹-40,391      -12.3%    91d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  RVNL        2      PSE        5.459    0.86   +128.6%   +129.8%   72.2        5036   ₹363,775      +19.4%    
  IRFC        3      FIN SVC    4.918    0.46   +55.2%    +69.4%    32.4        11228  ₹363,812      +18.8%    
  UNIONBANK   5      PSU BNK    4.213    1.16   +96.9%    +92.4%    72.2        5040   ₹363,756      +14.1%    
  GODFRYPHLP  7      FMCG       3.149    0.87   +57.5%    +63.7%    577.8       629    ₹363,412      +5.4%     
  INDIANB     10     PSU BNK    2.984    1.19   +94.8%    +42.1%    250.0       1455   ₹363,685      +4.6%     
  CUMMINSIND  13     INFRA      2.719    0.77   +67.1%    +20.6%    1,372.2     265    ₹363,620      +5.9%     
  BANKINDIA   14     PSU BNK    2.938    1.19   +53.1%    +60.1%    73.9        4922   ₹363,758      +9.9%     
  HUDCO       15     FIN SVC    2.643    0.87   +44.8%    +40.3%    47.2        7702   ₹363,820      +14.1%    

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MAZDOCK     1      DEFENCE    01-Sep-22   193.0       443.8       1711   ₹429,172      +130.0%     +10.9%    
  HAL         8      DEFENCE    01-Apr-22   723.8       1,323.9     433    ₹259,855      +82.9%      +4.1%     
  KARURVYSYA  9      PVT BNK    01-Sep-22   57.0        80.3        5791   ₹134,686      +40.8%      -0.1%     
  COCHINSHIP  4      DEFENCE    03-Oct-22   230.3       321.4       1417   ₹128,976      +39.5%      +6.8%     
  ITC         30     FMCG       01-Apr-22   204.2       280.5       1537   ₹117,148      +37.3%      -0.9%     
  VBL         17     FMCG       01-Aug-22   183.1       250.6       1700   ₹114,712      +36.8%      +9.5%     
  KRBL        26     FMCG       01-Sep-22   317.2       423.5       1041   ₹110,657      +33.5%      +8.6%     
  TIINDIA     37     AUTO       01-Sep-22   2,251.6     2,806.1     146    ₹80,970       +24.6%      +5.2%     
  CGPOWER     25     ENERGY     01-Aug-22   222.0       276.0       1402   ₹75,688       +24.3%      +3.4%     
  SOLARINDS   43     DEFENCE    01-Sep-22   3,315.3     4,064.7     99     ₹74,190       +22.6%      +4.1%     
  RHIM        12     METAL      03-Oct-22   661.6       788.3       493    ₹62,475       +19.2%      +11.6%    
  SHOPERSTOP  27     CONSUMP    01-Aug-22   608.6       700.9       511    ₹47,165       +15.2%      -0.0%     
  TRITURBINE  28     ENERGY     03-Oct-22   250.8       288.1       1302   ₹48,566       +14.9%      +4.6%     
  TVSMOTOR    56     AUTO       01-Aug-22   911.5       1,032.8     341    ₹41,374       +13.3%      -2.1%     
  RITES       31     INFRA      03-Oct-22   149.3       167.3       2188   ₹39,457       +12.1%      -1.4%     
  BEL         61     DEFENCE    01-Aug-22   90.6        100.1       3438   ₹32,820       +10.5%      -2.3%     
  BDL         20     DEFENCE    03-Oct-22   427.7       471.0       763    ₹33,011       +10.1%      +1.0%     
  AMBUJACEM   15     INFRA      01-Nov-22   534.7       570.9       658    ₹23,861       +6.8%       +3.4%     
  TATAINVEST  13     FIN SVC    03-Oct-22   224.3       226.3       1455   ₹2,876        +0.9%       -1.1%     

  AFTER: Invested ₹10,908,307 | Cash ₹2,961 | Total ₹10,911,268 | Positions 27/30 | Slot ₹363,824

========================================================================
  REBALANCE #97  —  02 Jan 2023
  NAV: ₹10,626,708  |  Slot: ₹354,224  |  Cash: ₹2,961
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  MAZDOCK     4      DEFENCE    01-Sep-22   193.0       383.6       1711   ₹326,121      +98.8%    123d  
  TIINDIA     96     AUTO       01-Sep-22   2,251.6     2,773.5     146    ₹76,201       +23.2%    123d  
  SHOPERSTOP  51     CONSUMP    01-Aug-22   608.6       714.6       511    ₹54,166       +17.4%    154d  
  BANKINDIA   3      PSU BNK    01-Dec-22   73.9        81.4        4922   ₹36,770       +10.1%    32d   
  INDIANB     9      PSU BNK    01-Dec-22   250.0       267.8       1455   ₹25,954       +7.1%     32d   
  UNIONBANK   6      PSU BNK    01-Dec-22   72.2        72.1        5040   ₹-446         -0.1%     32d   
  TRITURBINE  123    ENERGY     03-Oct-22   250.8       245.0       1302   ₹-7,604       -2.3%     91d   
  AMBUJACEM   107    INFRA      01-Nov-22   534.7       517.4       658    ₹-11,348      -3.2%     62d   
  TATAINVEST  119    FIN SVC    03-Oct-22   224.3       204.7       1455   ₹-28,621      -8.8%     91d   

  ENTRIES (9)
  [52w filter blocked 1: CENTRALBK(-22.7%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  PFC         4      FIN SVC    3.664    0.91   +39.1%    +46.6%    102.6       3452   ₹354,128      +7.9%     
  RECLTD      6      FIN SVC    3.145    0.92   +38.7%    +34.6%    100.3       3532   ₹354,182      +6.9%     
  MAHABANK    9      PSU BNK    3.357    1.03   +61.8%    +71.6%    27.0        13140  ₹354,209      +2.5%     
  KPIL        10     INFRA      3.027    0.35   +58.3%    +34.1%    545.2       649    ₹353,863      +5.0%     
  RCF         15     MFG        2.722    1.17   +85.1%    +45.1%    126.2       2806   ₹354,098      +5.7%     
  IIFL        16     FIN SVC    2.667    0.86   +75.8%    +35.4%    466.1       759    ₹353,768      +0.0%     
  GESHIP      17     INFRA      3.394    0.64   +137.6%   +26.6%    598.1       592    ₹354,051      -0.1%     
  JYOTHYLAB   23     FMCG       2.163    0.33   +52.7%    +8.5%     194.7       1819   ₹354,099      +0.1%     
  REDINGTON   24     IT         2.194    1.05   +36.3%    +34.8%    168.1       2107   ₹354,085      +3.3%     

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         23     DEFENCE    01-Apr-22   723.8       1,220.7     433    ₹215,145      +68.6%      -1.7%     
  KARURVYSYA  11     PVT BNK    01-Sep-22   57.0        88.9        5791   ₹184,233      +55.8%      +1.8%     
  VBL         15     FMCG       01-Aug-22   183.1       264.3       1700   ₹137,931      +44.3%      +0.3%     
  ITC         39     FMCG       01-Apr-22   204.2       274.9       1537   ₹108,643      +34.6%      -1.0%     
  SOLARINDS   27     DEFENCE    01-Sep-22   3,315.3     4,419.1     99     ₹109,275      +33.3%      +7.3%     
  RHIM        17     METAL      03-Oct-22   661.6       859.3       493    ₹97,473       +29.9%      +8.1%     
  CGPOWER     64     ENERGY     01-Aug-22   222.0       267.3       1402   ₹63,500       +20.4%      +0.6%     
  KRBL        81     FMCG       01-Sep-22   317.2       381.2       1041   ₹66,663       +20.2%      -2.0%     
  TVSMOTOR    41     AUTO       01-Aug-22   911.5       1,055.5     341    ₹49,106       +15.8%      +1.7%     
  GODFRYPHLP  5      FMCG       01-Dec-22   577.8       637.9       629    ₹37,818       +10.4%      +6.6%     
  COCHINSHIP  42     DEFENCE    03-Oct-22   230.3       254.2       1417   ₹33,789       +10.4%      -7.8%     
  BDL         29     DEFENCE    03-Oct-22   427.7       461.4       763    ₹25,708       +7.9%       +2.3%     
  BEL         88     DEFENCE    01-Aug-22   90.6        96.4        3438   ₹19,931       +6.4%       -1.1%     
  RITES       82     INFRA      03-Oct-22   149.3       153.5       2188   ₹9,177        +2.8%       -0.8%     
  HUDCO       20     FIN SVC    01-Dec-22   47.2        47.5        7702   ₹2,052        +0.6%       +3.9%     
  CUMMINSIND  37     INFRA      01-Dec-22   1,372.2     1,322.3     265    ₹-13,201      -3.6%       -1.9%     
  IRFC        8      FIN SVC    01-Dec-22   32.4        31.2        11228  ₹-13,829      -3.8%       +3.0%     
  RVNL        2      PSE        01-Dec-22   72.2        66.0        5036   ₹-31,506      -8.7%       +1.2%     

  AFTER: Invested ₹10,273,151 | Cash ₹349,773 | Total ₹10,622,924 | Positions 27/30 | Slot ₹354,224

========================================================================
  REBALANCE #98  —  01 Feb 2023
  NAV: ₹10,184,320  |  Slot: ₹339,477  |  Cash: ₹349,773
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  KRBL        61     FMCG       01-Sep-22   317.2       376.0       1041   ₹61,227       +18.5%    153d  
  TVSMOTOR    64     AUTO       01-Aug-22   911.5       1,001.7     341    ₹30,758       +9.9%     184d  
  BDL         52     DEFENCE    03-Oct-22   427.7       444.1       763    ₹12,519       +3.8%     121d  
  RITES       129    INFRA      03-Oct-22   149.3       152.5       2188   ₹7,070        +2.2%     121d  
  COCHINSHIP  149    DEFENCE    03-Oct-22   230.3       232.2       1417   ₹2,592        +0.8%     121d  
  BEL         176    DEFENCE    01-Aug-22   90.6        87.4        3438   ₹-10,802      -3.5%     184d  
  RCF         55     MFG        02-Jan-23   126.2       108.1       2806   ₹-50,872      -14.4%    30d   

  ENTRIES (7)
  [52w filter blocked 1: CENTRALBK(-31.1%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JSL         3      METAL      3.475    1.18   +28.4%    +78.3%    257.7       1317   ₹339,359      +6.3%     
  CIEINDIA    4      AUTO       3.392    0.68   +93.9%    +32.7%    382.9       886    ₹339,228      +9.3%     
  SWANCORP    8      ENERGY     3.159    0.79   +86.9%    +47.4%    311.1       1091   ₹339,439      -5.2%     
  RATNAMANI   9      METAL      2.863    0.34   +79.0%    +13.7%    2,223.4     152    ₹337,953      +11.2%    
  BRITANNIA   16     FMCG       2.713    0.55   +27.3%    +17.4%    4,187.2     81     ₹339,162      +0.4%     
  JKLAKSHMI   20     MFG        2.485    1.01   +37.5%    +33.4%    757.9       447    ₹338,761      +3.0%     
  AEGISLOG    22     INFRA      2.421    0.83   +82.8%    +16.7%    349.3       971    ₹339,171      +2.0%     

  HOLDS (20)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         42     DEFENCE    01-Apr-22   723.8       1,136.0     433    ₹178,497      +57.0%      -5.1%     
  KARURVYSYA  10     PVT BNK    01-Sep-22   57.0        84.0        5791   ₹155,854      +47.2%      -2.5%     
  ITC         5      FMCG       01-Apr-22   204.2       298.5       1537   ₹144,887      +46.2%      +6.2%     
  CGPOWER     18     ENERGY     01-Aug-22   222.0       298.7       1402   ₹107,542      +34.5%      +1.9%     
  VBL         21     FMCG       01-Aug-22   183.1       232.5       1700   ₹83,929       +27.0%      -4.7%     
  SOLARINDS   36     DEFENCE    01-Sep-22   3,315.3     3,931.0     99     ₹60,955       +18.6%      -5.1%     
  RHIM        24     METAL      03-Oct-22   661.6       777.1       493    ₹56,950       +17.5%      -4.5%     
  IIFL        16     FIN SVC    02-Jan-23   466.1       508.8       759    ₹32,384       +9.2%       +7.2%     
  JYOTHYLAB   34     FMCG       02-Jan-23   194.7       196.0       1819   ₹2,338        +0.7%       +0.9%     
  GODFRYPHLP  28     FMCG       01-Dec-22   577.8       578.5       629    ₹480          +0.1%       -5.4%     
  CUMMINSIND  17     INFRA      01-Dec-22   1,372.2     1,361.3     265    ₹-2,870       -0.8%       -0.0%     
  RECLTD      14     FIN SVC    02-Jan-23   100.3       97.8        3532   ₹-8,678       -2.5%       -2.1%     
  REDINGTON   47     IT         02-Jan-23   168.1       163.5       2107   ₹-9,527       -2.7%       -0.2%     
  RVNL        1      PSE        01-Dec-22   72.2        69.9        5036   ₹-11,633      -3.2%       -1.7%     
  MAHABANK    15     PSU BNK    02-Jan-23   27.0        25.3        13140  ₹-21,450      -6.1%       -4.9%     
  GESHIP      11     INFRA      02-Jan-23   598.1       556.8       592    ₹-24,420      -6.9%       -1.9%     
  IRFC        3      FIN SVC    01-Dec-22   32.4        29.8        11228  ₹-28,722      -7.9%       -2.7%     
  KPIL        45     INFRA      02-Jan-23   545.2       497.4       649    ₹-31,039      -8.8%       -1.8%     
  PFC         22     FIN SVC    02-Jan-23   102.6       93.4        3452   ₹-31,581      -8.9%       -6.0%     
  HUDCO       32     FIN SVC    01-Dec-22   47.2        41.6        7702   ₹-43,084      -11.8%      -4.9%     

  AFTER: Invested ₹9,869,308 | Cash ₹312,194 | Total ₹10,181,502 | Positions 27/30 | Slot ₹339,477

========================================================================
  REBALANCE #99  —  01 Mar 2023
  NAV: ₹10,010,070  |  Slot: ₹333,669  |  Cash: ₹312,194
========================================================================

  [REGIME OFF] Nifty 500 14,664.8 < SMA200 14,849.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (27)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         20     DEFENCE    01-Apr-22   723.8       1,289.7     433    ₹245,052      +78.2%      +5.6%     
  ITC         1      FMCG       01-Apr-22   204.2       317.9       1537   ₹174,728      +55.7%      +1.3%     
  VBL         12     FMCG       01-Aug-22   183.1       266.4       1700   ₹141,570      +45.5%      +4.6%     
  KARURVYSYA  13     PVT BNK    01-Sep-22   57.0        81.3        5791   ₹140,268      +42.5%      -1.1%     
  CGPOWER     18     ENERGY     01-Aug-22   222.0       303.1       1402   ₹113,636      +36.5%      -1.9%     
  SOLARINDS   54     DEFENCE    01-Sep-22   3,315.3     3,893.0     99     ₹57,196       +17.4%      -0.1%     
  CUMMINSIND  6      INFRA      01-Dec-22   1,372.2     1,540.7     265    ₹44,674       +12.3%      +3.2%     
  CIEINDIA    2      AUTO       01-Feb-23   382.9       407.3       886    ₹21,641       +6.4%       +6.1%     
  JSL         7      METAL      01-Feb-23   257.7       266.6       1317   ₹11,785       +3.5%       +3.5%     
  PFC         21     FIN SVC    02-Jan-23   102.6       104.9       3452   ₹7,998        +2.3%       +4.0%     
  AEGISLOG    26     INFRA      01-Feb-23   349.3       355.0       971    ₹5,557        +1.6%       +3.2%     
  BRITANNIA   80     FMCG       01-Feb-23   4,187.2     4,195.7     81     ₹687          +0.2%       -2.1%     
  GODFRYPHLP  43     FMCG       01-Dec-22   577.8       574.5       629    ₹-2,039       -0.6%       +2.1%     
  RECLTD      28     FIN SVC    02-Jan-23   100.3       98.5        3532   ₹-6,385       -1.8%       +0.3%     
  KPIL        60     INFRA      02-Jan-23   545.2       527.5       649    ₹-11,495      -3.2%       +5.8%     
  RHIM        224    METAL      03-Oct-22   661.6       640.2       493    ₹-10,538      -3.2% ⚠     -5.4%     
  RATNAMANI   42     METAL      01-Feb-23   2,223.4     2,122.7     152    ₹-15,310      -4.5%       +0.2%     
  JYOTHYLAB   83     FMCG       02-Jan-23   194.7       181.1       1819   ₹-24,761      -7.0%       -4.0%     
  JKLAKSHMI   71     MFG        01-Feb-23   757.9       700.0       447    ₹-25,841      -7.6%       -2.6%     
  IIFL        103    FIN SVC    02-Jan-23   466.1       430.4       759    ₹-27,096      -7.7%       -3.5%     
  REDINGTON   154    IT         02-Jan-23   168.1       152.1       2107   ₹-33,673      -9.5%       -3.8%     
  MAHABANK    85     PSU BNK    02-Jan-23   27.0        23.4        13140  ₹-46,957      -13.3%      -1.6%     
  GESHIP      89     INFRA      02-Jan-23   598.1       507.6       592    ₹-53,576      -15.1%      -3.2%     
  HUDCO       145    FIN SVC    01-Dec-22   47.2        39.8        7702   ₹-57,103      -15.7%      -1.4%     
  SWANCORP    110    ENERGY     01-Feb-23   311.1       258.9       1091   ₹-57,018      -16.8%      -4.8%     
  IRFC        235    FIN SVC    01-Dec-22   32.4        25.9        11228  ₹-73,401      -20.2% ⚠    -5.6%     
  RVNL        82     PSE        01-Dec-22   72.2        56.3        5036   ₹-80,220      -22.1%      -11.7%    
  ⚠  WAZ < 0 (momentum below universe mean): RHIM, IRFC

  AFTER: Invested ₹9,697,876 | Cash ₹312,194 | Total ₹10,010,070 | Positions 27/30 | Slot ₹333,669

========================================================================
  REBALANCE #100  —  03 Apr 2023
  NAV: ₹10,075,023  |  Slot: ₹335,834  |  Cash: ₹312,194
========================================================================

  [REGIME OFF] Nifty 500 14,602.0 < SMA200 14,914.8 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (27)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         14     DEFENCE    01-Apr-22   723.8       1,312.6     433    ₹254,970      +81.4%      +1.9%     
  ITC         4      FMCG       01-Apr-22   204.2       318.1       1537   ₹174,986      +55.7%      -0.3%     
  VBL         10     FMCG       01-Aug-22   183.1       280.4       1700   ₹165,344      +53.1%      +5.0%     
  KARURVYSYA  36     PVT BNK    01-Sep-22   57.0        80.2        5791   ₹134,220      +40.6%      -0.2%     
  CGPOWER     28     ENERGY     01-Aug-22   222.0       294.6       1402   ₹101,762      +32.7%      +0.1%     
  CUMMINSIND  11     INFRA      01-Dec-22   1,372.2     1,543.4     265    ₹45,375       +12.5%      -1.1%     
  SOLARINDS   186    DEFENCE    01-Sep-22   3,315.3     3,724.5     99     ₹40,511       +12.3%      -1.7%     
  JSL         41     METAL      01-Feb-23   257.7       284.4       1317   ₹35,224       +10.4%      -0.0%     
  PFC         21     FIN SVC    02-Jan-23   102.6       108.8       3452   ₹21,568       +6.1%       +1.5%     
  AEGISLOG    35     INFRA      01-Feb-23   349.3       364.7       971    ₹14,929       +4.4%       +0.1%     
  JKLAKSHMI   57     MFG        01-Feb-23   757.9       784.0       447    ₹11,706       +3.5%       +9.4%     
  RVNL        15     PSE        01-Dec-22   72.2        72.5        5036   ₹1,454        +0.4%       +13.3%    
  RECLTD      43     FIN SVC    02-Jan-23   100.3       100.2       3532   ₹-188         -0.1%       +0.3%     
  IIFL        74     FIN SVC    02-Jan-23   466.1       464.2       759    ₹-1,418       -0.4%       +5.8%     
  BRITANNIA   49     FMCG       01-Feb-23   4,187.2     4,142.3     81     ₹-3,639       -1.1%       +0.5%     
  GODFRYPHLP  85     FMCG       01-Dec-22   577.8       562.1       629    ₹-9,847       -2.7%       -3.8%     
  GESHIP      45     INFRA      02-Jan-23   598.1       577.8       592    ₹-11,997      -3.4%       +5.0%     
  RHIM        335    METAL      03-Oct-22   661.6       619.2       493    ₹-20,906      -6.4% ⚠     +0.7%     
  JYOTHYLAB   112    FMCG       02-Jan-23   194.7       179.4       1819   ₹-27,705      -7.8%       -0.1%     
  KPIL        115    INFRA      02-Jan-23   545.2       499.9       649    ₹-29,442      -8.3%       -5.5%     
  CIEINDIA    24     AUTO       01-Feb-23   382.9       345.2       886    ₹-33,372      -9.8%       -0.7%     
  REDINGTON   154    IT         02-Jan-23   168.1       151.2       2107   ₹-35,465      -10.0%      +0.6%     
  RATNAMANI   122    METAL      01-Feb-23   2,223.4     1,951.9     152    ₹-41,270      -12.2%      -4.2%     
  HUDCO       149    FIN SVC    01-Dec-22   47.2        40.5        7702   ₹-51,839      -14.2%      +2.2%     
  MAHABANK    145    PSU BNK    02-Jan-23   27.0        22.7        13140  ₹-55,653      -15.7%      +2.2%     
  IRFC        181    FIN SVC    01-Dec-22   32.4        26.3        11228  ₹-68,082      -18.7%      +2.5%     
  SWANCORP    440    ENERGY     01-Feb-23   311.1       213.1       1091   ₹-106,896     -31.5% ⚠    -8.7%     
  ⚠  WAZ < 0 (momentum below universe mean): RHIM, SWANCORP

  AFTER: Invested ₹9,762,829 | Cash ₹312,194 | Total ₹10,075,023 | Positions 27/30 | Slot ₹335,834

========================================================================
  REBALANCE #101  —  02 May 2023
  NAV: ₹10,807,875  |  Slot: ₹360,263  |  Cash: ₹312,194
========================================================================

  EXITS (16)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  KARURVYSYA  70     PVT BNK    01-Sep-22   57.0        78.3        5791   ₹123,287      +37.3%    243d  
  SOLARINDS   183    DEFENCE    01-Sep-22   3,315.3     3,825.9     99     ₹50,546       +15.4%    243d  
  JSL         111    METAL      01-Feb-23   257.7       281.2       1317   ₹31,042       +9.1%     90d   
  JKLAKSHMI   65     MFG        01-Feb-23   757.9       766.3       447    ₹3,785        +1.1%     90d   
  GESHIP      71     INFRA      02-Jan-23   598.1       598.0       592    ₹-27          -0.0%     120d  
  HUDCO       73     FIN SVC    01-Dec-22   47.2        46.8        7702   ₹-3,495       -1.0%     152d  
  IIFL        170    FIN SVC    02-Jan-23   466.1       460.6       759    ₹-4,188       -1.2%     120d  
  MAHABANK    101    PSU BNK    02-Jan-23   27.0        26.5        13140  ₹-6,377       -1.8%     120d  
  IRFC        76     FIN SVC    01-Dec-22   32.4        31.8        11228  ₹-6,915       -1.9%     152d  
  RHIM        379    METAL      03-Oct-22   661.6       635.3       493    ₹-12,972      -4.0%     211d  
  RATNAMANI   91     METAL      01-Feb-23   2,223.4     2,131.1     152    ₹-14,029      -4.2%     90d   
  JYOTHYLAB   209    FMCG       02-Jan-23   194.7       183.0       1819   ₹-21,211      -6.0%     120d  
  KPIL        74     INFRA      02-Jan-23   545.2       512.2       649    ₹-21,455      -6.1%     120d  
  GODFRYPHLP  200    FMCG       01-Dec-22   577.8       539.8       629    ₹-23,852      -6.6%     152d  
  REDINGTON   233    IT         02-Jan-23   168.1       153.7       2107   ₹-30,183      -8.5%     120d  
  SWANCORP    443    ENERGY     01-Feb-23   311.1       235.2       1091   ₹-82,802      -24.4%    90d   

  ENTRIES (16)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  APARINDS    1      ENERGY     4.909    0.87   +324.5%   +75.7%    2,777.7     129    ₹358,327      +5.6%     
  FINCABLES   3      ENERGY     4.411    1.10   +130.0%   +67.8%    889.3       405    ₹360,176      +6.4%     
  ANURAS      4      FIN SVC    4.324    0.70   +40.7%    +90.7%    1,179.0     305    ₹359,585      +15.9%    
  NCC         6      INFRA      3.204    0.90   +85.5%    +33.3%    118.8       3031   ₹360,213      +9.2%     
  TVSMOTOR    7      AUTO       2.833    0.86   +79.2%    +18.3%    1,144.6     314    ₹359,413      +3.5%     
  CYIENT      8      IT         2.780    0.60   +34.3%    +34.6%    1,102.3     326    ₹359,364      +6.0%     
  ZYDUSLIFE   9      HEALTH     2.687    0.59   +52.6%    +18.3%    505.1       713    ₹360,146      +1.9%     
  INGERRAND   10     MFG        2.680    0.64   +60.1%    +33.5%    2,526.7     142    ₹358,794      -0.5%     
  SONATSOFTW  11     IT         2.665    0.80   +55.2%    +37.4%    411.2       876    ₹360,231      +3.4%     
  SIEMENS     14     ENERGY     2.611    0.81   +55.7%    +15.9%    2,039.2     176    ₹358,907      +4.3%     
  KSB         15     MFG        2.609    0.60   +70.3%    +22.0%    435.6       827    ₹360,245      +1.5%     
  INDIANB     17     PSU BNK    2.579    1.18   +115.1%   +13.4%    298.0       1208   ₹360,027      +7.7%     
  KPITTECH    19     IT         2.530    0.89   +79.3%    +33.4%    921.2       391    ₹360,188      +7.1%     
  BAJAJ-AUTO  20     AUTO       2.486    0.65   +26.7%    +22.0%    4,163.7     86     ₹358,077      +6.2%     
  BOSCHLTD    21     AUTO       2.481    0.99   +43.2%    +17.4%    19,016.2    18     ₹342,291      +4.7%     
  KEI         22     MFG        2.431    0.86   +60.3%    +24.6%    1,877.4     191    ₹358,591      +5.5%     

  HOLDS (11)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         19     DEFENCE    01-Apr-22   723.8       1,422.0     433    ₹302,307      +96.5%      +4.6%     
  ITC         5      FMCG       01-Apr-22   204.2       356.3       1537   ₹233,760      +74.5%      +5.2%     
  RVNL        2      PSE        01-Dec-22   72.2        116.6       5036   ₹223,459      +61.4%      +36.1%    
  VBL         29     FMCG       01-Aug-22   183.1       280.7       1700   ₹165,895      +53.3%      -0.1%     
  CGPOWER     64     ENERGY     01-Aug-22   222.0       304.7       1402   ₹115,892      +37.2%      +1.6%     
  PFC         14     FIN SVC    02-Jan-23   102.6       121.9       3452   ₹66,515       +18.8%      +6.9%     
  RECLTD      17     FIN SVC    02-Jan-23   100.3       116.8       3532   ₹58,307       +16.5%      +9.9%     
  CUMMINSIND  13     INFRA      01-Dec-22   1,372.2     1,538.5     265    ₹44,088       +12.1%      +1.9%     
  AEGISLOG    54     INFRA      01-Feb-23   349.3       382.3       971    ₹32,024       +9.4%       +1.3%     
  BRITANNIA   51     FMCG       01-Feb-23   4,187.2     4,395.5     81     ₹16,873       +5.0%       +3.9%     
  CIEINDIA    34     AUTO       01-Feb-23   382.9       379.4       886    ₹-3,092       -0.9%       +5.5%     

  AFTER: Invested ₹10,693,760 | Cash ₹107,306 | Total ₹10,801,066 | Positions 27/30 | Slot ₹360,263

========================================================================
  REBALANCE #102  —  01 Jun 2023
  NAV: ₹11,246,295  |  Slot: ₹374,877  |  Cash: ₹107,306
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BRITANNIA   131    FMCG       01-Feb-23   4,187.2     4,516.5     81     ₹26,675       +7.9%     120d  
  BAJAJ-AUTO  51     AUTO       02-May-23   4,163.7     4,298.5     86     ₹11,591       +3.2%     30d   
  SIEMENS     96     ENERGY     02-May-23   2,039.2     2,059.2     176    ₹3,507        +1.0%     30d   
  ZYDUSLIFE   108    HEALTH     02-May-23   505.1       501.6       713    ₹-2,473       -0.7%     30d   
  AEGISLOG    258    INFRA      01-Feb-23   349.3       332.1       971    ₹-16,718      -4.9%     120d  
  KSB         133    MFG        02-May-23   435.6       414.1       827    ₹-17,759      -4.9%     30d   
  BOSCHLTD    157    AUTO       02-May-23   19,016.2    17,867.1    18     ₹-20,683      -6.0%     30d   
  FINCABLES   53     ENERGY     02-May-23   889.3       772.7       405    ₹-47,249      -13.1%    30d   
  INDIANB     171    PSU BNK    02-May-23   298.0       248.9       1208   ₹-59,339      -16.5%    30d   

  ENTRIES (8)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ENGINERSIN  7      INFRA      3.036    0.74   +94.5%    +52.0%    104.1       3601   ₹374,837      +7.8%     
  INDHOTEL    14     CONSUMP    2.734    0.83   +79.9%    +27.8%    390.6       959    ₹374,570      +6.3%     
  CHOLAFIN    15     FIN SVC    2.701    1.11   +63.8%    +36.8%    1,039.4     360    ₹374,170      +3.0%     
  GLENMARK    16     HEALTH     2.639    0.78   +55.9%    +39.0%    605.5       619    ₹374,784      +3.6%     
  CHOLAHLDNG  17     FIN SVC    2.613    0.50   +29.4%    +39.1%    813.1       461    ₹374,824      +3.7%     
  CERA        18     CON DUR    2.611    0.52   +91.9%    +24.8%    7,452.5     50     ₹372,625      +4.6%     
  EQUITASBNK  19     PVT BNK    2.575    0.79   +104.7%   +30.6%    85.7        4373   ₹374,834      +7.7%     
  DATAPATTNS  20     DEFENCE    2.529    1.10   +130.9%   +36.5%    1,659.9     225    ₹373,484      +2.5%     

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         38     DEFENCE    01-Apr-22   723.8       1,490.9     433    ₹332,157      +106.0%     +1.9%     
  ITC         6      FMCG       01-Apr-22   204.2       377.4       1537   ₹266,180      +84.8%      +3.5%     
  VBL         12     FMCG       01-Aug-22   183.1       335.0       1700   ₹258,260      +83.0%      +5.7%     
  CGPOWER     4      ENERGY     01-Aug-22   222.0       385.8       1402   ₹229,625      +73.8%      +9.2%     
  RVNL        1      PSE        01-Dec-22   72.2        116.8       5036   ₹224,204      +61.6%      +2.4%     
  PFC         14     FIN SVC    02-Jan-23   102.6       128.6       3452   ₹89,777       +25.4%      +7.0%     
  CUMMINSIND  49     INFRA      01-Dec-22   1,372.2     1,683.2     265    ₹82,433       +22.7%      +4.5%     
  RECLTD      11     FIN SVC    02-Jan-23   100.3       120.3       3532   ₹70,853       +20.0%      +5.1%     
  KPITTECH    13     IT         02-May-23   921.2       1,083.7     391    ₹63,542       +17.6%      +13.3%    
  CYIENT      5      IT         02-May-23   1,102.3     1,292.8     326    ₹62,098       +17.3%      +7.6%     
  CIEINDIA    50     AUTO       01-Feb-23   382.9       441.1       886    ₹51,625       +15.2%      +3.9%     
  SONATSOFTW  9      IT         02-May-23   411.2       473.4       876    ₹54,441       +15.1%      +6.6%     
  TVSMOTOR    29     AUTO       02-May-23   1,144.6     1,257.5     314    ₹35,431       +9.9%       +1.9%     
  KEI         27     MFG        02-May-23   1,877.4     2,051.4     191    ₹33,222       +9.3%       +4.2%     
  INGERRAND   34     MFG        02-May-23   2,526.7     2,557.7     142    ₹4,402        +1.2%       +2.7%     
  NCC         10     INFRA      02-May-23   118.8       119.2       3031   ₹1,175        +0.3%       +6.8%     
  APARINDS    3      ENERGY     02-May-23   2,777.7     2,657.1     129    ₹-15,562      -4.3%       -1.1%     
  ANURAS      2      FIN SVC    02-May-23   1,179.0     1,112.9     305    ₹-20,156      -5.6%       -3.3%     

  AFTER: Invested ₹11,077,364 | Cash ₹165,376 | Total ₹11,242,740 | Positions 26/30 | Slot ₹374,877

========================================================================
  REBALANCE #103  —  03 Jul 2023
  NAV: ₹12,049,779  |  Slot: ₹401,659  |  Cash: ₹165,376
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  VBL         111    FMCG       01-Aug-22   183.1       324.7       1700   ₹240,623      +77.3%    336d  
  PFC         5      FIN SVC    02-Jan-23   102.6       157.8       3452   ₹190,540      +53.8%    182d  
  KPITTECH    108    IT         02-May-23   921.2       1,060.1     391    ₹54,319       +15.1%    62d   
  INGERRAND   116    MFG        02-May-23   2,526.7     2,624.8     142    ₹13,921       +3.9%     62d   
  CERA        70     CON DUR    01-Jun-23   7,452.5     7,517.3     50     ₹3,238        +0.9%     32d   
  INDHOTEL    74     CONSUMP    01-Jun-23   390.6       386.4       959    ₹-4,044       -1.1%     32d   
  ANURAS      164    FIN SVC    02-May-23   1,179.0     1,002.4     305    ₹-53,846      -15.0%    62d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  JBMA        1      AUTO       4.415    0.80   +229.4%   +121.7%   694.1       578    ₹401,170      +32.1%    
  FACT        2      ENERGY     4.296    1.14   +387.0%   +138.5%   468.5       857    ₹401,465      +18.1%    
  JSL         8      METAL      2.947    0.77   +245.5%   +21.9%    335.6       1196   ₹401,402      +5.4%     
  EXIDEIND    12     AUTO       2.698    0.76   +74.9%    +38.5%    234.6       1712   ₹401,557      +7.7%     
  INDIGO      14     CONSUMP    2.694    0.78   +61.8%    +44.5%    2,633.7     152    ₹400,330      +6.7%     
  HEG         16     METAL      2.616    0.94   +70.9%    +75.9%    314.6       1276   ₹401,481      +7.6%     
  KARURVYSYA  17     PVT BNK    2.481    1.05   +191.3%   +30.4%    102.8       3909   ₹401,657      +7.6%     

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         15     DEFENCE    01-Apr-22   723.8       1,824.5     433    ₹476,598      +152.1%     +3.8%     
  ITC         11     FMCG       01-Apr-22   204.2       397.6       1537   ₹297,250      +94.7%      +3.9%     
  CGPOWER     31     ENERGY     01-Aug-22   222.0       376.2       1402   ₹216,191      +69.5%      +1.7%     
  RVNL        4      PSE        01-Dec-22   72.2        120.1       5036   ₹241,081      +66.3%      -0.1%     
  RECLTD      9      FIN SVC    02-Jan-23   100.3       142.3       3532   ₹148,393      +41.9%      +6.9%     
  CUMMINSIND  52     INFRA      01-Dec-22   1,372.2     1,817.1     265    ₹117,912      +32.4%      +2.6%     
  CYIENT      14     IT         02-May-23   1,102.3     1,442.3     326    ₹110,824      +30.8%      +4.8%     
  CIEINDIA    24     AUTO       01-Feb-23   382.9       496.2       886    ₹100,381      +29.6%      +2.6%     
  APARINDS    16     ENERGY     02-May-23   2,777.7     3,464.2     129    ₹88,559       +24.7%      +12.2%    
  KEI         35     MFG        02-May-23   1,877.4     2,286.0     191    ₹78,039       +21.8%      +2.7%     
  SONATSOFTW  61     IT         02-May-23   411.2       478.8       876    ₹59,167       +16.4%      +0.2%     
  TVSMOTOR    65     AUTO       02-May-23   1,144.6     1,303.2     314    ₹49,786       +13.9%      +0.2%     
  CHOLAHLDNG  8      FIN SVC    01-Jun-23   813.1       920.4       461    ₹49,476       +13.2%      +4.4%     
  DATAPATTNS  20     DEFENCE    01-Jun-23   1,659.9     1,876.9     225    ₹48,817       +13.1%      +4.2%     
  CHOLAFIN    10     FIN SVC    01-Jun-23   1,039.4     1,164.1     360    ₹44,894       +12.0%      +6.3%     
  GLENMARK    25     HEALTH     01-Jun-23   605.5       655.0       619    ₹30,648       +8.2%       +2.7%     
  ENGINERSIN  18     INFRA      01-Jun-23   104.1       108.6       3601   ₹16,371       +4.4%       +1.7%     
  EQUITASBNK  26     PVT BNK    01-Jun-23   85.7        88.9        4373   ₹14,104       +3.8%       +5.2%     
  NCC         69     INFRA      02-May-23   118.8       118.0       3031   ₹-2,497       -0.7%       +0.7%     

  AFTER: Invested ₹11,757,522 | Cash ₹288,921 | Total ₹12,046,443 | Positions 26/30 | Slot ₹401,659

========================================================================
  REBALANCE #104  —  01 Aug 2023
  NAV: ₹12,803,787  |  Slot: ₹426,793  |  Cash: ₹288,921
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ITC         53     FMCG       01-Apr-22   204.2       399.0       1537   ₹299,295      +95.3%    487d  
  CUMMINSIND  50     INFRA      01-Dec-22   1,372.2     1,865.6     265    ₹130,758      +36.0%    243d  
  CIEINDIA    135    AUTO       01-Feb-23   382.9       478.5       886    ₹84,754       +25.0%    181d  
  SONATSOFTW  80     IT         02-May-23   411.2       506.3       876    ₹83,251       +23.1%    91d   
  KEI         69     MFG        02-May-23   1,877.4     2,263.0     191    ₹73,651       +20.5%    91d   
  DATAPATTNS  49     DEFENCE    01-Jun-23   1,659.9     1,996.1     225    ₹75,633       +20.3%    61d   
  TVSMOTOR    100    AUTO       02-May-23   1,144.6     1,353.9     314    ₹65,724       +18.3%    91d   
  CHOLAFIN    48     FIN SVC    01-Jun-23   1,039.4     1,126.2     360    ₹31,270       +8.4%     61d   
  EQUITASBNK  58     PVT BNK    01-Jun-23   85.7        90.6        4373   ₹21,521       +5.7%     61d   
  INDIGO      147    CONSUMP    03-Jul-23   2,633.7     2,566.4     152    ₹-10,235      -2.6%     29d   

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  PFC         1      FIN SVC    3.949    1.20   +150.3%   +55.3%    185.9       2296   ₹426,728      +10.0%    
  POLYCAB     5      MFG        3.128    0.82   +109.5%   +41.5%    4,561.7     93     ₹424,237      +7.3%     
  ZENSARTECH  7      IT         3.055    0.97   +96.1%    +71.7%    475.8       897    ₹426,789      +8.6%     
  JYOTHYLAB   9      FMCG       2.988    0.49   +85.0%    +59.7%    294.5       1449   ₹426,764      +14.9%    
  HBLENGINE   12     MFG        2.709    0.98   +119.7%   +76.8%    199.6       2137   ₹426,628      +17.9%    
  SJVN        15     ENERGY     2.563    0.84   +111.0%   +49.1%    54.0        7901   ₹426,747      +10.8%    
  NTPC        16     ENERGY     2.524    0.70   +56.9%    +27.7%    207.6       2055   ₹426,665      +12.7%    
  ZYDUSLIFE   17     HEALTH     2.522    0.42   +85.5%    +22.9%    625.2       682    ₹426,360      +4.7%     
  TATACOMM    18     CONSUMP    2.565    0.90   +69.3%    +44.3%    1,705.7     250    ₹426,418      +6.9%     
  BEML        20     DEFENCE    2.428    0.77   +83.1%    +55.3%    970.7       439    ₹426,157      +14.0%    
  BSE         21     FIN SVC    2.422    0.92   +26.5%    +62.3%    274.3       1556   ₹426,759      +13.4%    

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  HAL         28     DEFENCE    01-Apr-22   723.8       1,870.9     433    ₹496,683      +158.5%     +0.8%     
  CGPOWER     25     ENERGY     01-Aug-22   222.0       405.1       1402   ₹256,701      +82.5%      +2.1%     
  RECLTD      3      FIN SVC    02-Jan-23   100.3       175.8       3532   ₹266,760      +75.3%      +14.9%    
  RVNL        38     PSE        01-Dec-22   72.2        123.3       5036   ₹257,214      +70.7%      -0.1%     
  ENGINERSIN  8      INFRA      01-Jun-23   104.1       147.3       3601   ₹155,528      +41.5%      +13.1%    
  APARINDS    32     ENERGY     02-May-23   2,777.7     3,627.4     129    ₹109,611      +30.6%      +2.7%     
  GLENMARK    10     HEALTH     01-Jun-23   605.5       782.9       619    ₹109,804      +29.3%      +6.1%     
  NCC         18     INFRA      02-May-23   118.8       150.3       3031   ₹95,342       +26.5%      +11.5%    
  CYIENT      43     IT         02-May-23   1,102.3     1,393.5     326    ₹94,912       +26.4%      +0.2%     
  CHOLAHLDNG  16     FIN SVC    01-Jun-23   813.1       963.3       461    ₹69,267       +18.5%      +3.9%     
  JSL         6      METAL      03-Jul-23   335.6       396.2       1196   ₹72,408       +18.0%      +8.5%     
  HEG         37     METAL      03-Jul-23   314.6       348.2       1276   ₹42,775       +10.7%      +9.9%     
  EXIDEIND    14     AUTO       03-Jul-23   234.6       259.4       1712   ₹42,576       +10.6%      +6.6%     
  FACT        13     ENERGY     03-Jul-23   468.5       493.8       857    ₹21,739       +5.4%       +5.2%     
  JBMA        5      AUTO       03-Jul-23   694.1       698.4       578    ₹2,521        +0.6%       +2.9%     
  KARURVYSYA  44     PVT BNK    03-Jul-23   102.8       101.2       3909   ₹-6,124       -1.5%       +0.2%     

  AFTER: Invested ₹12,731,685 | Cash ₹66,533 | Total ₹12,798,218 | Positions 27/30 | Slot ₹426,793

========================================================================
  REBALANCE #105  —  01 Sep 2023
  NAV: ₹14,057,842  |  Slot: ₹468,595  |  Cash: ₹66,533
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  HAL         58     DEFENCE    01-Apr-22   723.8       1,914.7     433    ₹515,662      +164.5%   518d  
  CGPOWER     73     ENERGY     01-Aug-22   222.0       421.0       1402   ₹278,905      +89.6%    396d  
  CYIENT      55     IT         02-May-23   1,102.3     1,629.9     326    ₹171,984      +47.9%    122d  
  HBLENGINE   2      MFG        01-Aug-23   199.6       272.6       2137   ₹155,871      +36.5%    31d   
  CHOLAHLDNG  113    FIN SVC    01-Jun-23   813.1       962.8       461    ₹69,047       +18.4%    92d   
  HEG         135    METAL      03-Jul-23   314.6       344.2       1276   ₹37,778       +9.4%     60d   
  PFC         17     FIN SVC    01-Aug-23   185.9       184.8       2296   ₹-2,390       -0.6%     31d   
  ZYDUSLIFE   47     HEALTH     01-Aug-23   625.2       605.5       682    ₹-13,418      -3.1%     31d   
  KARURVYSYA  143    PVT BNK    03-Jul-23   102.8       99.4        3909   ₹-13,079      -3.3%     60d   

  ENTRIES (10)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SUPREMEIND  7      MFG        3.580    0.48   +130.5%   +56.9%    4,295.5     109    ₹468,211      +4.3%     
  IRFC        9      FIN SVC    3.375    0.87   +178.0%   +74.2%    52.8        8871   ₹468,561      +16.5%    
  SHYAMMETL   11     METAL      3.240    0.73   +62.2%    +64.1%    476.0       984    ₹468,347      +5.4%     
  LINDEINDIA  12     MFG        3.336    0.78   +97.7%    +63.8%    6,527.5     71     ₹463,450      +16.8%    
  BHEL        13     ENERGY     3.090    1.11   +137.8%   +64.3%    135.8       3451   ₹468,475      +24.2%    
  COCHINSHIP  14     DEFENCE    3.018    1.10   +157.9%   +81.2%    442.1       1059   ₹468,160      +12.6%    
  GMRINFRA    18     INFRA      2.625    1.01   +78.8%    +52.4%    63.0        7443   ₹468,537      +9.8%     
  APLAPOLLO   20     METAL      2.607    0.43   +78.3%    +52.8%    1,716.9     272    ₹466,994      +8.4%     
  NATCOPHARM  21     HEALTH     2.595    0.48   +46.3%    +44.9%    890.6       526    ₹468,464      +4.3%     
  ESCORTS     22     MFG        2.565    0.62   +71.1%    +44.0%    3,077.7     152    ₹467,808      +8.7%     

  HOLDS (18)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RECLTD      3      FIN SVC    02-Jan-23   100.3       212.8       3532   ₹397,273      +112.2%     +4.9%     
  RVNL        20     PSE        01-Dec-22   72.2        136.3       5036   ₹322,738      +88.7%      +8.7%     
  APARINDS    7      ENERGY     02-May-23   2,777.7     4,898.1     129    ₹273,530      +76.3%      +7.4%     
  ENGINERSIN  21     INFRA      01-Jun-23   104.1       150.3       3601   ₹166,555      +44.4%      +3.2%     
  NCC         25     INFRA      02-May-23   118.8       168.1       3031   ₹149,289      +41.4%      +11.2%    
  JSL         4      METAL      03-Jul-23   335.6       457.5       1196   ₹145,820      +36.3%      +10.0%    
  BSE         5      FIN SVC    01-Aug-23   274.3       372.2       1556   ₹152,310      +35.7%      +22.9%    
  BEML        28     DEFENCE    01-Aug-23   970.7       1,226.7     439    ₹112,383      +26.4%      +15.4%    
  GLENMARK    40     HEALTH     01-Jun-23   605.5       751.9       619    ₹90,653       +24.2%      -1.2%     
  JYOTHYLAB   9      FMCG       01-Aug-23   294.5       352.9       1449   ₹84,556       +19.8%      +11.5%    
  POLYCAB     11     MFG        01-Aug-23   4,561.7     5,131.2     93     ₹52,963       +12.5%      +6.5%     
  EXIDEIND    41     AUTO       03-Jul-23   234.6       262.0       1712   ₹47,049       +11.7%      +1.4%     
  SJVN        14     ENERGY     01-Aug-23   54.0        60.0        7901   ₹47,208       +11.1%      +8.3%     
  ZENSARTECH  33     IT         01-Aug-23   475.8       517.4       897    ₹37,290       +8.7%       +5.8%     
  JBMA        6      AUTO       03-Jul-23   694.1       748.5       578    ₹31,435       +7.8%       +2.9%     
  TATACOMM    46     CONSUMP    01-Aug-23   1,705.7     1,792.4     250    ₹21,687       +5.1%       +5.4%     
  NTPC        22     ENERGY     01-Aug-23   207.6       215.7       2055   ₹16,635       +3.9%       +6.2%     
  FACT        31     ENERGY     03-Jul-23   468.5       465.7       857    ₹-2,392       -0.6%       +1.2%     

  AFTER: Invested ₹14,026,238 | Cash ₹26,051 | Total ₹14,052,288 | Positions 28/30 | Slot ₹468,595

========================================================================
  REBALANCE #106  —  03 Oct 2023
  NAV: ₹14,706,867  |  Slot: ₹490,229  |  Cash: ₹26,051
========================================================================

  EXITS (12)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  IRFC        2      FIN SVC    01-Sep-23   52.8        73.2        8871   ₹180,832      +38.6%    32d   
  COCHINSHIP  20     DEFENCE    01-Sep-23   442.1       521.6       1059   ₹84,262       +18.0%    32d   
  BEML        72     DEFENCE    01-Aug-23   970.7       1,138.5     439    ₹73,648       +17.3%    63d   
  FACT        39     ENERGY     03-Jul-23   468.5       531.5       857    ₹53,997       +13.4%    92d   
  EXIDEIND    93     AUTO       03-Jul-23   234.6       256.4       1712   ₹37,343       +9.3%     92d   
  TATACOMM    92     CONSUMP    01-Aug-23   1,705.7     1,840.2     250    ₹33,625       +7.9%     63d   
  BHEL        29     ENERGY     01-Sep-23   135.8       130.8       3451   ₹-17,204      -3.7%     32d   
  NATCOPHARM  76     HEALTH     01-Sep-23   890.6       856.8       526    ₹-17,802      -3.8%     32d   
  JBMA        71     AUTO       03-Jul-23   694.1       659.6       578    ₹-19,935      -5.0%     92d   
  APLAPOLLO   107    METAL      01-Sep-23   1,716.9     1,628.4     272    ₹-24,072      -5.2%     32d   
  GMRINFRA    57     INFRA      01-Sep-23   63.0        59.5        7443   ₹-26,050      -5.6%     32d   
  SHYAMMETL   99     METAL      01-Sep-23   476.0       439.7       984    ₹-35,637      -7.6%     32d   

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HBLENGINE   7      MFG        3.379    1.19   +217.9%   +83.7%    279.9       1751   ₹490,048      +8.0%     
  MAPMYINDIA  8      IT         3.340    0.39   +62.8%    +78.5%    2,109.7     232    ₹489,442      +17.0%    
  JINDALSAW   10     METAL      3.219    0.98   +338.0%   +35.3%    177.9       2755   ₹490,073      +3.9%     
  CENTRALBK   11     PSU BNK    3.045    1.18   +169.9%   +75.6%    50.6        9688   ₹490,223      +13.9%    
  UJJIVANSFB  12     PVT BNK    3.207    0.97   +178.0%   +53.9%    57.0        8593   ₹490,222      +14.8%    
  HUDCO       13     FIN SVC    2.888    1.17   +174.2%   +62.1%    85.0        5769   ₹490,182      +13.4%    
  SAFARI      14     CON DUR    2.875    0.41   +168.5%   +37.4%    2,063.9     237    ₹489,142      +10.7%    
  LT          15     INFRA      2.849    0.83   +67.9%    +26.7%    2,991.9     163    ₹487,677      +5.9%     
  ITI         17     IT         2.777    0.72   +99.6%    +85.4%    201.0       2438   ₹490,038      +13.3%    
  INDIANB     18     PSU BNK    2.706    1.10   +144.4%   +48.2%    412.0       1189   ₹489,873      +7.3%     
  LUPIN       22     HEALTH     2.535    0.58   +80.3%    +32.2%    1,161.8     421    ₹489,138      +4.0%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  RECLTD      1      FIN SVC    02-Jan-23   100.3       260.0       3532   ₹564,054      +159.3%     +11.0%    
  RVNL        10     PSE        01-Dec-22   72.2        170.4       5036   ₹494,373      +135.9%     +6.7%     
  APARINDS    8      ENERGY     02-May-23   2,777.7     5,430.4     129    ₹342,192      +95.5%      +4.6%     
  BSE         4      FIN SVC    01-Aug-23   274.3       428.9       1556   ₹240,598      +56.4%      +6.5%     
  JSL         5      METAL      03-Jul-23   335.6       490.4       1196   ₹185,165      +46.1%      +5.5%     
  GLENMARK    31     HEALTH     01-Jun-23   605.5       839.3       619    ₹144,765      +38.6%      +4.3%     
  ENGINERSIN  52     INFRA      01-Jun-23   104.1       138.6       3601   ₹124,342      +33.2%      -2.1%     
  NCC         42     INFRA      02-May-23   118.8       157.9       3031   ₹118,442      +32.9%      +3.8%     
  SJVN        24     ENERGY     01-Aug-23   54.0        69.1        7901   ₹119,163      +27.9%      +3.4%     
  JYOTHYLAB   13     FMCG       01-Aug-23   294.5       355.9       1449   ₹88,892       +20.8%      +3.6%     
  POLYCAB     9      MFG        01-Aug-23   4,561.7     5,293.7     93     ₹68,081       +16.0%      +3.5%     
  NTPC        27     ENERGY     01-Aug-23   207.6       225.5       2055   ₹36,811       +8.6%       +2.0%     
  ZENSARTECH  28     IT         01-Aug-23   475.8       513.2       897    ₹33,552       +7.9%       +1.1%     
  ESCORTS     40     MFG        01-Sep-23   3,077.7     3,068.6     152    ₹-1,381       -0.3%       -0.2%     
  LINDEINDIA  45     MFG        01-Sep-23   6,527.5     5,964.1     71     ₹-40,002      -8.6%       +0.1%     
  SUPREMEIND  51     MFG        01-Sep-23   4,295.5     3,924.0     109    ₹-40,492      -8.6%       -4.4%     

  AFTER: Invested ₹14,409,563 | Cash ₹290,909 | Total ₹14,700,472 | Positions 27/30 | Slot ₹490,229

========================================================================
  REBALANCE #107  —  01 Nov 2023
  NAV: ₹14,442,254  |  Slot: ₹481,408  |  Cash: ₹290,909
========================================================================

  EXITS (11)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  RECLTD      2      FIN SVC    02-Jan-23   100.3       252.0       3532   ₹535,890      +151.3%   303d  
  RVNL        18     PSE        01-Dec-22   72.2        151.1       5036   ₹397,365      +109.2%   335d  
  SJVN        47     ENERGY     01-Aug-23   54.0        67.9        7901   ₹109,725      +25.7%    92d   
  GLENMARK    115    HEALTH     01-Jun-23   605.5       743.2       619    ₹85,263       +22.8%    153d  
  NCC         162    INFRA      02-May-23   118.8       138.8       3031   ₹60,473       +16.8%    183d  
  ENGINERSIN  218    INFRA      01-Jun-23   104.1       117.8       3601   ₹49,182       +13.1%    153d  
  POLYCAB     82     MFG        01-Aug-23   4,561.7     4,828.1     93     ₹24,780       +5.8%     92d   
  HBLENGINE   19     MFG        03-Oct-23   279.9       288.8       1751   ₹15,687       +3.2%     29d   
  UJJIVANSFB  83     PVT BNK    03-Oct-23   57.0        50.5        8593   ₹-56,500      -11.5%    29d   
  CENTRALBK   30     PSU BNK    03-Oct-23   50.6        41.6        9688   ₹-87,306      -17.8%    29d   
  HUDCO       53     FIN SVC    03-Oct-23   85.0        69.8        5769   ₹-87,620      -17.9%    29d   

  ENTRIES (12)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ANGELONE    7      FIN SVC    3.109    0.97   +68.4%    +71.6%    253.2       1901   ₹481,344      +16.1%    
  COALINDIA   8      ENERGY     3.098    0.81   +41.8%    +36.1%    254.9       1888   ₹481,161      +0.6%     
  SOLARINDS   9      DEFENCE    3.064    0.21   +42.4%    +45.3%    5,514.0     87     ₹479,715      +6.3%     
  MCX         11     FIN SVC    2.761    0.71   +61.4%    +45.3%    475.1       1013   ₹481,304      +9.7%     
  SWANCORP    12     ENERGY     2.741    0.94   +87.3%    +77.2%    388.5       1239   ₹481,295      +17.0%    
  NMDC        13     METAL      2.666    1.07   +61.7%    +34.1%    45.9        10486  ₹481,368      +0.3%     
  PERSISTENT  15     IT         2.603    1.20   +64.0%    +30.9%    3,057.8     157    ₹480,076      +5.3%     
  TRENT       16     CONSUMP    2.471    0.95   +53.4%    +25.1%    2,192.1     219    ₹480,078      +5.4%     
  SUNDARMFIN  17     FIN SVC    2.426    0.41   +37.8%    +20.2%    3,127.1     153    ₹478,449      +0.5%     
  SONATSOFTW  18     IT         2.365    0.71   +129.4%   +8.3%     553.3       870    ₹481,357      +3.4%     
  COHANCE     19     MNC        2.360    0.08   +39.4%    +15.8%    573.2       839    ₹480,873      +0.7%     
  CYIENT      20     IT         2.351    0.73   +120.8%   +12.3%    1,570.7     306    ₹480,634      -0.7%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BSE         1      FIN SVC    01-Aug-23   274.3       593.1       1556   ₹496,028      +116.2%     +11.1%    
  APARINDS    16     ENERGY     02-May-23   2,777.7     4,993.7     129    ₹285,856      +79.8%      -3.0%     
  ITI         4      IT         03-Oct-23   201.0       265.9       2438   ₹158,104      +32.3%      +4.7%     
  JSL         11     METAL      03-Jul-23   335.6       444.1       1196   ₹129,778      +32.3%      -2.0%     
  JINDALSAW   3      METAL      03-Oct-23   177.9       211.9       2755   ₹93,785       +19.1%      +13.0%    
  JYOTHYLAB   42     FMCG       01-Aug-23   294.5       349.0       1449   ₹78,960       +18.5%      -0.1%     
  NTPC        69     ENERGY     01-Aug-23   207.6       217.5       2055   ₹20,286       +4.8%       -1.7%     
  SAFARI      8      CON DUR    03-Oct-23   2,063.9     2,108.5     237    ₹10,569       +2.2%       +2.4%     
  SUPREMEIND  21     MFG        01-Sep-23   4,295.5     4,374.0     109    ₹8,553        +1.8%       +3.2%     
  ZENSARTECH  60     IT         01-Aug-23   475.8       478.6       897    ₹2,477        +0.6%       -2.7%     
  MAPMYINDIA  24     IT         03-Oct-23   2,109.7     2,106.2     232    ₹-798         -0.2%       +5.1%     
  INDIANB     45     PSU BNK    03-Oct-23   412.0       402.4       1189   ₹-11,434      -2.3%       +3.3%     
  ESCORTS     46     MFG        01-Sep-23   3,077.7     3,003.4     152    ₹-11,285      -2.4%       -4.2%     
  LUPIN       34     HEALTH     03-Oct-23   1,161.8     1,120.8     421    ₹-17,267      -3.5%       -1.7%     
  LT          39     INFRA      03-Oct-23   2,991.9     2,818.6     163    ₹-28,238      -5.8%       -2.4%     
  LINDEINDIA  38     MFG        01-Sep-23   6,527.5     5,936.2     71     ₹-41,976      -9.1%       -2.4%     

  AFTER: Invested ₹14,232,611 | Cash ₹202,794 | Total ₹14,435,405 | Positions 28/30 | Slot ₹481,408

========================================================================
  REBALANCE #108  —  01 Dec 2023
  NAV: ₹16,324,624  |  Slot: ₹544,154  |  Cash: ₹202,794
========================================================================

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SOLARINDS   76     DEFENCE    01-Nov-23   5,514.0     6,180.5     87     ₹57,986       +12.1%    30d   
  ZENSARTECH  72     IT         01-Aug-23   475.8       518.5       897    ₹38,289       +9.0%     122d  
  SWANCORP    115    ENERGY     01-Nov-23   388.5       415.1       1239   ₹33,007       +6.9%     30d   
  PERSISTENT  60     IT         01-Nov-23   3,057.8     3,167.3     157    ₹17,183       +3.6%     30d   
  ESCORTS     194    MFG        01-Sep-23   3,077.7     3,147.7     152    ₹10,643       +2.3%     91d   
  SUPREMEIND  145    MFG        01-Sep-23   4,295.5     4,367.0     109    ₹7,794        +1.7%     91d   
  INDIANB     249    PSU BNK    03-Oct-23   412.0       374.5       1189   ₹-44,569      -9.1%     59d   
  LINDEINDIA  185    MFG        01-Sep-23   6,527.5     5,929.7     71     ₹-42,440      -9.2%     91d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  PRESTIGE    4      REALTY     3.635    0.61   +127.5%   +81.6%    1,035.6     525    ₹543,697      +15.9%    
  CDSL        7      FIN SVC    3.090    0.89   +58.4%    +68.0%    932.8       583    ₹543,848      +9.4%     
  TVSMOTOR    8      AUTO       3.023    0.63   +82.3%    +38.3%    1,887.8     288    ₹543,694      +9.7%     
  RATNAMANI   9      METAL      2.997    0.34   +97.2%    +48.1%    3,785.4     143    ₹541,313      +12.2%    
  AUROPHARMA  11     HEALTH     2.872    0.39   +128.7%   +25.1%    1,028.4     529    ₹544,024      +4.9%     
  PCBL        12     MFG        2.799    1.08   +105.6%   +56.4%    253.5       2146   ₹543,940      +12.0%    
  BAJAJ-AUTO  13     AUTO       2.793    0.63   +72.1%    +29.6%    5,767.9     94     ₹542,182      +5.8%     

  HOLDS (20)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BSE         1      FIN SVC    01-Aug-23   274.3       825.7       1556   ₹858,051      +201.1%     +14.9%    
  APARINDS    36     ENERGY     02-May-23   2,777.7     5,464.9     129    ₹346,640      +96.7%      +1.0%     
  JSL         25     METAL      03-Jul-23   335.6       512.9       1196   ₹212,050      +52.8%      +3.8%     
  JYOTHYLAB   31     FMCG       01-Aug-23   294.5       428.7       1449   ₹194,359      +45.5%      +4.7%     
  ITI         9      IT         03-Oct-23   201.0       267.5       2438   ₹162,005      +33.1%      -0.7%     
  MCX         4      FIN SVC    01-Nov-23   475.1       621.8       1013   ₹148,613      +30.9%      +10.4%    
  JINDALSAW   5      METAL      03-Oct-23   177.9       229.4       2755   ₹141,920      +29.0%      +2.8%     
  TRENT       14     CONSUMP    01-Nov-23   2,192.1     2,800.7     219    ₹133,266      +27.8%      +10.1%    
  NTPC        26     ENERGY     01-Aug-23   207.6       253.9       2055   ₹95,124       +22.3%      +7.3%     
  CYIENT      34     IT         01-Nov-23   1,570.7     1,919.6     306    ₹106,770      +22.2%      +10.7%    
  SONATSOFTW  22     IT         01-Nov-23   553.3       665.6       870    ₹97,756       +20.3%      +6.4%     
  NMDC        21     METAL      01-Nov-23   45.9        54.4        10486  ₹88,974       +18.5%      +6.4%     
  COALINDIA   8      ENERGY     01-Nov-23   254.9       301.3       1888   ₹87,710       +18.2%      +6.1%     
  COHANCE     20     MNC        01-Nov-23   573.2       674.3       839    ₹84,907       +17.7%      +10.5%    
  ANGELONE    18     FIN SVC    01-Nov-23   253.2       294.2       1901   ₹77,980       +16.2%      +6.6%     
  LUPIN       48     HEALTH     03-Oct-23   1,161.8     1,283.0     421    ₹51,009       +10.4%      +6.4%     
  SAFARI      43     CON DUR    03-Oct-23   2,063.9     2,199.0     237    ₹32,018       +6.5%       +1.2%     
  SUNDARMFIN  40     FIN SVC    01-Nov-23   3,127.1     3,297.0     153    ₹25,995       +5.4%       +3.6%     
  MAPMYINDIA  53     IT         03-Oct-23   2,109.7     2,200.0     232    ₹20,946       +4.3%       +2.4%     
  LT          44     INFRA      03-Oct-23   2,991.9     3,106.2     163    ₹18,630       +3.8%       +4.3%     

  AFTER: Invested ₹16,089,418 | Cash ₹230,691 | Total ₹16,320,109 | Positions 27/30 | Slot ₹544,154

========================================================================
  REBALANCE #109  —  01 Jan 2024
  NAV: ₹16,944,678  |  Slot: ₹564,823  |  Cash: ₹230,691
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ITI         62     IT         03-Oct-23   201.0       307.2       2438   ₹259,038      +52.9%    90d   
  LUPIN       103    HEALTH     03-Oct-23   1,161.8     1,299.0     421    ₹57,758       +11.8%    90d   
  SUNDARMFIN  155    FIN SVC    01-Nov-23   3,127.1     3,435.7     153    ₹47,208       +9.9%     61d   
  CDSL        77     FIN SVC    01-Dec-23   932.8       893.6       583    ₹-22,891      -4.2%     31d   
  SAFARI      161    CON DUR    03-Oct-23   2,063.9     1,883.6     237    ₹-42,734      -8.7%     90d   
  MAPMYINDIA  226    IT         03-Oct-23   2,109.7     1,922.1     232    ₹-43,511      -8.9%     90d   
  RATNAMANI   81     METAL      01-Dec-23   3,785.4     3,289.9     143    ₹-70,855      -13.1%    31d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IOC         3      OIL&GAS    3.595    0.86   +88.1%    +51.5%    117.3       4815   ₹564,817      +6.7%     
  HAL         6      DEFENCE    3.137    0.99   +126.4%   +47.3%    2,745.8     205    ₹562,890      +5.3%     
  ALKEM       9      HEALTH     3.009    0.55   +68.3%    +43.8%    4,993.4     113    ₹564,255      +4.7%     
  BRIGADE     10     REALTY     2.990    0.56   +98.1%    +53.2%    668.3       845    ₹564,737      +5.3%     
  KAYNES      11     MFG        2.970    0.48   +280.1%   +19.7%    2,669.1     211    ₹563,170      +3.4%     
  CESC        12     ENERGY     2.915    1.09   +90.2%    +52.1%    124.5       4536   ₹564,707      +13.5%    

  HOLDS (20)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BSE         3      FIN SVC    01-Aug-23   274.3       728.0       1556   ₹706,053      +165.4%     -3.8%     
  APARINDS    59     ENERGY     02-May-23   2,777.7     5,985.8     129    ₹413,842      +115.5%     +7.8%     
  JSL         48     METAL      03-Jul-23   335.6       568.9       1196   ₹278,959      +69.5%      +5.0%     
  JYOTHYLAB   50     FMCG       01-Aug-23   294.5       465.8       1449   ₹248,211      +58.2%      +3.6%     
  NTPC        11     ENERGY     01-Aug-23   207.6       292.4       2055   ₹174,280      +40.8%      +4.8%     
  CYIENT      25     IT         01-Nov-23   1,570.7     2,210.0     306    ₹195,640      +40.7%      +5.8%     
  NMDC        21     METAL      01-Nov-23   45.9        63.2        10486  ₹181,081      +37.6%      +9.0%     
  TRENT       9      CONSUMP    01-Nov-23   2,192.1     2,994.6     219    ₹175,733      +36.6%      +2.9%     
  ANGELONE    8      FIN SVC    01-Nov-23   253.2       345.0       1901   ₹174,586      +36.3%      +10.7%    
  MCX         24     FIN SVC    01-Nov-23   475.1       630.6       1013   ₹157,542      +32.7%      +0.8%     
  COALINDIA   20     ENERGY     01-Nov-23   254.9       331.9       1888   ₹145,393      +30.2%      +6.5%     
  SONATSOFTW  29     IT         01-Nov-23   553.3       703.1       870    ₹130,311      +27.1%      -0.6%     
  COHANCE     36     MNC        01-Nov-23   573.2       720.3       839    ₹123,501      +25.7%      +5.1%     
  JINDALSAW   33     METAL      03-Oct-23   177.9       205.7       2755   ₹76,733       +15.7%      -1.7%     
  LT          52     INFRA      03-Oct-23   2,991.9     3,432.1     163    ₹71,757       +14.7%      +3.3%     
  PRESTIGE    4      REALTY     01-Dec-23   1,035.6     1,186.2     525    ₹79,056       +14.5%      +7.5%     
  BAJAJ-AUTO  13     AUTO       01-Dec-23   5,767.9     6,392.8     94     ₹58,742       +10.8%      +5.2%     
  TVSMOTOR    19     AUTO       01-Dec-23   1,887.8     1,995.8     288    ₹31,102       +5.7%       +3.4%     
  AUROPHARMA  26     HEALTH     01-Dec-23   1,028.4     1,074.6     529    ₹24,419       +4.5%       +3.1%     
  PCBL        31     MFG        01-Dec-23   253.5       243.4       2146   ₹-21,660      -4.0%       +1.0%     

  AFTER: Invested ₹16,393,181 | Cash ₹547,478 | Total ₹16,940,660 | Positions 26/30 | Slot ₹564,823

========================================================================
  REBALANCE #110  —  01 Feb 2024
  NAV: ₹17,738,197  |  Slot: ₹591,273  |  Cash: ₹547,478
========================================================================

  EXITS (11)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  APARINDS    77     ENERGY     02-May-23   2,777.7     6,226.6     129    ₹444,903      +124.2%   275d  
  JSL         134    METAL      03-Jul-23   335.6       560.0       1196   ₹268,401      +66.9%    213d  
  NMDC        69     METAL      01-Nov-23   45.9        67.3        10486  ₹224,785      +46.7%    92d   
  MCX         68     FIN SVC    01-Nov-23   475.1       686.5       1013   ₹214,081      +44.5%    92d   
  SONATSOFTW  79     IT         01-Nov-23   553.3       743.7       870    ₹165,637      +34.4%    92d   
  ANGELONE    111    FIN SVC    01-Nov-23   253.2       324.7       1901   ₹135,848      +28.2%    92d   
  CYIENT      149    IT         01-Nov-23   1,570.7     1,862.7     306    ₹89,340       +18.6%    92d   
  COHANCE     183    MNC        01-Nov-23   573.2       660.0       839    ₹72,825       +15.1%    92d   
  LT          146    INFRA      03-Oct-23   2,991.9     3,308.0     163    ₹51,533       +10.6%    121d  
  AUROPHARMA  63     HEALTH     01-Dec-23   1,028.4     1,065.3     529    ₹19,498       +3.6%     62d   
  ALKEM       87     HEALTH     01-Jan-24   4,993.4     4,734.7     113    ₹-29,232      -5.2%     31d   

  ENTRIES (12)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  IRFC        1      FIN SVC    4.383    1.11   +423.4%   +136.1%   164.1       3602   ₹591,189      +18.3%    
  BSOFT       6      IT         3.125    1.04   +187.7%   +54.0%    807.8       731    ₹590,486      +5.3%     
  NHPC        8      ENERGY     3.086    0.83   +124.2%   +79.2%    85.8        6892   ₹591,266      +17.7%    
  MEDANTA     9      HEALTH     3.082    0.45   +165.3%   +56.8%    1,198.0     493    ₹590,599      +11.7%    
  GLAXO       10     HEALTH     2.961    0.56   +76.8%    +55.8%    2,108.7     280    ₹590,424      +3.1%     
  SUZLON      11     ENERGY     2.979    1.18   +382.0%   +52.8%    48.2        12267  ₹591,269      +13.0%    
  RVNL        14     PSE        2.852    1.17   +297.9%   +90.3%    293.9       2011   ₹591,098      +16.8%    
  TMPV        16     AUTO       2.837    0.98   +118.6%   +39.8%    858.4       688    ₹590,572      +7.3%     
  TATAPOWER   17     ENERGY     2.817    1.13   +89.4%    +62.6%    382.4       1546   ₹591,226      +8.5%     
  SWSOLAR     18     ENERGY     2.818    0.68   +110.3%   +112.3%   573.2       1031   ₹591,021      +12.6%    
  COCHINSHIP  19     DEFENCE    2.812    1.18   +267.6%   +91.7%    896.3       659    ₹590,633      +12.1%    
  HINDPETRO   21     OIL&GAS    2.711    1.13   +89.4%    +89.4%    282.9       2090   ₹591,196      +5.5%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BSE         10     FIN SVC    01-Aug-23   274.3       824.0       1556   ₹855,377      +200.4%     +10.5%    
  JYOTHYLAB   61     FMCG       01-Aug-23   294.5       490.6       1449   ₹284,090      +66.6%      -0.9%     
  NTPC        29     ENERGY     01-Aug-23   207.6       304.0       2055   ₹198,047      +46.4%      +3.3%     
  TRENT       8      CONSUMP    01-Nov-23   2,192.1     3,095.0     219    ₹197,731      +41.2%      -0.6%     
  JINDALSAW   48     METAL      03-Oct-23   177.9       249.8       2755   ₹198,177      +40.4%      +3.7%     
  COALINDIA   47     ENERGY     01-Nov-23   254.9       353.5       1888   ₹186,256      +38.7%      +4.9%     
  BAJAJ-AUTO  11     AUTO       01-Dec-23   5,767.9     7,303.5     94     ₹144,351      +26.6%      +5.9%     
  PCBL        19     MFG        01-Dec-23   253.5       307.8       2146   ₹116,503      +21.4%      +8.8%     
  PRESTIGE    35     REALTY     01-Dec-23   1,035.6     1,239.1     525    ₹106,822      +19.6%      -0.0%     
  IOC         6      OIL&GAS    01-Jan-24   117.3       134.8       4815   ₹84,311       +14.9%      +6.8%     
  BRIGADE     25     REALTY     01-Jan-24   668.3       754.0       845    ₹72,359       +12.8%      +5.2%     
  HAL         12     DEFENCE    01-Jan-24   2,745.8     2,912.1     205    ₹34,096       +6.1%       +2.0%     
  CESC        21     ENERGY     01-Jan-24   124.5       131.6       4536   ₹32,181       +5.7%       +5.7%     
  KAYNES      53     MFG        01-Jan-24   2,669.1     2,813.9     211    ₹30,574       +5.4%       +2.6%     
  TVSMOTOR    49     AUTO       01-Dec-23   1,887.8     1,972.3     288    ₹24,326       +4.5%       +0.1%     

  AFTER: Invested ₹17,381,512 | Cash ₹348,265 | Total ₹17,729,777 | Positions 27/30 | Slot ₹591,273

========================================================================
  REBALANCE #111  —  01 Mar 2024
  NAV: ₹17,834,290  |  Slot: ₹594,476  |  Cash: ₹348,265
========================================================================

  EXITS (13)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  JYOTHYLAB   178    FMCG       01-Aug-23   294.5       436.8       1449   ₹206,108      +48.3%    213d  
  JINDALSAW   155    METAL      03-Oct-23   177.9       238.7       2755   ₹167,565      +34.2%    150d  
  TVSMOTOR    70     AUTO       01-Dec-23   1,887.8     2,216.8     288    ₹94,744       +17.4%    91d   
  PCBL        139    MFG        01-Dec-23   253.5       285.7       2146   ₹69,106       +12.7%    91d   
  PRESTIGE    104    REALTY     01-Dec-23   1,035.6     1,165.0     525    ₹67,923       +12.5%    91d   
  BRIGADE     108    REALTY     01-Jan-24   668.3       743.4       845    ₹63,397       +11.2%    60d   
  KAYNES      73     MFG        01-Jan-24   2,669.1     2,945.5     211    ₹58,331       +10.4%    60d   
  SWSOLAR     60     ENERGY     01-Feb-24   573.2       597.8       1031   ₹25,311       +4.3%     29d   
  TATAPOWER   65     ENERGY     01-Feb-24   382.4       371.2       1546   ₹-17,409      -2.9%     29d   
  GLAXO       89     HEALTH     01-Feb-24   2,108.7     2,032.0     280    ₹-21,451      -3.6%     29d   
  CESC        111    ENERGY     01-Jan-24   124.5       119.0       4536   ₹-24,909      -4.4%     60d   
  IRFC        2      FIN SVC    01-Feb-24   164.1       142.4       3602   ₹-78,292      -13.2%    29d   
  RVNL        58     PSE        01-Feb-24   293.9       243.2       2011   ₹-102,009     -17.3%    29d   

  ENTRIES (13)
  [52w filter blocked 1: MRPL(-20.8%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  OIL         1      OIL&GAS    3.470    0.95   +148.2%   +93.0%    366.3       1623   ₹594,441      +9.2%     
  OFSS        2      IT         3.336    0.96   +156.5%   +92.2%    6,933.3     85     ₹589,328      +3.6%     
  TATAINVEST  3      FIN SVC    3.330    0.50   +277.7%   +80.4%    754.4       787    ₹593,743      +21.0%    
  OLECTRA     5      AUTO       3.224    0.54   +392.0%   +64.3%    1,961.5     303    ₹594,324      +0.5%     
  ZYDUSLIFE   6      HEALTH     3.180    0.39   +99.4%    +45.1%    912.6       651    ₹594,098      +5.3%     
  ADANIGREEN  8      ENERGY     3.061    1.03   +247.1%   +91.4%    1,969.6     301    ₹592,835      +5.0%     
  LUPIN       10     HEALTH     2.947    0.42   +143.4%   +26.8%    1,607.5     369    ₹593,153      +2.5%     
  JBMA        13     AUTO       2.803    0.69   +272.7%   +72.4%    1,069.3     555    ₹593,452      +1.2%     
  BEL         14     DEFENCE    2.782    1.11   +118.4%   +41.5%    201.9       2944   ₹594,277      +5.8%     
  EIHOTEL     16     CONSUMP    2.719    0.93   +152.4%   +70.7%    403.4       1473   ₹594,193      +5.1%     
  ZOMATO      17     CONSUMP    2.695    1.08   +200.8%   +40.4%    166.5       3570   ₹594,405      +6.7%     
  BPCL        19     OIL&GAS    2.658    0.92   +104.2%   +49.8%    277.9       2138   ₹594,249      +3.3%     
  SWANCORP    21     ENERGY     2.685    0.96   +190.4%   +80.2%    761.9       780    ₹594,293      +7.4%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BSE         22     FIN SVC    01-Aug-23   274.3       772.1       1556   ₹774,648      +181.5%     +0.4%     
  TRENT       11     CONSUMP    01-Nov-23   2,192.1     3,891.5     219    ₹372,153      +77.5%      +3.3%     
  NTPC        28     ENERGY     01-Aug-23   207.6       327.5       2055   ₹246,266      +57.7%      +3.3%     
  COALINDIA   44     ENERGY     01-Nov-23   254.9       392.4       1888   ₹259,761      +54.0%      +2.1%     
  BAJAJ-AUTO  21     AUTO       01-Dec-23   5,767.9     7,670.2     94     ₹178,813      +33.0%      -0.2%     
  IOC         18     OIL&GAS    01-Jan-24   117.3       152.5       4815   ₹169,489      +30.0%      -1.7%     
  HINDPETRO   55     OIL&GAS    01-Feb-24   282.9       323.1       2090   ₹84,078       +14.2%      +0.1%     
  HAL         33     DEFENCE    01-Jan-24   2,745.8     3,086.9     205    ₹69,921       +12.4%      +4.5%     
  TMPV        15     AUTO       01-Feb-24   858.4       955.0       688    ₹66,463       +11.3%      +5.6%     
  MEDANTA     43     HEALTH     01-Feb-24   1,198.0     1,298.2     493    ₹49,405       +8.4%       -3.6%     
  NHPC        36     ENERGY     01-Feb-24   85.8        85.6        6892   ₹-1,109       -0.2%       +0.4%     
  COCHINSHIP  59     DEFENCE    01-Feb-24   896.3       862.6       659    ₹-22,206      -3.8%       +2.6%     
  BSOFT       46     IT         01-Feb-24   807.8       755.1       731    ₹-38,542      -6.5%       -3.4%     
  SUZLON      30     ENERGY     01-Feb-24   48.2        44.3        12267  ₹-47,841      -8.1%       -1.4%     

  AFTER: Invested ₹17,498,663 | Cash ₹326,464 | Total ₹17,825,127 | Positions 27/30 | Slot ₹594,476

========================================================================
  REBALANCE #112  —  01 Apr 2024
  NAV: ₹18,166,002  |  Slot: ₹605,533  |  Cash: ₹326,464
========================================================================

  EXITS (8)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  NTPC        85     ENERGY     01-Aug-23   207.6       326.4       2055   ₹244,013      +57.2%    244d  
  COALINDIA   56     ENERGY     01-Nov-23   254.9       388.7       1888   ₹252,624      +52.5%    152d  
  IOC         47     OIL&GAS    01-Jan-24   117.3       152.4       4815   ₹168,838      +29.9%    91d   
  HINDPETRO   121    OIL&GAS    01-Feb-24   282.9       292.9       2090   ₹20,945       +3.5%     60d   
  BEL         60     DEFENCE    01-Mar-24   201.9       208.0       2944   ₹18,075       +3.0%     31d   
  NHPC        58     ENERGY     01-Feb-24   85.8        86.2        6892   ₹3,160        +0.5%     60d   
  ADANIGREEN  172    ENERGY     01-Mar-24   1,969.6     1,888.3     301    ₹-24,456      -4.1%     31d   
  JBMA        90     AUTO       01-Mar-24   1,069.3     918.0       555    ₹-83,988      -14.2%    31d   

  ENTRIES (8)
  [52w filter blocked 1: JAIBALAJI(-27.9%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SCHNEIDER   3      ENERGY     3.789    0.78   +404.4%   +88.0%    778.1       778    ₹605,362      +18.9%    
  ANANDRATHI  4      FIN SVC    3.491    -0.02  +353.1%   +40.8%    894.4       677    ₹605,503      -1.1%     
  KPIL        10     INFRA      2.765    0.63   +102.8%   +72.1%    1,092.5     554    ₹605,262      +7.8%     
  CUMMINSIND  11     INFRA      2.740    0.77   +84.2%    +52.5%    2,927.3     206    ₹603,030      +6.4%     
  BOSCHLTD    12     AUTO       2.716    0.68   +70.7%    +38.6%    29,732.3    20     ₹594,646      +2.7%     
  SUNPHARMA   13     HEALTH     2.702    0.47   +71.1%    +30.8%    1,598.7     378    ₹604,304      +3.3%     
  TORNTPOWER  15     ENERGY     2.569    0.67   +172.7%   +59.9%    1,379.4     438    ₹604,156      +13.6%    
  CHALET      20     CONSUMP    2.436    0.38   +146.9%   +36.7%    900.4       672    ₹605,070      +10.7%    

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BSE         8      FIN SVC    01-Aug-23   274.3       895.5       1556   ₹966,676      +226.5%     +18.3%    
  TRENT       24     CONSUMP    01-Nov-23   2,192.1     3,877.1     219    ₹368,996      +76.9%      -0.9%     
  BAJAJ-AUTO  9      AUTO       01-Dec-23   5,767.9     8,626.2     94     ₹268,677      +49.6%      +4.3%     
  HAL         40     DEFENCE    01-Jan-24   2,745.8     3,330.6     205    ₹119,885      +21.3%      +6.8%     
  EIHOTEL     5      CONSUMP    01-Mar-24   403.4       471.7       1473   ₹100,666      +16.9%      +12.5%    
  OFSS        2      IT         01-Mar-24   6,933.3     8,048.0     85     ₹94,750       +16.1%      +7.5%     
  TMPV        11     AUTO       01-Feb-24   858.4       969.6       688    ₹76,482       +13.0%      +1.9%     
  MEDANTA     22     HEALTH     01-Feb-24   1,198.0     1,330.8     493    ₹65,472       +11.1%      +4.8%     
  ZOMATO      16     CONSUMP    01-Mar-24   166.5       184.5       3570   ₹64,260       +10.8%      +10.0%    
  ZYDUSLIFE   7      HEALTH     01-Mar-24   912.6       986.0       651    ₹47,799       +8.0%       +2.8%     
  COCHINSHIP  37     DEFENCE    01-Feb-24   896.3       955.8       659    ₹39,269       +6.6%       +10.7%    
  OIL         21     OIL&GAS    01-Mar-24   366.3       373.1       1623   ₹11,130       +1.9%       +2.3%     
  LUPIN       20     HEALTH     01-Mar-24   1,607.5     1,606.7     369    ₹-292         -0.0%       +0.7%     
  OLECTRA     38     AUTO       01-Mar-24   1,961.5     1,914.0     303    ₹-14,371      -2.4%       +4.4%     
  BPCL        62     OIL&GAS    01-Mar-24   277.9       267.9       2138   ₹-21,547      -3.6%       +0.3%     
  BSOFT       88     IT         01-Feb-24   807.8       742.4       731    ₹-47,803      -8.1%       +0.7%     
  SWANCORP    89     ENERGY     01-Mar-24   761.9       674.1       780    ₹-68,497      -11.5%      +4.8%     
  TATAINVEST  32     FIN SVC    01-Mar-24   754.4       647.0       787    ₹-84,559      -14.2%      -4.3%     
  SUZLON      41     ENERGY     01-Feb-24   48.2        41.3        12267  ₹-84,029      -14.2%      +3.9%     

  AFTER: Invested ₹17,631,992 | Cash ₹528,278 | Total ₹18,160,270 | Positions 27/30 | Slot ₹605,533

========================================================================
  REBALANCE #113  —  02 May 2024
  NAV: ₹18,845,497  |  Slot: ₹628,183  |  Cash: ₹528,278
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BPCL        55     OIL&GAS    01-Mar-24   277.9       283.0       2138   ₹10,821       +1.8%     62d   
  OFSS        123    IT         01-Mar-24   6,933.3     6,978.4     85     ₹3,837        +0.7%     62d   
  CHALET      90     CONSUMP    01-Apr-24   900.4       878.6       672    ₹-14,633      -2.4%     31d   
  SUNPHARMA   110    HEALTH     01-Apr-24   1,598.7     1,490.5     378    ₹-40,893      -6.8%     31d   
  OLECTRA     212    AUTO       01-Mar-24   1,961.5     1,721.4     303    ₹-72,731      -12.2%    62d   
  SWANCORP    207    ENERGY     01-Mar-24   761.9       606.3       780    ₹-121,362     -20.4%    62d   
  BSOFT       281    IT         01-Feb-24   807.8       626.8       731    ₹-132,260     -22.4%    91d   

  ENTRIES (6)
  [52w filter blocked 1: JAIBALAJI(-23.5%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  POWERINDIA  7      ENERGY     2.847    0.63   +204.0%   +66.1%    9,777.6     64     ₹625,765      +19.1%    
  DIXON       11     CON DUR    2.688    0.78   +187.2%   +43.0%    8,403.6     74     ₹621,863      +6.4%     
  INDIGO      15     CONSUMP    2.643    0.83   +106.7%   +43.5%    4,101.3     153    ₹627,502      +10.4%    
  EXIDEIND    17     AUTO       2.532    0.97   +149.1%   +49.5%    457.5       1373   ₹628,170      +10.3%    
  VOLTAS      18     CON DUR    2.494    0.81   +73.9%    +47.4%    1,463.8     429    ₹627,956      +10.1%    
  SUNDARMFIN  19     FIN SVC    2.475    0.46   +109.9%   +37.3%    4,786.9     131    ₹627,080      +6.1%     

  HOLDS (20)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BSE         6      FIN SVC    01-Aug-23   274.3       944.7       1556   ₹1,043,215    +244.5%     +1.0%     
  TRENT       5      CONSUMP    01-Nov-23   2,192.1     4,635.0     219    ₹534,997      +111.4%     +11.0%    
  BAJAJ-AUTO  44     AUTO       01-Dec-23   5,767.9     8,691.5     94     ₹274,819      +50.7%      +2.6%     
  COCHINSHIP  14     DEFENCE    01-Feb-24   896.3       1,308.6     659    ₹271,733      +46.0%      +12.9%    
  HAL         9      DEFENCE    01-Jan-24   2,745.8     3,862.8     205    ₹228,992      +40.7%      +5.5%     
  MEDANTA     26     HEALTH     01-Feb-24   1,198.0     1,423.9     493    ₹111,403      +18.9%      +1.9%     
  ZOMATO      12     CONSUMP    01-Mar-24   166.5       195.4       3570   ₹103,351      +17.4%      +4.6%     
  TMPV        28     AUTO       01-Feb-24   858.4       1,004.6     688    ₹100,586      +17.0%      +3.2%     
  EIHOTEL     39     CONSUMP    01-Mar-24   403.4       464.0       1473   ₹89,326       +15.0%      +0.9%     
  ANANDRATHI  3      FIN SVC    01-Apr-24   894.4       986.8       677    ₹62,591       +10.3%      +1.8%     
  CUMMINSIND  4      INFRA      01-Apr-24   2,927.3     3,220.5     206    ₹60,395       +10.0%      +5.7%     
  KPIL        8      INFRA      01-Apr-24   1,092.5     1,192.4     554    ₹55,302       +9.1%       +4.8%     
  OIL         13     OIL&GAS    01-Mar-24   366.3       398.0       1623   ₹51,553       +8.7%       +3.1%     
  ZYDUSLIFE   23     HEALTH     01-Mar-24   912.6       974.4       651    ₹40,223       +6.8%       +2.7%     
  SCHNEIDER   2      ENERGY     01-Apr-24   778.1       828.8       778    ₹39,406       +6.5%       +9.3%     
  TORNTPOWER  19     ENERGY     01-Apr-24   1,379.4     1,460.4     438    ₹35,485       +5.9%       +1.9%     
  LUPIN       51     HEALTH     01-Mar-24   1,607.5     1,630.3     369    ₹8,434        +1.4%       +2.3%     
  BOSCHLTD    15     AUTO       01-Apr-24   29,732.3    30,113.4    20     ₹7,622        +1.3%       +3.5%     
  SUZLON      65     ENERGY     01-Feb-24   48.2        41.7        12267  ₹-79,736      -13.5%      +0.9%     
  TATAINVEST  58     FIN SVC    01-Mar-24   754.4       647.3       787    ₹-84,287      -14.2%      -3.4%     

  AFTER: Invested ₹18,270,719 | Cash ₹570,315 | Total ₹18,841,034 | Positions 26/30 | Slot ₹628,183

========================================================================
  REBALANCE #114  —  03 Jun 2024
  NAV: ₹19,520,043  |  Slot: ₹650,668  |  Cash: ₹570,315
========================================================================

  EXITS (13)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COCHINSHIP  1      DEFENCE    01-Feb-24   896.3       1,987.1     659    ₹718,870      +121.7%   123d  
  HAL         3      DEFENCE    01-Jan-24   2,745.8     5,160.9     205    ₹495,102      +88.0%    154d  
  ZYDUSLIFE   95     HEALTH     01-Mar-24   912.6       1,018.0     651    ₹68,633       +11.6%    94d   
  TMPV        161    AUTO       01-Feb-24   858.4       928.9       688    ₹48,511       +8.2%     123d  
  ZOMATO      103    CONSUMP    01-Mar-24   166.5       175.4       3570   ₹31,951       +5.4%     94d   
  EIHOTEL     184    CONSUMP    01-Mar-24   403.4       419.4       1473   ₹23,630       +4.0%     94d   
  SUZLON      26     ENERGY     01-Feb-24   48.2        50.0        12267  ₹22,081       +3.7%     123d  
  BOSCHLTD    143    AUTO       01-Apr-24   29,732.3    29,444.7    20     ₹-5,753       -1.0%     63d   
  LUPIN       135    HEALTH     01-Mar-24   1,607.5     1,567.3     369    ₹-14,824      -2.5%     94d   
  MEDANTA     288    HEALTH     01-Feb-24   1,198.0     1,165.8     493    ₹-15,844      -2.7%     123d  
  VOLTAS      72     CON DUR    02-May-24   1,463.8     1,389.0     429    ₹-32,056      -5.1%     32d   
  SUNDARMFIN  157    FIN SVC    02-May-24   4,786.9     4,377.0     131    ₹-53,699      -8.6%     32d   
  TATAINVEST  173    FIN SVC    01-Mar-24   754.4       634.1       787    ₹-94,717      -16.0%    94d   

  ENTRIES (14)
  [52w filter blocked 1: JAIBALAJI(-31.4%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HINDZINC    2      METAL      3.843    0.74   +140.0%   +128.5%   646.6       1006   ₹650,518      +11.1%    
  SIEMENS     4      ENERGY     3.554    0.93   +114.1%   +59.3%    4,254.8     152    ₹646,737      +6.5%     
  VEDL        7      METAL      2.956    1.09   +80.8%    +76.7%    398.1       1634   ₹650,427      +5.5%     
  MAZDOCK     8      DEFENCE    2.889    1.11   +327.5%   +55.7%    1,602.9     405    ₹649,173      +14.1%    
  BDL         11     DEFENCE    2.821    1.16   +198.4%   +70.1%    1,583.6     410    ₹649,264      +22.1%    
  THERMAX     12     ENERGY     2.776    0.61   +149.5%   +51.8%    5,601.1     116    ₹649,729      +11.3%    
  PRESTIGE    15     REALTY     2.666    0.94   +250.6%   +41.9%    1,731.5     375    ₹649,329      +12.4%    
  CGPOWER     16     ENERGY     2.530    0.47   +92.6%    +60.2%    683.4       952    ₹650,577      +10.2%    
  SOLARINDS   17     DEFENCE    2.504    0.51   +158.6%   +44.8%    9,847.1     66     ₹649,909      +6.6%     
  LINDEINDIA  18     MFG        2.557    0.76   +130.9%   +61.2%    8,887.2     73     ₹648,762      +3.6%     
  BRIGADE     20     REALTY     2.432    1.07   +157.7%   +36.9%    1,023.3     635    ₹649,815      +15.4%    
  TITAGARH    21     INFRA      3.084    0.97   +341.5%   +51.9%    1,489.9     436    ₹649,603      +20.1%    
  ASHOKLEY    22     AUTO       2.424    0.93   +62.4%    +39.8%    112.4       5786   ₹650,615      +11.7%    
  ESCORTS     23     MFG        2.385    0.74   +89.4%    +34.6%    3,820.6     170    ₹649,496      +5.0%     

  HOLDS (13)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BSE         19     FIN SVC    01-Aug-23   274.3       894.9       1556   ₹965,648      +226.3%     -0.2%     
  TRENT       23     CONSUMP    01-Nov-23   2,192.1     4,652.6     219    ₹538,849      +112.2%     +2.2%     
  BAJAJ-AUTO  68     AUTO       01-Dec-23   5,767.9     8,906.0     94     ₹294,978      +54.4%      +3.9%     
  CUMMINSIND  22     INFRA      01-Apr-24   2,927.3     3,618.2     206    ₹142,325      +23.6%      +2.9%     
  DIXON       12     CON DUR    02-May-24   8,403.6     9,878.1     74     ₹109,114      +17.5%      +10.6%    
  OIL         75     OIL&GAS    01-Mar-24   366.3       422.6       1623   ₹91,412       +15.4%      +4.4%     
  POWERINDIA  9      ENERGY     02-May-24   9,777.6     11,094.8    64     ₹84,304       +13.5%      +8.2%     
  ANANDRATHI  8      FIN SVC    01-Apr-24   894.4       1,013.3     677    ₹80,469       +13.3%      +1.3%     
  KPIL        67     INFRA      01-Apr-24   1,092.5     1,195.6     554    ₹57,117       +9.4%       +1.7%     
  EXIDEIND    15     AUTO       02-May-24   457.5       497.8       1373   ₹55,371       +8.8%       +6.1%     
  TORNTPOWER  35     ENERGY     01-Apr-24   1,379.4     1,469.3     438    ₹39,392       +6.5%       +6.0%     
  INDIGO      55     CONSUMP    02-May-24   4,101.3     4,290.8     153    ₹28,990       +4.6%       +3.6%     
  SCHNEIDER   56     ENERGY     01-Apr-24   778.1       719.7       778    ₹-45,435      -7.5%       -8.9%     

  AFTER: Invested ₹19,106,560 | Cash ₹402,685 | Total ₹19,509,245 | Positions 27/30 | Slot ₹650,668

========================================================================
  REBALANCE #115  —  01 Jul 2024
  NAV: ₹20,671,637  |  Slot: ₹689,055  |  Cash: ₹402,685
========================================================================

  EXITS (15)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DIXON       9      CON DUR    02-May-24   8,403.6     12,436.5    74     ₹298,436      +48.0%    60d   
  MAZDOCK     8      DEFENCE    03-Jun-24   1,602.9     2,162.0     405    ₹226,441      +34.9%    28d   
  EXIDEIND    6      AUTO       02-May-24   457.5       561.4       1373   ₹142,692      +22.7%    60d   
  TITAGARH    26     INFRA      03-Jun-24   1,489.9     1,822.4     436    ₹144,958      +22.3%    28d   
  SCHNEIDER   43     ENERGY     01-Apr-24   778.1       904.7       778    ₹98,456       +16.3%    91d   
  SIEMENS     18     ENERGY     03-Jun-24   4,254.8     4,604.4     152    ₹53,129       +8.2%     28d   
  PRESTIGE    27     REALTY     03-Jun-24   1,731.5     1,836.5     375    ₹39,367       +6.1%     28d   
  KPIL        177    INFRA      01-Apr-24   1,092.5     1,151.2     554    ₹32,479       +5.4%     91d   
  INDIGO      164    CONSUMP    02-May-24   4,101.3     4,215.0     153    ₹17,397       +2.8%     60d   
  TORNTPOWER  225    ENERGY     01-Apr-24   1,379.4     1,415.8     438    ₹15,971       +2.6%     91d   
  SOLARINDS   129    DEFENCE    03-Jun-24   9,847.1     10,085.1    66     ₹15,707       +2.4%     28d   
  VEDL        23     METAL      03-Jun-24   398.1       404.3       1634   ₹10,157       +1.6%     28d   
  ASHOKLEY    55     AUTO       03-Jun-24   112.4       113.5       5786   ₹6,272        +1.0%     28d   
  BRIGADE     48     REALTY     03-Jun-24   1,023.3     1,010.1     635    ₹-8,395       -1.3%     28d   
  LINDEINDIA  157    MFG        03-Jun-24   8,887.2     8,417.3     73     ₹-34,301      -5.3%     28d   

  ENTRIES (15)
  [52w filter blocked 1: JAIBALAJI(-30.2%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MOTHERSON   3      AUTO       3.449    0.88   +144.0%   +67.8%    129.4       5325   ₹688,982      +11.3%    
  AEGISLOG    5      INFRA      3.071    0.63   +151.7%   +119.5%   846.1       814    ₹688,741      +7.5%     
  UNOMINDA    7      AUTO       2.907    0.66   +95.8%    +69.8%    1,137.1     605    ₹687,968      +12.5%    
  COROMANDEL  11     MFG        2.510    0.87   +71.0%    +50.3%    1,574.0     437    ₹687,846      +6.9%     
  EMAMILTD    12     FMCG       2.439    0.52   +76.2%    +65.7%    693.5       993    ₹688,680      +5.5%     
  WHIRLPOOL   13     CON DUR    2.430    0.50   +35.4%    +60.1%    1,944.7     354    ₹688,423      +9.3%     
  360ONE      14     FIN SVC    2.354    0.74   +126.5%   +50.3%    954.2       722    ₹688,943      +13.8%    
  CROMPTON    15     CON DUR    2.207    0.70   +44.8%    +53.7%    409.9       1680   ₹688,681      +1.1%     
  HONAUT      16     MFG        2.210    0.62   +34.0%    +51.0%    56,410.4    12     ₹676,925      +2.9%     
  ENDURANCE   17     AUTO       2.180    0.59   +71.3%    +53.5%    2,685.6     256    ₹687,526      +6.2%     
  FINCABLES   18     ENERGY     2.126    0.74   +93.4%    +69.9%    1,619.0     425    ₹688,093      +7.5%     
  JKPAPER     20     MFG        2.227    1.09   +68.6%    +64.9%    522.0       1320   ₹689,023      +10.1%    
  RAYMOND     21     CONSUMP    2.452    0.97   +76.8%    +73.6%    3,043.4     226    ₹687,797      +16.4%    
  CHOLAHLDNG  22     FIN SVC    2.029    0.72   +72.1%    +48.4%    1,607.6     428    ₹688,061      +24.6%    
  BIKAJI      23     FMCG       2.142    0.58   +73.1%    +48.9%    716.9       961    ₹688,986      +4.4%     

  HOLDS (12)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  BSE         36     FIN SVC    01-Aug-23   274.3       855.3       1556   ₹904,144      +211.9%     -1.7%     
  TRENT       10     CONSUMP    01-Nov-23   2,192.1     5,504.3     219    ₹725,371      +151.1%     +6.6%     
  BAJAJ-AUTO  82     AUTO       01-Dec-23   5,767.9     9,167.8     94     ₹319,587      +58.9%      +0.2%     
  POWERINDIA  12     ENERGY     02-May-24   9,777.6     13,003.8    64     ₹206,479      +33.0%      +13.3%    
  CUMMINSIND  42     INFRA      01-Apr-24   2,927.3     3,884.9     206    ₹197,249      +32.7%      +4.0%     
  OIL         69     OIL&GAS    01-Mar-24   366.3       450.7       1623   ₹137,009      +23.0%      +4.8%     
  ANANDRATHI  13     FIN SVC    01-Apr-24   894.4       970.6       677    ₹51,616       +8.5%       -0.4%     
  ESCORTS     20     MFG        03-Jun-24   3,820.6     4,067.5     170    ₹41,971       +6.5%       +0.8%     
  CGPOWER     94     ENERGY     03-Jun-24   683.4       718.6       952    ₹33,512       +5.2%       +6.5%     
  BDL         19     DEFENCE    03-Jun-24   1,583.6     1,602.2     410    ₹7,646        +1.2%       +6.9%     
  HINDZINC    7      METAL      03-Jun-24   646.6       609.8       1006   ₹-37,038      -5.7%       -0.1%     
  THERMAX     77     ENERGY     03-Jun-24   5,601.1     5,244.3     116    ₹-41,387      -6.4%       +1.7%     

  AFTER: Invested ₹19,988,173 | Cash ₹671,216 | Total ₹20,659,389 | Positions 27/30 | Slot ₹689,055

========================================================================
  REBALANCE #116  —  01 Aug 2024
  NAV: ₹20,543,956  |  Slot: ₹684,799  |  Cash: ₹671,216
========================================================================

  EXITS (11)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BSE         112    FIN SVC    01-Aug-23   274.3       878.3       1556   ₹939,897      +220.2%   366d  
  OIL         9      OIL&GAS    01-Mar-24   366.3       567.3       1623   ₹326,288      +54.9%    153d  
  CUMMINSIND  89     INFRA      01-Apr-24   2,927.3     3,738.1     206    ₹167,011      +27.7%    122d  
  CGPOWER     103    ENERGY     03-Jun-24   683.4       725.6       952    ₹40,148       +6.2%     59d   
  HONAUT      183    MFG        01-Jul-24   56,410.4    53,971.9    12     ₹-29,262      -4.3%     31d   
  ENDURANCE   106    AUTO       01-Jul-24   2,685.6     2,550.7     256    ₹-34,540      -5.0%     31d   
  FINCABLES   101    ENERGY     01-Jul-24   1,619.0     1,501.3     425    ₹-50,027      -7.3%     31d   
  JKPAPER     178    MFG        01-Jul-24   522.0       481.9       1320   ₹-52,883      -7.7%     31d   
  THERMAX     157    ENERGY     03-Jun-24   5,601.1     5,139.2     116    ₹-53,578      -8.2%     59d   
  AEGISLOG    221    INFRA      01-Jul-24   846.1       760.5       814    ₹-69,671      -10.1%    31d   
  RAYMOND     417    CONSUMP    01-Jul-24   3,043.4     1,996.1     226    ₹-236,679     -34.4%    31d   

  ENTRIES (12)
  [52w filter blocked 1: JAIBALAJI(-29.3%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  OFSS        5      IT         2.732    0.95   +186.5%   +48.1%    10,120.6    67     ₹678,083      +1.7%     
  GRANULES    6      HEALTH     2.721    1.14   +104.3%   +49.6%    629.6       1087   ₹684,348      +15.5%    
  COHANCE     7      MNC        2.653    0.41   +104.9%   +49.9%    990.3       691    ₹684,297      +12.7%    
  PERSISTENT  8      IT         2.466    0.84   +91.3%    +42.7%    4,750.7     144    ₹684,103      +2.8%     
  ZYDUSLIFE   9      HEALTH     2.583    0.63   +103.4%   +30.5%    1,227.3     557    ₹683,629      +5.2%     
  SUZLON      10     ENERGY     3.346    0.88   +263.5%   +63.4%    68.0        10073  ₹684,763      +13.4%    
  FSL         11     IT         2.330    1.17   +123.9%   +42.4%    291.3       2350   ₹684,648      +19.3%    
  TVSMOTOR    12     AUTO       2.348    0.70   +92.9%    +25.4%    2,564.6     267    ₹684,739      +5.0%     
  ZOMATO      15     CONSUMP    2.449    0.89   +201.7%   +21.2%    234.1       2925   ₹684,713      +6.6%     
  SUNTV       17     MEDIA      2.150    0.48   +78.1%    +35.8%    852.7       803    ₹684,718      +8.2%     
  COLPAL      18     FMCG       2.174    0.33   +88.0%    +21.1%    3,238.2     211    ₹683,262      +7.2%     
  VGUARD      21     CON DUR    2.112    0.54   +60.8%    +34.8%    459.3       1491   ₹684,776      +1.6%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       5      CONSUMP    01-Nov-23   2,192.1     5,759.0     219    ₹781,134      +162.7%     +5.3%     
  BAJAJ-AUTO  61     AUTO       01-Dec-23   5,767.9     9,358.3     94     ₹337,496      +62.2%      +2.2%     
  POWERINDIA  49     ENERGY     02-May-24   9,777.6     12,384.2    64     ₹166,824      +26.7%      +3.3%     
  EMAMILTD    7      FMCG       01-Jul-24   693.5       792.6       993    ₹98,369       +14.3%      +5.5%     
  360ONE      28     FIN SVC    01-Jul-24   954.2       1,077.9     722    ₹89,322       +13.0%      +8.9%     
  WHIRLPOOL   30     CON DUR    01-Jul-24   1,944.7     2,124.6     354    ₹63,688       +9.3%       +4.8%     
  CROMPTON    29     CON DUR    01-Jul-24   409.9       447.1       1680   ₹62,397       +9.1%       +4.0%     
  ESCORTS     59     MFG        03-Jun-24   3,820.6     4,093.5     170    ₹46,399       +7.1%       +1.4%     
  ANANDRATHI  40     FIN SVC    01-Apr-24   894.4       921.5       677    ₹18,355       +3.0%       -2.8%     
  COROMANDEL  26     MFG        01-Jul-24   1,574.0     1,615.9     437    ₹18,324       +2.7%       +1.7%     
  MOTHERSON   10     AUTO       01-Jul-24   129.4       129.1       5325   ₹-1,576       -0.2%       +1.1%     
  BIKAJI      67     FMCG       01-Jul-24   716.9       713.9       961    ₹-2,975       -0.4%       +0.9%     
  CHOLAHLDNG  38     FIN SVC    01-Jul-24   1,607.6     1,545.9     428    ₹-26,401      -3.8%       +4.7%     
  HINDZINC    37     METAL      03-Jun-24   646.6       601.3       1006   ₹-45,585      -7.0%       +0.5%     
  UNOMINDA    31     AUTO       01-Jul-24   1,137.1     1,034.2     605    ₹-62,274      -9.1%       +0.1%     
  BDL         42     DEFENCE    03-Jun-24   1,583.6     1,438.5     410    ₹-59,477      -9.2%       -3.8%     

  AFTER: Invested ₹20,089,476 | Cash ₹444,736 | Total ₹20,534,212 | Positions 28/30 | Slot ₹684,799

========================================================================
  REBALANCE #117  —  02 Sep 2024
  NAV: ₹21,299,366  |  Slot: ₹709,979  |  Cash: ₹444,736
========================================================================

  EXITS (12)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  POWERINDIA  110    ENERGY     02-May-24   9,777.6     11,876.3    64     ₹134,316      +21.5%    123d  
  EMAMILTD    130    FMCG       01-Jul-24   693.5       780.2       993    ₹86,049       +12.5%    63d   
  CROMPTON    137    CON DUR    01-Jul-24   409.9       460.7       1680   ₹85,377       +12.4%    63d   
  ANANDRATHI  86     FIN SVC    01-Apr-24   894.4       958.6       677    ₹43,496       +7.2%     154d  
  FSL         21     IT         01-Aug-24   291.3       304.9       2350   ₹31,958       +4.7%     32d   
  UNOMINDA    57     AUTO       01-Jul-24   1,137.1     1,163.0     605    ₹15,659       +2.3%     63d   
  VGUARD      131    CON DUR    01-Aug-24   459.3       451.3       1491   ₹-11,882      -1.7%     32d   
  ESCORTS     334    MFG        03-Jun-24   3,820.6     3,732.5     170    ₹-14,965      -2.3%     91d   
  SUNTV       161    MEDIA      01-Aug-24   852.7       781.2       803    ₹-57,381      -8.4%     32d   
  ZYDUSLIFE   159    HEALTH     01-Aug-24   1,227.3     1,099.0     557    ₹-71,507      -10.5%    32d   
  BDL         273    DEFENCE    03-Jun-24   1,583.6     1,293.3     410    ₹-119,015     -18.3%    91d   
  HINDZINC    408    METAL      03-Jun-24   646.6       475.5       1006   ₹-172,131     -26.5%    91d   

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  PCBL        1      MFG        3.781    0.95   +222.5%   +107.8%   472.6       1502   ₹709,887      +12.3%    
  ERIS        7      HEALTH     2.779    0.41   +67.9%    +55.1%    1,405.2     505    ₹709,620      +13.6%    
  GODFRYPHLP  9      FMCG       2.754    0.91   +207.6%   +73.7%    2,188.0     324    ₹708,928      +24.1%    
  ICICIGI     11     FIN SVC    2.583    0.63   +65.6%    +38.8%    2,156.1     329    ₹709,348      +5.9%     
  HCLTECH     13     IT         2.452    0.72   +59.2%    +37.5%    1,684.5     421    ₹709,155      +7.5%     
  INFY        14     IT         2.418    0.77   +44.4%    +39.6%    1,846.6     384    ₹709,092      +5.0%     
  LUPIN       15     HEALTH     3.059    0.42   +106.4%   +41.7%    2,218.9     319    ₹707,845      +6.4%     
  SUNPHARMA   18     HEALTH     2.427    0.47   +61.1%    +24.8%    1,787.5     397    ₹709,652      +3.2%     
  DEEPAKFERT  19     MFG        2.646    0.98   +96.9%    +89.1%    1,055.8     672    ₹709,490      +6.5%     
  SBILIFE     21     FIN SVC    2.192    0.80   +48.0%    +36.2%    1,882.5     377    ₹709,719      +6.4%     
  VOLTAS      22     CON DUR    2.291    0.89   +119.8%   +30.6%    1,754.3     404    ₹708,743      +7.0%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       1      CONSUMP    01-Nov-23   2,192.1     7,133.5     219    ₹1,082,166    +225.4%     +7.8%     
  BAJAJ-AUTO  7      AUTO       01-Dec-23   5,767.9     10,700.5    94     ₹463,665      +85.5%      +8.8%     
  BIKAJI      44     FMCG       01-Jul-24   716.9       839.9       961    ₹118,198      +17.2%      +2.4%     
  WHIRLPOOL   47     CON DUR    01-Jul-24   1,944.7     2,212.0     354    ₹94,615       +13.7%      +6.1%     
  360ONE      39     FIN SVC    01-Jul-24   954.2       1,068.2     722    ₹82,304       +11.9%      +1.8%     
  COROMANDEL  49     MFG        01-Jul-24   1,574.0     1,724.4     437    ₹65,720       +9.6%       +1.5%     
  COHANCE     5      MNC        01-Aug-24   990.3       1,082.0     691    ₹63,365       +9.3%       +5.9%     
  PERSISTENT  12     IT         01-Aug-24   4,750.7     5,157.2     144    ₹58,527       +8.6%       +6.1%     
  SUZLON      6      ENERGY     01-Aug-24   68.0        73.8        10073  ₹58,726       +8.6%       -1.3%     
  GRANULES    4      HEALTH     01-Aug-24   629.6       681.8       1087   ₹56,731       +8.3%       +2.4%     
  TVSMOTOR    20     AUTO       01-Aug-24   2,564.6     2,769.5     267    ₹54,717       +8.0%       +4.5%     
  COLPAL      13     FMCG       01-Aug-24   3,238.2     3,483.3     211    ₹51,707       +7.6%       +3.4%     
  ZOMATO      25     CONSUMP    01-Aug-24   234.1       244.4       2925   ₹30,303       +4.4%       -3.0%     
  CHOLAHLDNG  30     FIN SVC    01-Jul-24   1,607.6     1,668.7     428    ₹26,142       +3.8%       +4.4%     
  OFSS        23     IT         01-Aug-24   10,120.6    10,145.5    67     ₹1,669        +0.2%       +0.9%     
  MOTHERSON   52     AUTO       01-Jul-24   129.4       127.6       5325   ₹-9,353       -1.4%       +0.5%     

  AFTER: Invested ₹20,722,492 | Cash ₹567,610 | Total ₹21,290,103 | Positions 27/30 | Slot ₹709,979

========================================================================
  REBALANCE #118  —  01 Oct 2024
  NAV: ₹22,167,515  |  Slot: ₹738,917  |  Cash: ₹567,610
========================================================================

  EXITS (11)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  WHIRLPOOL   138    CON DUR    01-Jul-24   1,944.7     2,275.6     354    ₹117,131      +17.0%    92d   
  COROMANDEL  119    MFG        01-Jul-24   1,574.0     1,709.2     437    ₹59,085       +8.6%     92d   
  MOTHERSON   88     AUTO       01-Jul-24   129.4       139.2       5325   ₹52,307       +7.6%     92d   
  OFSS        59     IT         01-Aug-24   10,120.6    10,613.9    67     ₹33,047       +4.9%     61d   
  360ONE      126    FIN SVC    01-Jul-24   954.2       997.0       722    ₹30,876       +4.5%     92d   
  DEEPAKFERT  47     MFG        02-Sep-24   1,055.8     1,102.5     672    ₹31,393       +4.4%     29d   
  ICICIGI     65     FIN SVC    02-Sep-24   2,156.1     2,124.8     329    ₹-10,304      -1.5%     29d   
  SBILIFE     77     FIN SVC    02-Sep-24   1,882.5     1,828.2     377    ₹-20,498      -2.9%     29d   
  INFY        102    IT         02-Sep-24   1,846.6     1,790.1     384    ₹-21,711      -3.1%     29d   
  ERIS        55     HEALTH     02-Sep-24   1,405.2     1,309.7     505    ₹-48,199      -6.8%     29d   
  GRANULES    116    HEALTH     01-Aug-24   629.6       577.4       1087   ₹-56,731      -8.3%     61d   

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HSCL        5      MFG        3.419    0.83   +182.0%   +66.4%    665.9       1109   ₹738,431      +8.5%     
  AJANTPHARM  8      HEALTH     2.751    0.19   +93.5%    +45.3%    3,188.6     231    ₹736,557      +1.6%     
  JSWDULUX    10     CON DUR    2.583    0.50   +60.3%    +38.4%    3,609.6     204    ₹736,352      +6.9%     
  JUBLPHARMA  11     HEALTH     2.834    0.99   +168.9%   +59.7%    1,165.3     634    ₹738,813      +4.4%     
  KALYANKJIL  13     CON DUR    3.671    0.54   +243.1%   +51.6%    748.0       987    ₹738,238      +6.8%     
  NEWGEN      14     IT         2.468    1.06   +209.1%   +33.8%    1,312.3     563    ₹738,809      +6.1%     
  BALRAMCHIN  15     FMCG       2.482    1.09   +55.5%    +56.2%    671.5       1100   ₹738,653      +13.2%    
  BSE         16     FIN SVC    2.621    0.76   +222.9%   +55.2%    1,282.3     576    ₹738,579      +11.1%    
  BLUESTARCO  17     CON DUR    2.473    0.52   +132.0%   +31.1%    2,085.8     354    ₹738,384      +7.9%     
  TORNTPHARM  20     HEALTH     2.379    0.37   +80.7%    +19.5%    3,308.2     223    ₹737,721      -1.3%     
  BRITANNIA   22     FMCG       2.196    0.26   +44.2%    +20.9%    6,362.7     116    ₹738,069      +5.0%     

  HOLDS (16)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       2      CONSUMP    01-Nov-23   2,192.1     7,597.1     219    ₹1,183,683    +246.6%     +2.9%     
  BAJAJ-AUTO  4      AUTO       01-Dec-23   5,767.9     11,692.4    94     ₹556,903      +102.7%     +2.8%     
  BIKAJI      38     FMCG       01-Jul-24   716.9       943.9       961    ₹218,065      +31.7%      +3.5%     
  CHOLAHLDNG  50     FIN SVC    01-Jul-24   1,607.6     2,098.7     428    ₹210,184      +30.5%      +8.6%     
  COHANCE     17     MNC        01-Aug-24   990.3       1,202.7     691    ₹146,734      +21.4%      +3.2%     
  SUZLON      11     ENERGY     01-Aug-24   68.0        79.7        10073  ₹118,458      +17.3%      -0.7%     
  PCBL        1      MFG        02-Sep-24   472.6       553.3       1502   ₹121,200      +17.1%      +10.0%    
  ZOMATO      18     CONSUMP    01-Aug-24   234.1       274.1       2925   ₹117,175      +17.1%      -0.2%     
  PERSISTENT  53     IT         01-Aug-24   4,750.7     5,435.4     144    ₹98,598       +14.4%      +3.4%     
  COLPAL      9      FMCG       01-Aug-24   3,238.2     3,666.2     211    ₹90,303       +13.2%      +3.9%     
  TVSMOTOR    30     AUTO       01-Aug-24   2,564.6     2,817.1     267    ₹67,422       +9.8%       +0.7%     
  SUNPHARMA   12     HEALTH     02-Sep-24   1,787.5     1,889.9     397    ₹40,642       +5.7%       +2.9%     
  VOLTAS      27     CON DUR    02-Sep-24   1,754.3     1,838.5     404    ₹34,002       +4.8%       +0.4%     
  GODFRYPHLP  14     FMCG       02-Sep-24   2,188.0     2,234.8     324    ₹15,141       +2.1%       -1.9%     
  HCLTECH     56     IT         02-Sep-24   1,684.5     1,693.6     421    ₹3,866        +0.5%       +2.4%     
  LUPIN       13     HEALTH     02-Sep-24   2,218.9     2,180.8     319    ₹-12,158      -1.7%       -0.1%     

  AFTER: Invested ₹21,888,221 | Cash ₹269,655 | Total ₹22,157,875 | Positions 27/30 | Slot ₹738,917

========================================================================
  REBALANCE #119  —  01 Nov 2024
  NAV: ₹20,725,254  |  Slot: ₹690,842  |  Cash: ₹269,655
========================================================================

  EXITS (13)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  BAJAJ-AUTO  73     AUTO       01-Dec-23   5,767.9     9,498.2     94     ₹350,646      +64.7%    336d  
  BIKAJI      56     FMCG       01-Jul-24   716.9       855.9       961    ₹133,555      +19.4%    123d  
  CHOLAHLDNG  93     FIN SVC    01-Jul-24   1,607.6     1,744.1     428    ₹58,434       +8.5%     123d  
  ZOMATO      85     CONSUMP    01-Aug-24   234.1       249.0       2925   ₹43,583       +6.4%     92d   
  SUZLON      134    ENERGY     01-Aug-24   68.0        68.1        10073  ₹1,612        +0.2%     92d   
  HCLTECH     68     IT         02-Sep-24   1,684.5     1,649.3     421    ₹-14,812      -2.1%     60d   
  TVSMOTOR    144    AUTO       01-Aug-24   2,564.6     2,490.6     267    ₹-19,740      -2.9%     92d   
  TORNTPHARM  83     HEALTH     01-Oct-24   3,308.2     3,146.6     223    ₹-36,033      -4.9%     31d   
  AJANTPHARM  62     HEALTH     01-Oct-24   3,188.6     3,024.5     231    ₹-37,908      -5.1%     31d   
  VOLTAS      46     CON DUR    02-Sep-24   1,754.3     1,628.6     404    ₹-50,783      -7.2%     60d   
  COLPAL      222    FMCG       01-Aug-24   3,238.2     2,942.4     211    ₹-62,409      -9.1%     92d   
  PCBL        51     MFG        02-Sep-24   472.6       419.8       1502   ₹-79,369      -11.2%    60d   
  BRITANNIA   176    FMCG       01-Oct-24   6,362.7     5,619.4     116    ₹-86,218      -11.7%    31d   

  ENTRIES (13)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  POLYMED     2      HEALTH     4.283    0.50   +140.2%   +69.4%    3,123.4     221    ₹690,275      +20.7%    
  BASF        4      MFG        3.531    0.87   +235.4%   +38.3%    8,277.4     83     ₹687,024      +8.6%     
  RADICO      7      FMCG       3.150    0.62   +102.7%   +40.8%    2,398.9     287    ₹688,490      +7.2%     
  FORTIS      10     HEALTH     2.805    0.52   +95.9%    +25.6%    634.6       1088   ₹690,444      +5.1%     
  UTIAMC      11     FIN SVC    2.761    0.56   +87.4%    +31.4%    1,316.0     524    ₹689,570      +8.4%     
  GLENMARK    12     HEALTH     2.764    0.84   +122.8%   +17.3%    1,686.2     409    ₹689,641      -0.7%     
  NETWEB      13     IT         2.695    0.94   +224.3%   +15.9%    2,713.3     254    ₹689,169      +4.1%     
  KIMS        14     HEALTH     2.626    0.36   +44.4%    +27.3%    541.5       1275   ₹690,349      +1.0%     
  POWERINDIA  15     ENERGY     2.615    1.19   +222.8%   +15.7%    13,919.7    49     ₹682,066      -1.7%     
  CDSL        16     FIN SVC    2.508    1.03   +140.6%   +27.0%    1,545.4     447    ₹690,794      +4.1%     
  DIVISLAB    17     HEALTH     2.469    0.54   +69.7%    +18.3%    5,876.8     117    ₹687,584      +1.6%     
  AMBER       18     CON DUR    2.408    0.89   +120.5%   +40.4%    6,161.0     112    ₹690,026      +7.2%     
  MANKIND     20     HEALTH     2.393    0.42   +53.2%    +34.0%    2,679.6     257    ₹688,668      +3.3%     

  HOLDS (14)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       2      CONSUMP    01-Nov-23   2,192.1     7,134.3     219    ₹1,082,341    +225.5%     -4.5%     
  COHANCE     12     MNC        01-Aug-24   990.3       1,327.2     691    ₹232,763      +34.0%      +6.1%     
  BSE         11     FIN SVC    01-Oct-24   1,282.3     1,483.2     576    ₹115,747      +15.7%      +5.7%     
  JSWDULUX    8      CON DUR    01-Oct-24   3,609.6     4,145.2     204    ₹109,265      +14.8%      +14.6%    
  PERSISTENT  44     IT         01-Aug-24   4,750.7     5,337.6     144    ₹84,514       +12.4%      -1.7%     
  JUBLPHARMA  3      HEALTH     01-Oct-24   1,165.3     1,257.2     634    ₹58,269       +7.9%       +9.6%     
  SUNPHARMA   29     HEALTH     02-Sep-24   1,787.5     1,829.3     397    ₹16,589       +2.3%       -0.9%     
  GODFRYPHLP  7      FMCG       02-Sep-24   2,188.0     2,195.7     324    ₹2,479        +0.3%       +0.8%     
  LUPIN       33     HEALTH     02-Sep-24   2,218.9     2,184.1     319    ₹-11,112      -1.6%       +0.8%     
  NEWGEN      25     IT         01-Oct-24   1,312.3     1,285.2     563    ₹-15,267      -2.1%       +2.8%     
  BALRAMCHIN  36     FMCG       01-Oct-24   671.5       620.1       1100   ₹-56,494      -7.6%       -0.3%     
  BLUESTARCO  40     CON DUR    01-Oct-24   2,085.8     1,907.9     354    ₹-62,975      -8.5%       -0.1%     
  KALYANKJIL  34     CON DUR    01-Oct-24   748.0       668.8       987    ₹-78,090      -10.6%      -3.1%     
  HSCL        26     MFG        01-Oct-24   665.9       574.8       1109   ₹-100,987     -13.7%      -2.8%     

  AFTER: Invested ₹20,212,306 | Cash ₹502,315 | Total ₹20,714,622 | Positions 27/30 | Slot ₹690,842

========================================================================
  REBALANCE #120  —  02 Dec 2024
  NAV: ₹19,985,357  |  Slot: ₹666,179  |  Cash: ₹502,315
========================================================================

  EXITS (10)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  SUNPHARMA   73     HEALTH     02-Sep-24   1,787.5     1,780.3     397    ₹-2,892       -0.4%     91d   
  MANKIND     127    HEALTH     01-Nov-24   2,679.6     2,615.4     257    ₹-16,506      -2.4%     31d   
  JSWDULUX    92     CON DUR    01-Oct-24   3,609.6     3,399.3     204    ₹-42,898      -5.8%     62d   
  LUPIN       117    HEALTH     02-Sep-24   2,218.9     2,056.8     319    ₹-51,739      -7.3%     91d   
  GLENMARK    94     HEALTH     01-Nov-24   1,686.2     1,543.8     409    ₹-58,242      -8.4%     31d   
  NEWGEN      103    IT         01-Oct-24   1,312.3     1,170.9     563    ₹-79,612      -10.8%    62d   
  POWERINDIA  45     ENERGY     01-Nov-24   13,919.7    12,260.7    49     ₹-81,289      -11.9%    31d   
  BALRAMCHIN  190    FMCG       01-Oct-24   671.5       587.4       1100   ₹-92,513      -12.5%    62d   
  HSCL        80     MFG        01-Oct-24   665.9       532.9       1109   ₹-147,443     -20.0%    62d   
  BASF        215    MFG        01-Nov-24   8,277.4     5,636.0     83     ₹-219,236     -31.9%    31d   

  ENTRIES (10)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  OFSS        3      IT         3.033    1.11   +204.6%   +11.6%    11,378.1    58     ₹659,928      +5.8%     
  COFORGE     7      IT         2.923    0.76   +56.8%    +37.7%    1,722.2     386    ₹664,765      +5.8%     
  INDHOTEL    8      CONSUMP    2.939    1.13   +90.9%    +23.7%    795.2       837    ₹665,549      +6.1%     
  ZOMATO      10     CONSUMP    2.845    0.76   +142.5%   +12.8%    282.5       2358   ₹666,135      +4.5%     
  DEEPAKFERT  12     MFG        2.783    0.87   +129.8%   +31.8%    1,358.5     490    ₹665,675      +6.1%     
  ABSLAMC     14     FIN SVC    2.560    1.17   +98.9%    +17.0%    852.0       781    ₹665,403      +5.6%     
  FSL         15     IT         2.528    1.15   +125.3%   +20.7%    354.2       1881   ₹666,169      +3.5%     
  VIJAYA      17     HEALTH     2.489    0.10   +85.3%    +26.0%    1,147.7     580    ₹665,688      +5.0%     
  CONCORDBIO  19     HEALTH     2.462    0.62   +72.1%    +31.1%    2,170.4     306    ₹664,145      +9.8%     
  POLICYBZR   20     FIN SVC    2.460    0.75   +138.1%   +9.8%     1,945.1     342    ₹665,224      +9.4%     

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       25     CONSUMP    01-Nov-23   2,192.1     6,791.3     219    ₹1,007,225    +209.8%     +0.3%     
  COHANCE     7      MNC        01-Aug-24   990.3       1,309.1     691    ₹220,291      +32.2%      +2.3%     
  PERSISTENT  31     IT         01-Aug-24   4,750.7     5,875.8     144    ₹162,017      +23.7%      +3.1%     
  BSE         5      FIN SVC    01-Oct-24   1,282.3     1,516.5     576    ₹134,925      +18.3%      +0.2%     
  KIMS        14     HEALTH     01-Nov-24   541.5       602.5       1275   ₹77,775       +11.3%      +5.0%     
  CDSL        43     FIN SVC    01-Nov-24   1,545.4     1,651.6     447    ₹47,464       +6.9%       +6.6%     
  FORTIS      13     HEALTH     01-Nov-24   634.6       676.2       1088   ₹45,207       +6.5%       +4.4%     
  DIVISLAB    3      HEALTH     01-Nov-24   5,876.8     6,227.0     117    ₹40,974       +6.0%       +4.7%     
  JUBLPHARMA  2      HEALTH     01-Oct-24   1,165.3     1,197.1     634    ₹20,170       +2.7%       +2.5%     
  NETWEB      10     IT         01-Nov-24   2,713.3     2,784.8     254    ₹18,177       +2.6%       +0.8%     
  RADICO      18     FMCG       01-Nov-24   2,398.9     2,415.9     287    ₹4,886        +0.7%       +3.6%     
  AMBER       29     CON DUR    01-Nov-24   6,161.0     6,051.0     112    ₹-12,309      -1.8%       -1.7%     
  KALYANKJIL  27     CON DUR    01-Oct-24   748.0       720.0       987    ₹-27,605      -3.7%       +3.3%     
  UTIAMC      42     FIN SVC    01-Nov-24   1,316.0     1,256.8     524    ₹-30,996      -4.5%       +0.5%     
  BLUESTARCO  54     CON DUR    01-Oct-24   2,085.8     1,833.5     354    ₹-89,310      -12.1%      +0.1%     
  POLYMED     44     HEALTH     01-Nov-24   3,123.4     2,739.8     221    ₹-84,770      -12.3%      +1.2%     
  GODFRYPHLP  72     FMCG       02-Sep-24   2,188.0     1,898.0     324    ₹-93,973      -13.3%      -5.1%     

  AFTER: Invested ₹19,806,950 | Cash ₹170,512 | Total ₹19,977,462 | Positions 27/30 | Slot ₹666,179

========================================================================
  REBALANCE #121  —  01 Jan 2025
  NAV: ₹20,603,957  |  Slot: ₹686,799  |  Cash: ₹170,512
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  COHANCE     105    MNC        01-Aug-24   990.3       1,122.5     691    ₹91,350       +13.3%    153d  
  CONCORDBIO  96     HEALTH     02-Dec-24   2,170.4     2,135.1     306    ₹-10,794      -1.6%     30d   
  ABSLAMC     31     FIN SVC    02-Dec-24   852.0       813.9       781    ₹-29,758      -4.5%     30d   
  VIJAYA      71     HEALTH     02-Dec-24   1,147.7     1,060.6     580    ₹-50,532      -7.6%     30d   
  JUBLPHARMA  78     HEALTH     01-Oct-24   1,165.3     1,075.3     634    ₹-57,070      -7.7%     92d   
  DEEPAKFERT  63     MFG        02-Dec-24   1,358.5     1,209.8     490    ₹-72,880      -10.9%    30d   
  GODFRYPHLP  134    FMCG       02-Sep-24   2,188.0     1,660.0     324    ₹-171,089     -24.1%    121d  

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  KFINTECH    1      FIN SVC    3.894    1.13   +220.2%   +51.3%    1,534.9     447    ₹686,089      +11.2%    
  CRISIL      8      MNC        2.805    0.07   +51.8%    +37.8%    6,267.9     109    ₹683,200      +10.8%    
  LAURUSLABS  10     HEALTH     2.784    1.03   +55.3%    +32.9%    613.7       1119   ₹686,722      +7.3%     
  360ONE      12     FIN SVC    2.669    0.75   +103.7%   +22.8%    1,255.2     547    ₹686,598      +4.9%     
  NEWGEN      13     IT         2.668    0.98   +143.2%   +26.7%    1,691.4     406    ₹686,714      +11.6%    
  CAPLIPOINT  14     HEALTH     2.646    0.79   +93.7%    +32.4%    2,500.5     274    ₹685,136      +4.7%     

  HOLDS (20)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       35     CONSUMP    01-Nov-23   2,192.1     7,053.5     219    ₹1,064,649    +221.8%     +1.1%     
  BSE         12     FIN SVC    01-Oct-24   1,282.3     1,803.0     576    ₹299,937      +40.6%      +1.7%     
  PERSISTENT  24     IT         01-Aug-24   4,750.7     6,375.4     144    ₹233,958      +34.2%      +1.4%     
  AMBER       4      CON DUR    01-Nov-24   6,161.0     7,671.7     112    ₹169,204      +24.5%      +14.2%    
  CDSL        23     FIN SVC    01-Nov-24   1,545.4     1,794.0     447    ₹111,134      +16.1%      +0.2%     
  FORTIS      27     HEALTH     01-Nov-24   634.6       707.7       1088   ₹79,493       +11.5%      +2.4%     
  KIMS        37     HEALTH     01-Nov-24   541.5       603.3       1275   ₹78,922       +11.4%      +1.3%     
  COFORGE     9      IT         02-Dec-24   1,722.2     1,903.7     386    ₹70,047       +10.5%      +3.7%     
  INDHOTEL    5      CONSUMP    02-Dec-24   795.2       867.2       837    ₹60,278       +9.1%       +2.7%     
  POLICYBZR   3      FIN SVC    02-Dec-24   1,945.1     2,119.2     342    ₹59,542       +9.0%       +2.9%     
  BLUESTARCO  11     CON DUR    01-Oct-24   2,085.8     2,249.0     354    ₹57,761       +7.8%       +10.1%    
  RADICO      14     FMCG       01-Nov-24   2,398.9     2,578.4     287    ₹51,499       +7.5%       +3.6%     
  NETWEB      20     IT         01-Nov-24   2,713.3     2,886.5     254    ₹43,990       +6.4%       +3.4%     
  FSL         25     IT         02-Dec-24   354.2       371.6       1881   ₹32,862       +4.9%       +4.1%     
  KALYANKJIL  30     CON DUR    01-Oct-24   748.0       773.6       987    ₹25,292       +3.4%       +4.9%     
  OFSS        8      IT         02-Dec-24   11,378.1    11,715.6    58     ₹19,578       +3.0%       +1.8%     
  DIVISLAB    22     HEALTH     01-Nov-24   5,876.8     6,045.5     117    ₹19,736       +2.9%       +2.1%     
  UTIAMC      51     FIN SVC    01-Nov-24   1,316.0     1,321.2     524    ₹2,737        +0.4%       +4.4%     
  ZOMATO      36     CONSUMP    02-Dec-24   282.5       276.5       2358   ₹-14,148      -2.1%       -1.5%     
  POLYMED     38     HEALTH     01-Nov-24   3,123.4     2,675.2     221    ₹-99,055      -14.4%      +0.1%     

  AFTER: Invested ₹20,055,728 | Cash ₹543,343 | Total ₹20,599,072 | Positions 26/30 | Slot ₹686,799

========================================================================
  REBALANCE #122  —  01 Feb 2025
  NAV: ₹17,479,693  |  Slot: ₹582,656  |  Cash: ₹543,343
========================================================================

  [REGIME OFF] Nifty 500 21,580.9 < SMA200 22,498.0 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (26)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       50     CONSUMP    01-Nov-23   2,192.1     6,176.8     219    ₹872,641      +181.8%     +2.9%     
  BSE         4      FIN SVC    01-Oct-24   1,282.3     1,795.7     576    ₹295,736      +40.0%      -1.5%     
  PERSISTENT  36     IT         01-Aug-24   4,750.7     5,896.4     144    ₹164,985      +24.1%      -2.5%     
  KIMS        6      HEALTH     01-Nov-24   541.5       617.2       1275   ₹96,518       +14.0%      +0.9%     
  AMBER       47     CON DUR    01-Nov-24   6,161.0     6,617.1     112    ₹51,089       +7.4%       -2.3%     
  INDHOTEL    8      CONSUMP    02-Dec-24   795.2       795.6       837    ₹332          +0.0%       +1.3%     
  RADICO      61     FMCG       01-Nov-24   2,398.9     2,376.8     287    ₹-6,362       -0.9%       +4.9%     
  FORTIS      62     HEALTH     01-Nov-24   634.6       626.1       1088   ₹-9,237       -1.3%       -2.9%     
  BLUESTARCO  15     CON DUR    01-Oct-24   2,085.8     2,046.8     354    ₹-13,811      -1.9%       +7.3%     
  LAURUSLABS  10     HEALTH     01-Jan-25   613.7       593.7       1119   ₹-22,365      -3.3%       +3.8%     
  DIVISLAB    39     HEALTH     01-Nov-24   5,876.8     5,593.1     117    ₹-33,194      -4.8%       -2.4%     
  FSL         38     IT         02-Dec-24   354.2       332.9       1881   ₹-40,054      -6.0%       -4.2%     
  COFORGE     52     IT         02-Dec-24   1,722.2     1,600.2     386    ₹-47,069      -7.1%       -7.2%     
  POLICYBZR   28     FIN SVC    02-Dec-24   1,945.1     1,716.2     342    ₹-78,301      -11.8%      -3.0%     
  CRISIL      60     MNC        01-Jan-25   6,267.9     5,333.1     109    ₹-101,889     -14.9%      -0.6%     
  CDSL        154    FIN SVC    01-Nov-24   1,545.4     1,296.3     447    ₹-111,334     -16.1%      -11.5%    
  ZOMATO      —      OTHER      02-Dec-24   282.5       236.3       2358   ₹-108,892     -16.3%      +1.5%     
  CAPLIPOINT  71     HEALTH     01-Jan-25   2,500.5     2,073.0     274    ₹-117,144     -17.1%      -4.0%     
  360ONE      56     FIN SVC    01-Jan-25   1,255.2     1,006.4     547    ₹-136,088     -19.8%      -7.8%     
  POLYMED     124    HEALTH     01-Nov-24   3,123.4     2,429.6     221    ₹-153,326     -22.2%      -1.2%     
  UTIAMC      290    FIN SVC    01-Nov-24   1,316.0     1,019.9     524    ₹-155,158     -22.5% ⚠    -8.9%     
  OFSS        207    IT         02-Dec-24   11,378.1    8,249.1     58     ₹-181,478     -27.5% ⚠    -11.8%    
  KFINTECH    25     FIN SVC    01-Jan-25   1,534.9     1,094.5     447    ₹-196,859     -28.7%      -6.9%     
  KALYANKJIL  326    CON DUR    01-Oct-24   748.0       504.0       987    ₹-240,814     -32.6% ⚠    -5.2%     
  NETWEB      369    IT         01-Nov-24   2,713.3     1,785.2     254    ₹-235,718     -34.2% ⚠    -13.5%    
  NEWGEN      205    IT         01-Jan-25   1,691.4     1,090.0     406    ₹-244,170     -35.6% ⚠    -14.9%    
  ⚠  WAZ < 0 (momentum below universe mean): OFSS, NEWGEN, UTIAMC, NETWEB, KALYANKJIL

  AFTER: Invested ₹16,936,349 | Cash ₹543,343 | Total ₹17,479,693 | Positions 26/30 | Slot ₹582,656

========================================================================
  REBALANCE #123  —  03 Mar 2025
  NAV: ₹15,445,202  |  Slot: ₹514,840  |  Cash: ₹543,343
========================================================================

  [REGIME OFF] Nifty 500 19,896.9 < SMA200 22,517.6 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (26)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       172    CONSUMP    01-Nov-23   2,192.1     4,936.8     219    ₹601,070      +125.2%     -4.9%     
  BSE         16     FIN SVC    01-Oct-24   1,282.3     1,448.6     576    ₹95,803       +13.0%      -17.7%    
  PERSISTENT  76     IT         01-Aug-24   4,750.7     5,259.5     144    ₹73,258       +10.7%      -7.3%     
  FORTIS      43     HEALTH     01-Nov-24   634.6       628.4       1088   ₹-6,792       -1.0%       +1.4%     
  FSL         13     IT         02-Dec-24   354.2       344.6       1881   ₹-18,015      -2.7%       -0.2%     
  BLUESTARCO  8      CON DUR    01-Oct-24   2,085.8     2,022.8     354    ₹-22,319      -3.0%       +4.9%     
  KIMS        144    HEALTH     01-Nov-24   541.5       518.6       1275   ₹-29,134      -4.2%       -7.5%     
  DIVISLAB    27     HEALTH     01-Nov-24   5,876.8     5,515.1     117    ₹-42,313      -6.2%       -4.2%     
  AMBER       42     CON DUR    01-Nov-24   6,161.0     5,657.0     112    ₹-56,442      -8.2%       -7.0%     
  INDHOTEL    71     CONSUMP    02-Dec-24   795.2       721.4       837    ₹-61,732      -9.3%       -2.7%     
  RADICO      75     FMCG       01-Nov-24   2,398.9     2,062.4     287    ₹-96,593      -14.0%      -2.5%     
  LAURUSLABS  52     HEALTH     01-Jan-25   613.7       527.1       1119   ₹-96,876      -14.1%      -5.3%     
  COFORGE     134    IT         02-Dec-24   1,722.2     1,457.7     386    ₹-102,083     -15.4%      -6.4%     
  ZOMATO      —      OTHER      02-Dec-24   282.5       222.1       2358   ₹-142,329     -21.4%      -1.8%     
  360ONE      66     FIN SVC    01-Jan-25   1,255.2     962.1       547    ₹-160,316     -23.3%      -1.7%     
  POLICYBZR   86     FIN SVC    02-Dec-24   1,945.1     1,451.8     342    ₹-168,692     -25.4%      -7.6%     
  CDSL        201    FIN SVC    01-Nov-24   1,545.4     1,095.3     447    ₹-201,208     -29.1%      -10.9%    
  CRISIL      207    MNC        01-Jan-25   6,267.9     4,351.5     109    ₹-208,889     -30.6% ⚠    -8.3%     
  UTIAMC      243    FIN SVC    01-Nov-24   1,316.0     910.2       524    ₹-212,639     -30.8% ⚠    -5.6%     
  CAPLIPOINT  163    HEALTH     01-Jan-25   2,500.5     1,714.9     274    ₹-215,252     -31.4%      -13.4%    
  POLYMED     148    HEALTH     01-Nov-24   3,123.4     2,041.7     221    ₹-239,056     -34.6%      -7.4%     
  OFSS        348    IT         02-Dec-24   11,378.1    7,269.2     58     ₹-238,314     -36.1% ⚠    -10.1%    
  KALYANKJIL  362    CON DUR    01-Oct-24   748.0       439.2       987    ₹-304,782     -41.3% ⚠    -10.8%    
  KFINTECH    151    FIN SVC    01-Jan-25   1,534.9     866.5       447    ₹-298,764     -43.5%      -8.9%     
  NEWGEN      124    IT         01-Jan-25   1,691.4     940.5       406    ₹-304,854     -44.4%      -7.7%     
  NETWEB      378    IT         01-Nov-24   2,713.3     1,417.2     254    ₹-329,192     -47.8% ⚠    -11.0%    
  ⚠  WAZ < 0 (momentum below universe mean): CRISIL, UTIAMC, OFSS, NETWEB, KALYANKJIL

  AFTER: Invested ₹14,901,859 | Cash ₹543,343 | Total ₹15,445,202 | Positions 26/30 | Slot ₹514,840

========================================================================
  REBALANCE #124  —  01 Apr 2025
  NAV: ₹16,558,979  |  Slot: ₹551,966  |  Cash: ₹543,343
========================================================================

  [REGIME OFF] Nifty 500 21,070.8 < SMA200 22,473.0 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (26)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       218    CONSUMP    01-Nov-23   2,192.1     5,565.3     219    ₹738,724      +153.9%     +6.8%     
  BSE         25     FIN SVC    01-Oct-24   1,282.3     1,816.3     576    ₹307,593      +41.6%      +15.8%    
  KIMS        33     HEALTH     01-Nov-24   541.5       618.2       1275   ₹97,856       +14.2%      +4.1%     
  AMBER       45     CON DUR    01-Nov-24   6,161.0     6,879.5     112    ₹80,483       +11.7%      +2.8%     
  PERSISTENT  180    IT         01-Aug-24   4,750.7     5,179.0     144    ₹61,667       +9.0% ⚠     -3.6%     
  FORTIS      55     HEALTH     01-Nov-24   634.6       687.8       1088   ₹57,922       +8.4%       +7.1%     
  INDHOTEL    121    CONSUMP    02-Dec-24   795.2       799.8       837    ₹3,905        +0.6%       +2.7%     
  BLUESTARCO  48     CON DUR    01-Oct-24   2,085.8     2,075.4     354    ₹-3,699       -0.5%       -1.0%     
  LAURUSLABS  67     HEALTH     01-Jan-25   613.7       597.0       1119   ₹-18,684      -2.7%       +0.8%     
  RADICO      105    FMCG       01-Nov-24   2,398.9     2,323.9     287    ₹-21,537      -3.1%       +2.7%     
  DIVISLAB    42     HEALTH     01-Nov-24   5,876.8     5,524.5     117    ₹-41,213      -6.0%       -3.3%     
  FSL         68     IT         02-Dec-24   354.2       327.1       1881   ₹-50,966      -7.7%       +1.5%     
  COFORGE     155    IT         02-Dec-24   1,722.2     1,541.7     386    ₹-69,673      -10.5%      +0.1%     
  CAPLIPOINT  132    HEALTH     01-Jan-25   2,500.5     1,968.4     274    ₹-145,792     -21.3%      +0.6%     
  UTIAMC      192    FIN SVC    01-Nov-24   1,316.0     1,027.4     524    ₹-151,205     -21.9% ⚠    +4.8%     
  POLICYBZR   231    FIN SVC    02-Dec-24   1,945.1     1,514.6     342    ₹-147,231     -22.1% ⚠    -1.6%     
  CDSL        237    FIN SVC    01-Nov-24   1,545.4     1,188.1     447    ₹-159,707     -23.1%      +1.5%     
  ZOMATO      —      OTHER      02-Dec-24   282.5       202.0       2358   ₹-189,795     -28.5%      -5.3%     
  360ONE      232    FIN SVC    01-Jan-25   1,255.2     873.2       547    ₹-208,959     -30.4% ⚠    -5.4%     
  POLYMED     152    HEALTH     01-Nov-24   3,123.4     2,148.3     221    ₹-215,494     -31.2%      -3.3%     
  KFINTECH    331    FIN SVC    01-Jan-25   1,534.9     1,025.2     447    ₹-227,830     -33.2%      +2.9%     
  CRISIL      399    MNC        01-Jan-25   6,267.9     4,085.7     109    ₹-237,859     -34.8% ⚠    -4.3%     
  OFSS        380    IT         02-Dec-24   11,378.1    7,036.0     58     ₹-251,842     -38.2% ⚠    -3.4%     
  KALYANKJIL  452    CON DUR    01-Oct-24   748.0       456.7       987    ₹-287,462     -38.9% ⚠    -1.1%     
  NEWGEN      287    IT         01-Jan-25   1,691.4     987.2       406    ₹-285,925     -41.6%      +0.5%     
  NETWEB      381    IT         01-Nov-24   2,713.3     1,508.7     254    ₹-305,953     -44.4% ⚠    -1.8%     
  ⚠  WAZ < 0 (momentum below universe mean): PERSISTENT, UTIAMC, 360ONE, POLICYBZR, OFSS, CRISIL, NETWEB, KALYANKJIL

  AFTER: Invested ₹16,015,636 | Cash ₹543,343 | Total ₹16,558,979 | Positions 26/30 | Slot ₹551,966

========================================================================
  REBALANCE #125  —  02 May 2025
  NAV: ₹16,992,760  |  Slot: ₹566,425  |  Cash: ₹543,343
========================================================================

  [REGIME OFF] Nifty 500 22,006.0 < SMA200 22,380.3 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (26)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  TRENT       237    CONSUMP    01-Nov-23   2,192.1     5,143.4     219    ₹646,332      +134.6%     -0.7%     
  BSE         44     FIN SVC    01-Oct-24   1,282.3     2,099.0     576    ₹470,471      +63.7%      +5.7%     
  KIMS        15     HEALTH     01-Nov-24   541.5       662.5       1275   ₹154,275      +22.3%      +2.7%     
  PERSISTENT  116    IT         01-Aug-24   4,750.7     5,372.7     144    ₹89,564       +13.1%      +5.4%     
  FORTIS      42     HEALTH     01-Nov-24   634.6       682.6       1088   ₹52,271       +7.6%       +3.0%     
  DIVISLAB    19     HEALTH     01-Nov-24   5,876.8     6,051.9     117    ₹20,493       +3.0%       +3.4%     
  RADICO      46     FMCG       01-Nov-24   2,398.9     2,445.5     287    ₹13,355       +1.9%       +1.5%     
  LAURUSLABS  53     HEALTH     01-Jan-25   613.7       617.3       1119   ₹4,071        +0.6%       +0.5%     
  AMBER       118    CON DUR    01-Nov-24   6,161.0     6,188.0     112    ₹3,030        +0.4%       -4.6%     
  INDHOTEL    121    CONSUMP    02-Dec-24   795.2       795.5       837    ₹291          +0.0%       -0.2%     
  FSL         87     IT         02-Dec-24   354.2       329.0       1881   ₹-47,284      -7.1%       -1.0%     
  COFORGE     197    IT         02-Dec-24   1,722.2     1,461.4     386    ₹-100,672     -15.1% ⚠    +2.6%     
  CDSL        160    FIN SVC    01-Nov-24   1,545.4     1,310.5     447    ₹-104,994     -15.2%      +3.5%     
  ZOMATO      —      OTHER      02-Dec-24   282.5       234.1       2358   ₹-114,127     -17.1%      +3.7%     
  POLICYBZR   185    FIN SVC    02-Dec-24   1,945.1     1,590.0     342    ₹-121,444     -18.3% ⚠    -0.6%     
  BLUESTARCO  233    CON DUR    01-Oct-24   2,085.8     1,671.6     354    ₹-146,648     -19.9% ⚠    -11.6%    
  POLYMED     71     HEALTH     01-Nov-24   3,123.4     2,491.8     221    ₹-139,582     -20.2%      +4.3%     
  CAPLIPOINT  119    HEALTH     01-Jan-25   2,500.5     1,876.7     274    ₹-170,909     -24.9%      -0.7%     
  360ONE      243    FIN SVC    01-Jan-25   1,255.2     938.5       547    ₹-173,238     -25.2% ⚠    -0.4%     
  UTIAMC      258    FIN SVC    01-Nov-24   1,316.0     976.9       524    ₹-177,664     -25.8% ⚠    -3.8%     
  CRISIL      235    MNC        01-Jan-25   6,267.9     4,603.5     109    ₹-181,421     -26.6% ⚠    +4.2%     
  KFINTECH    174    FIN SVC    01-Jan-25   1,534.9     1,125.8     447    ₹-182,873     -26.7%      -0.8%     
  OFSS        198    IT         02-Dec-24   11,378.1    8,047.6     58     ₹-193,166     -29.3% ⚠    +5.0%     
  KALYANKJIL  280    CON DUR    01-Oct-24   748.0       505.8       987    ₹-239,043     -32.4%      +0.1%     
  NEWGEN      157    IT         01-Jan-25   1,691.4     1,025.0     406    ₹-270,552     -39.4%      +4.2%     
  NETWEB      342    IT         01-Nov-24   2,713.3     1,416.3     254    ₹-329,433     -47.8% ⚠    -4.3%     
  ⚠  WAZ < 0 (momentum below universe mean): COFORGE, POLICYBZR, OFSS, CRISIL, 360ONE, BLUESTARCO, UTIAMC, NETWEB

  AFTER: Invested ₹16,449,416 | Cash ₹543,343 | Total ₹16,992,760 | Positions 26/30 | Slot ₹566,425

========================================================================
  REBALANCE #126  —  02 Jun 2025
  NAV: ₹18,405,371  |  Slot: ₹613,512  |  Cash: ₹543,343
========================================================================

  EXITS (19)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  TRENT       253    CONSUMP    01-Nov-23   2,192.1     5,610.0     219    ₹748,504      +155.9%   579d  
  BSE         7      FIN SVC    01-Oct-24   1,282.3     2,693.3     576    ₹812,761      +110.0%   244d  
  PERSISTENT  172    IT         01-Aug-24   4,750.7     5,484.5     144    ₹105,663      +15.4%    305d  
  CDSL        32     FIN SVC    01-Nov-24   1,545.4     1,669.3     447    ₹55,379       +8.0%     213d  
  AMBER       126    CON DUR    01-Nov-24   6,161.0     6,382.0     112    ₹24,758       +3.6%     213d  
  LAURUSLABS  115    HEALTH     01-Jan-25   613.7       610.0       1119   ₹-4,145       -0.6%     152d  
  INDHOTEL    196    CONSUMP    02-Dec-24   795.2       777.8       837    ₹-14,498      -2.2%     182d  
  POLICYBZR   154    FIN SVC    02-Dec-24   1,945.1     1,757.7     342    ₹-64,091      -9.6%     182d  
  CAPLIPOINT  100    HEALTH     01-Jan-25   2,500.5     2,152.6     274    ₹-95,313      -13.9%    152d  
  ZOMATO      —      OTHER      02-Dec-24   282.5       241.2       2358   ₹-97,385      -14.6%    182d  
  CRISIL      117    MNC        01-Jan-25   6,267.9     5,121.8     109    ₹-124,927     -18.3%    152d  
  360ONE      249    FIN SVC    01-Jan-25   1,255.2     1,008.5     547    ₹-134,956     -19.7%    152d  
  KALYANKJIL  400    CON DUR    01-Oct-24   748.0       555.0       987    ₹-190,427     -25.8%    244d  
  BLUESTARCO  404    CON DUR    01-Oct-24   2,085.8     1,538.4     354    ₹-193,787     -26.2%    244d  
  KFINTECH    223    FIN SVC    01-Jan-25   1,534.9     1,116.9     447    ₹-186,824     -27.2%    152d  
  NEWGEN      112    IT         01-Jan-25   1,691.4     1,224.9     406    ₹-189,405     -27.6%    152d  
  POLYMED     224    HEALTH     01-Nov-24   3,123.4     2,242.8     221    ₹-194,624     -28.2%    213d  
  NETWEB      235    IT         01-Nov-24   2,713.3     1,946.7     254    ₹-194,696     -28.3%    213d  
  OFSS        250    IT         02-Dec-24   11,378.1    8,038.0     58     ₹-193,722     -29.4%    182d  

  ENTRIES (21)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  SOLARINDS   1      DEFENCE    4.002    1.12   +68.3%    +83.8%    16,284.3    37     ₹602,517      +10.5%    
  JSWHL       2      FIN SVC    3.875    0.72   +241.4%   +39.8%    22,430.0    27     ₹605,610      -4.3%     
  MFSL        3      FIN SVC    3.626    0.67   +58.9%    +46.9%    1,522.3     403    ₹613,487      +8.2%     
  COROMANDEL  4      MFG        3.430    0.83   +84.5%    +36.7%    2,265.8     270    ₹611,774      -2.5%     
  BDL         6      DEFENCE    3.000    1.08   +28.4%    +95.2%    1,966.8     311    ₹611,675      +9.0%     
  DEEPAKFERT  7      MFG        3.148    1.11   +169.1%   +40.1%    1,469.7     417    ₹612,855      +7.7%     
  PAYTM       8      FIN SVC    2.645    1.12   +159.1%   +22.4%    924.3       663    ₹612,844      +7.0%     
  ZENTEC      9      DEFENCE    3.515    1.08   +121.1%   +93.6%    2,114.7     290    ₹613,269      +15.6%    
  ERIS        10     HEALTH     2.544    0.44   +78.5%    +24.3%    1,557.7     393    ₹612,192      +3.2%     
  CCL         11     FMCG       2.545    0.51   +49.2%    +51.1%    882.3       695    ₹613,204      +13.9%    
  HDFCLIFE    12     FIN SVC    2.425    0.72   +36.3%    +24.3%    761.9       805    ₹613,306      +1.4%     
  BHARTIHEXA  13     IT         2.718    1.13   +80.6%    +46.6%    1,840.5     333    ₹612,873      +7.6%     
  CUB         14     PVT BNK    2.400    0.92   +38.7%    +34.3%    147.6       4156   ₹613,382      +4.1%     
  CEATLTD     15     AUTO       2.672    0.85   +59.0%    +41.4%    3,705.6     165    ₹611,431      +1.8%     
  GODFRYPHLP  16     FMCG       2.300    1.02   +115.2%   +50.0%    2,761.7     222    ₹613,101      +0.2%     
  EIDPARRY    17     FMCG       2.344    1.00   +50.8%    +37.0%    951.9       644    ₹613,024      +1.5%     
  ICICIBANK   18     PVT BNK    2.290    0.97   +29.5%    +19.1%    1,439.4     426    ₹613,182      +0.9%     
  HOMEFIRST   19     FIN SVC    2.220    0.62   +55.4%    +39.5%    1,256.9     488    ₹613,380      +6.2%     
  MRF         21     MFG        2.211    0.75   +7.6%     +29.3%    140,530.3   4      ₹562,121      +1.0%     
  SBILIFE     23     FIN SVC    2.022    0.79   +28.1%    +21.5%    1,800.0     340    ₹611,999      +2.0%     
  KPRMILL     24     MFG        2.057    0.41   +41.0%    +38.2%    1,120.4     547    ₹612,874      -1.0%     

  HOLDS (7)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  KIMS        6      HEALTH     01-Nov-24   541.5       664.1       1275   ₹156,379      +22.7%      +1.2%     
  FORTIS      57     HEALTH     01-Nov-24   634.6       721.5       1088   ₹94,544       +13.7%      +3.6%     
  DIVISLAB    33     HEALTH     01-Nov-24   5,876.8     6,509.4     117    ₹74,011       +10.8%      +2.0%     
  RADICO      30     FMCG       01-Nov-24   2,398.9     2,545.2     287    ₹41,985       +6.1%       +2.2%     
  FSL         77     IT         02-Dec-24   354.2       369.7       1881   ₹29,294       +4.4%       +2.2%     
  COFORGE     61     IT         02-Dec-24   1,722.2     1,705.3     386    ₹-6,515       -1.0%       +4.5%     
  UTIAMC      80     FIN SVC    01-Nov-24   1,316.0     1,155.2     524    ₹-84,245      -12.2%      +4.5%     

  AFTER: Invested ₹17,882,923 | Cash ₹507,250 | Total ₹18,390,172 | Positions 28/30 | Slot ₹613,512

========================================================================
  REBALANCE #127  —  01 Jul 2025
  NAV: ₹19,096,507  |  Slot: ₹636,550  |  Cash: ₹507,250
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  FORTIS      60     HEALTH     01-Nov-24   634.6       773.8       1088   ₹151,488      +21.9%    242d  
  RADICO      107    FMCG       01-Nov-24   2,398.9     2,565.5     287    ₹47,802       +6.9%     242d  
  HOMEFIRST   92     FIN SVC    02-Jun-25   1,256.9     1,314.0     488    ₹27,832       +4.5%     29d   
  PAYTM       26     FIN SVC    02-Jun-25   924.3       930.2       663    ₹3,912        +0.6%     29d   
  FSL         118    IT         02-Dec-24   354.2       350.3       1881   ₹-7,338       -1.1%     211d  
  ICICIBANK   101    PVT BNK    02-Jun-25   1,439.4     1,421.0     426    ₹-7,821       -1.3%     29d   
  KPRMILL     116    MFG        02-Jun-25   1,120.4     1,102.4     547    ₹-9,853       -1.6%     29d   
  UTIAMC      99     FIN SVC    01-Nov-24   1,316.0     1,241.2     524    ₹-39,182      -5.7%     242d  
  ZENTEC      132    DEFENCE    02-Jun-25   2,114.7     1,975.2     290    ₹-40,456      -6.6%     29d   

  ENTRIES (10)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  CRISIL      5      MNC        2.815    0.43   +47.3%    +46.1%    5,954.3     106    ₹631,154      +6.2%     
  AUBANK      6      PVT BNK    2.766    0.62   +21.3%    +51.2%    837.3       760    ₹636,361      +6.9%     
  ENDURANCE   10     AUTO       2.448    0.62   +9.6%     +46.2%    2,873.2     221    ₹634,976      +12.7%    
  GILLETTE    12     FMCG       2.400    0.63   +46.6%    +33.8%    10,485.4    60     ₹629,124      +5.1%     
  JKCEMENT    14     MFG        2.423    0.76   +40.1%    +25.6%    6,121.3     103    ₹630,490      +4.3%     
  KARURVYSYA  15     PVT BNK    2.340    0.84   +36.5%    +31.0%    226.0       2816   ₹636,485      +10.5%    
  BHARTIARTL  17     CONSUMP    2.463    0.93   +39.1%    +17.1%    2,002.7     317    ₹634,843      +4.7%     
  REDINGTON   20     IT         2.191    0.89   +58.5%    +35.4%    315.5       2017   ₹636,264      +10.3%    
  DELHIVERY   22     INFRA      2.256    0.79   -3.6%     +51.8%    384.0       1657   ₹636,205      +4.0%     
  RAMCOCEM    23     MFG        2.144    0.80   +24.9%    +22.0%    1,075.3     591    ₹635,529      +3.8%     

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  KIMS        44     HEALTH     01-Nov-24   541.5       686.2       1275   ₹184,620      +26.7%      +3.9%     
  DEEPAKFERT  5      MFG        02-Jun-25   1,469.7     1,731.1     417    ₹109,015      +17.8%      +10.9%    
  EIDPARRY    14     FMCG       02-Jun-25   951.9       1,109.0     644    ₹101,172      +16.5%      +10.1%    
  DIVISLAB    28     HEALTH     01-Nov-24   5,876.8     6,825.4     117    ₹110,990      +16.1%      +3.5%     
  CUB         7      PVT BNK    02-Jun-25   147.6       171.3       4156   ₹98,412       +16.0%      +14.2%    
  COFORGE     19     IT         02-Dec-24   1,722.2     1,909.3     386    ₹72,224       +10.9%      +5.1%     
  MFSL        3      FIN SVC    02-Jun-25   1,522.3     1,653.9     403    ₹53,035       +8.6%       +4.9%     
  BHARTIHEXA  29     IT         02-Jun-25   1,840.5     1,973.7     333    ₹44,378       +7.2%       +8.0%     
  ERIS        38     HEALTH     02-Jun-25   1,557.7     1,667.7     393    ₹43,195       +7.1%       +1.0%     
  HDFCLIFE    22     FIN SVC    02-Jun-25   761.9       807.0       805    ₹36,348       +5.9%       +3.9%     
  GODFRYPHLP  43     FMCG       02-Jun-25   2,761.7     2,922.3     222    ₹35,643       +5.8%       +5.1%     
  SOLARINDS   4      DEFENCE    02-Jun-25   16,284.3    17,185.7    37     ₹33,354       +5.5%       +2.2%     
  SBILIFE     34     FIN SVC    02-Jun-25   1,800.0     1,859.9     340    ₹20,372       +3.3%       +2.6%     
  COROMANDEL  42     MFG        02-Jun-25   2,265.8     2,307.7     270    ₹11,306       +1.8%       -1.0%     
  MRF         27     MFG        02-Jun-25   140,530.3   142,277.6   4      ₹6,989        +1.2%       +2.3%     
  BDL         37     DEFENCE    02-Jun-25   1,966.8     1,972.5     311    ₹1,766        +0.3%       +4.7%     
  CEATLTD     46     AUTO       02-Jun-25   3,705.6     3,621.3     165    ₹-13,910      -2.3%       +0.3%     
  JSWHL       15     FIN SVC    02-Jun-25   22,430.0    21,400.0    27     ₹-27,810      -4.6%       -2.8%     
  CCL         16     FMCG       02-Jun-25   882.3       830.9       695    ₹-35,696      -5.8%       +1.0%     

  AFTER: Invested ₹19,004,082 | Cash ₹84,895 | Total ₹19,088,977 | Positions 29/30 | Slot ₹636,550

========================================================================
  REBALANCE #128  —  01 Aug 2025
  NAV: ₹18,291,745  |  Slot: ₹609,725  |  Cash: ₹84,895
========================================================================

  EXITS (14)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  DIVISLAB    70     HEALTH     01-Nov-24   5,876.8     6,361.5     117    ₹56,711       +8.2%     273d  
  DEEPAKFERT  45     MFG        02-Jun-25   1,469.7     1,547.4     417    ₹32,419       +5.3%     60d   
  GODFRYPHLP  59     FMCG       02-Jun-25   2,761.7     2,893.6     222    ₹29,275       +4.8%     60d   
  MRF         108    MFG        02-Jun-25   140,530.3   145,899.2   4      ₹21,476       +3.8%     60d   
  BHARTIHEXA  58     IT         02-Jun-25   1,840.5     1,844.6     333    ₹1,379        +0.2%     60d   
  SBILIFE     161    FIN SVC    02-Jun-25   1,800.0     1,793.7     340    ₹-2,139       -0.3%     60d   
  HDFCLIFE    162    FIN SVC    02-Jun-25   761.9       739.1       805    ₹-18,358      -3.0%     60d   
  BHARTIARTL  79     CONSUMP    01-Jul-25   2,002.7     1,884.4     317    ₹-37,488      -5.9%     31d   
  AUBANK      98     PVT BNK    01-Jul-25   837.3       741.0       760    ₹-73,163      -11.5%    31d   
  CRISIL      73     MNC        01-Jul-25   5,954.3     5,150.3     106    ₹-85,219      -13.5%    31d   
  CEATLTD     270    AUTO       02-Jun-25   3,705.6     3,199.5     165    ₹-83,511      -13.7%    60d   
  SOLARINDS   94     DEFENCE    02-Jun-25   16,284.3    13,807.0    37     ₹-91,658      -15.2%    60d   
  BDL         216    DEFENCE    02-Jun-25   1,966.8     1,559.0     311    ₹-126,820     -20.7%    60d   
  REDINGTON   174    IT         01-Jul-25   315.5       244.0       2017   ₹-144,155     -22.7%    31d   

  ENTRIES (13)
  [52w filter blocked 1: KAJARIACER(-24.7%)]
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  ECLERX      1      IT         3.579    0.97   +58.4%    +57.8%    1,899.1     321    ₹609,619      +3.9%     
  ANANDRATHI  2      FIN SVC    3.485    0.44   +39.0%    +53.0%    1,295.5     470    ₹608,882      +3.8%     
  LAURUSLABS  4      HEALTH     3.497    1.18   +85.5%    +38.8%    847.8       719    ₹609,572      +2.8%     
  BOSCHLTD    6      AUTO       2.995    0.96   +17.2%    +36.8%    40,390.0    15     ₹605,850      +7.6%     
  GLENMARK    7      HEALTH     3.380    0.73   +44.5%    +47.1%    2,062.7     295    ₹608,508      -0.2%     
  FORTIS      12     HEALTH     3.089    0.67   +71.3%    +25.2%    859.2       709    ₹609,208      +4.7%     
  POWERINDIA  13     ENERGY     2.578    1.13   +73.2%    +42.6%    20,544.0    29     ₹595,777      +4.8%     
  MUTHOOTFIN  15     FIN SVC    2.333    0.69   +45.1%    +15.3%    2,571.0     237    ₹609,318      -1.5%     
  RADICO      16     FMCG       2.340    0.51   +64.4%    +12.0%    2,838.8     214    ₹607,503      +4.7%     
  GLAND       19     MNC        2.300    0.73   -2.9%     +40.0%    1,960.2     311    ₹609,634      +0.3%     
  ALKYLAMINE  21     MFG        2.356    1.05   +12.3%    +38.6%    2,327.0     262    ₹609,665      +3.8%     
  UTIAMC      22     FIN SVC    2.340    1.05   +30.1%    +35.9%    1,297.0     470    ₹609,590      -3.0%     
  ABSLAMC     23     FIN SVC    2.336    1.18   +23.8%    +35.6%    834.6       730    ₹609,258      -0.7%     

  HOLDS (15)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  KIMS        13     HEALTH     01-Nov-24   541.5       756.5       1275   ₹274,125      +39.7%      +2.5%     
  EIDPARRY    6      FMCG       02-Jun-25   951.9       1,204.6     644    ₹162,739      +26.5%      +3.6%     
  ERIS        17     HEALTH     02-Jun-25   1,557.7     1,798.7     393    ₹94,716       +15.5%      +2.4%     
  COROMANDEL  28     MFG        02-Jun-25   2,265.8     2,580.2     270    ₹84,888       +13.9%      +6.3%     
  DELHIVERY   23     INFRA      01-Jul-25   384.0       429.9       1657   ₹76,056       +12.0%      +2.9%     
  JKCEMENT    4      MFG        01-Jul-25   6,121.3     6,681.5     103    ₹57,704       +9.2%       +3.5%     
  CUB         42     PVT BNK    02-Jun-25   147.6       160.4       4156   ₹53,251       +8.7%       +2.3%     
  RAMCOCEM    10     MFG        01-Jul-25   1,075.3     1,149.1     591    ₹43,600       +6.9%       +0.1%     
  CCL         34     FMCG       02-Jun-25   882.3       886.5       695    ₹2,929        +0.5%       +3.8%     
  GILLETTE    16     FMCG       01-Jul-25   10,485.4    10,446.4    60     ₹-2,341       -0.4%       -0.3%     
  COFORGE     57     IT         02-Dec-24   1,722.2     1,698.5     386    ₹-9,157       -1.4%       -4.8%     
  MFSL        33     FIN SVC    02-Jun-25   1,522.3     1,472.5     403    ₹-20,069      -3.3%       -4.1%     
  KARURVYSYA  50     PVT BNK    01-Jul-25   226.0       214.3       2816   ₹-33,108      -5.2%       -1.6%     
  ENDURANCE   48     AUTO       01-Jul-25   2,873.2     2,476.6     221    ₹-87,648      -13.8%      -4.4%     
  JSWHL       36     FIN SVC    02-Jun-25   22,430.0    19,161.0    27     ₹-88,263      -14.6%      -9.9%     

  AFTER: Invested ₹17,952,403 | Cash ₹329,959 | Total ₹18,282,362 | Positions 28/30 | Slot ₹609,725

========================================================================
  REBALANCE #129  —  01 Sep 2025
  NAV: ₹18,324,741  |  Slot: ₹610,825  |  Cash: ₹329,959
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  CCL         95     FMCG       02-Jun-25   882.3       902.5       695    ₹14,042       +2.3%     91d   
  LAURUSLABS  2      HEALTH     01-Aug-25   847.8       858.4       719    ₹7,643        +1.3%     31d   
  CUB         103    PVT BNK    02-Jun-25   147.6       148.0       4156   ₹1,696        +0.3%     91d   
  UTIAMC      73     FIN SVC    01-Aug-25   1,297.0     1,297.0     470    ₹0            +0.0%     31d   
  GLAND       65     MNC        01-Aug-25   1,960.2     1,871.1     311    ₹-27,722      -4.5%     31d   
  GILLETTE    86     FMCG       01-Jul-25   10,485.4    9,861.3     60     ₹-37,447      -6.0%     62d   
  KARURVYSYA  93     PVT BNK    01-Jul-25   226.0       209.1       2816   ₹-47,659      -7.5%     62d   
  POWERINDIA  80     ENERGY     01-Aug-25   20,544.0    18,942.0    29     ₹-46,459      -7.8%     31d   
  ALKYLAMINE  156    MFG        01-Aug-25   2,327.0     2,024.4     262    ₹-79,279      -13.0%    31d   

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  MARUTI      6      AUTO       3.106    0.86   +20.4%    +22.4%    14,887.0    41     ₹610,367      +7.7%     
  EICHERMOT   7      AUTO       2.921    1.11   +30.7%    +18.7%    6,280.0     97     ₹609,160      +6.6%     
  GODFRYPHLP  11     FMCG       2.515    0.99   +86.4%    +26.9%    3,503.9     174    ₹609,679      +3.6%     
  METROPOLIS  13     HEALTH     2.465    0.74   +3.3%     +29.0%    535.0       1141   ₹610,416      +1.7%     
  HEROMOTOCO  15     AUTO       2.415    1.04   +1.9%     +25.8%    5,143.7     118    ₹606,961      +7.6%     
  ULTRACEMCO  16     INFRA      2.420    0.98   +14.2%    +15.3%    12,826.0    47     ₹602,822      +2.2%     
  UNOMINDA    17     AUTO       2.412    1.09   +18.1%    +29.1%    1,311.6     465    ₹609,897      +9.2%     
  DALBHARAT   18     MFG        2.784    0.81   +32.5%    +16.1%    2,391.8     255    ₹609,905      +3.8%     
  MEDANTA     22     HEALTH     2.319    0.54   +30.1%    +15.1%    1,396.0     437    ₹610,052      +2.1%     

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  KIMS        26     HEALTH     01-Nov-24   541.5       727.5       1275   ₹237,214      +34.4%      -1.8%     
  DELHIVERY   31     INFRA      01-Jul-25   384.0       473.8       1657   ₹148,799      +23.4%      +2.5%     
  EIDPARRY    23     FMCG       02-Jun-25   951.9       1,129.9     644    ₹114,632      +18.7%      -1.9%     
  ECLERX      9      IT         01-Aug-25   1,899.1     2,248.4     321    ₹112,117      +18.4%      +10.1%    
  JKCEMENT    3      MFG        01-Jul-25   6,121.3     7,067.5     103    ₹97,462       +15.5%      +1.6%     
  ERIS        36     HEALTH     02-Jun-25   1,557.7     1,762.7     393    ₹80,565       +13.2%      +0.6%     
  ANANDRATHI  1      FIN SVC    01-Aug-25   1,295.5     1,454.5     470    ₹74,720       +12.3%      +6.6%     
  FORTIS      6      HEALTH     01-Aug-25   859.2       924.7       709    ₹46,369       +7.6%       +1.9%     
  MFSL        16     FIN SVC    02-Jun-25   1,522.3     1,629.2     403    ₹43,081       +7.0%       +1.9%     
  MUTHOOTFIN  14     FIN SVC    01-Aug-25   2,571.0     2,687.3     237    ₹27,567       +4.5%       +1.8%     
  COROMANDEL  64     MFG        02-Jun-25   2,265.8     2,330.2     270    ₹17,384       +2.8%       -2.0%     
  ENDURANCE   32     AUTO       01-Jul-25   2,873.2     2,939.9     221    ₹14,742       +2.3%       +6.9%     
  COFORGE     52     IT         02-Dec-24   1,722.2     1,756.5     386    ₹13,244       +2.0%       +2.5%     
  BOSCHLTD    5      AUTO       01-Aug-25   40,390.0    40,785.0    15     ₹5,925        +1.0%       +4.0%     
  RADICO      17     FMCG       01-Aug-25   2,838.8     2,856.7     214    ₹3,831        +0.6%       +0.6%     
  ABSLAMC     58     FIN SVC    01-Aug-25   834.6       838.1       730    ₹2,555        +0.4%       -2.0%     
  RAMCOCEM    40     MFG        01-Jul-25   1,075.3     1,058.5     591    ₹-9,955       -1.6%       -2.4%     
  GLENMARK    39     HEALTH     01-Aug-25   2,062.7     1,922.1     295    ₹-41,493      -6.8%       -2.8%     
  JSWHL       37     FIN SVC    02-Jun-25   22,430.0    17,920.0    27     ₹-121,770     -20.1%      -3.0%     

  AFTER: Invested ₹18,162,794 | Cash ₹155,441 | Total ₹18,318,234 | Positions 28/30 | Slot ₹610,825

========================================================================
  REBALANCE #130  —  01 Oct 2025
  NAV: ₹17,765,215  |  Slot: ₹592,174  |  Cash: ₹155,441
========================================================================

  EXITS (9)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  EIDPARRY    103    FMCG       02-Jun-25   951.9       1,035.0     644    ₹53,516       +8.7%     121d  
  ENDURANCE   92     AUTO       01-Jul-25   2,873.2     2,812.6     221    ₹-13,392      -2.1%     92d   
  ABSLAMC     91     FIN SVC    01-Aug-25   834.6       799.7       730    ₹-25,477      -4.2%     61d   
  GLENMARK    63     HEALTH     01-Aug-25   2,062.7     1,956.7     295    ₹-31,282      -5.1%     61d   
  ULTRACEMCO  168    INFRA      01-Sep-25   12,826.0    12,095.0    47     ₹-34,357      -5.7%     30d   
  DALBHARAT   93     MFG        01-Sep-25   2,391.8     2,218.7     255    ₹-44,132      -7.2%     30d   
  COFORGE     194    IT         02-Dec-24   1,722.2     1,593.5     386    ₹-49,693      -7.5%     303d  
  RAMCOCEM    152    MFG        01-Jul-25   1,075.3     992.7       591    ₹-48,843      -7.7%     92d   
  JSWHL       —      OTHER      02-Jun-25   22,430.0    15,494.0    27     ₹-187,272     -30.9%    121d  

  ENTRIES (9)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TATAINVEST  1      FIN SVC    4.374    0.91   +54.0%    +55.0%    1,052.6     562    ₹591,545      +30.0%    
  CHOICEIN    6      FIN SVC    3.286    1.05   +63.6%    +10.6%    766.2       772    ₹591,506      -4.0%     
  INDIANB     8      PSU BNK    2.792    1.00   +41.9%    +13.3%    721.7       820    ₹591,775      +4.8%     
  USHAMART    11     METAL      2.652    1.01   +29.8%    +25.7%    452.8       1307   ₹591,744      +6.5%     
  BAJFINANCE  14     FIN SVC    2.348    1.13   +27.9%    +7.0%     981.7       603    ₹591,944      +0.7%     
  MANAPPURAM  18     FIN SVC    2.206    1.04   +42.8%    +5.7%     284.9       2078   ₹592,004      +0.8%     
  ADANIPOWER  19     ENERGY     2.256    1.15   +14.6%    +28.9%    152.5       3882   ₹592,044      +9.2%     
  NYKAA       21     CONSUMP    2.458    0.65   +20.0%    +14.0%    241.3       2454   ₹592,052      +2.1%     
  NLCINDIA    23     ENERGY     2.033    1.19   +1.7%     +21.7%    273.2       2167   ₹591,936      +4.8%     

  HOLDS (19)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  KIMS        50     HEALTH     01-Nov-24   541.5       687.9       1275   ₹186,724      +27.0%      -5.6%     
  MUTHOOTFIN  7      FIN SVC    01-Aug-25   2,571.0     3,118.1     237    ₹129,680      +21.3%      +5.7%     
  FORTIS      6      HEALTH     01-Aug-25   859.2       989.7       709    ₹92,454       +15.2%      +3.4%     
  DELHIVERY   77     INFRA      01-Jul-25   384.0       434.5       1657   ₹83,678       +13.2%      -5.4%     
  EICHERMOT   4      AUTO       01-Sep-25   6,280.0     7,021.5     97     ₹71,926       +11.8%      +3.0%     
  ANANDRATHI  3      FIN SVC    01-Aug-25   1,295.5     1,437.9     470    ₹66,949       +11.0%      -0.7%     
  MARUTI      5      AUTO       01-Sep-25   14,887.0    15,965.0    41     ₹44,198       +7.2%       +2.3%     
  MFSL        38     FIN SVC    02-Jun-25   1,522.3     1,621.9     403    ₹40,139       +6.5%       +2.7%     
  ECLERX      26     IT         01-Aug-25   1,899.1     1,986.8     321    ₹28,128       +4.6%       -5.4%     
  ERIS        62     HEALTH     02-Jun-25   1,557.7     1,616.2     393    ₹22,985       +3.8%       -1.6%     
  HEROMOTOCO  19     AUTO       01-Sep-25   5,143.7     5,328.6     118    ₹21,818       +3.6%       +2.3%     
  JKCEMENT    28     MFG        01-Jul-25   6,121.3     6,305.0     103    ₹18,925       +3.0%       -4.6%     
  RADICO      16     FMCG       01-Aug-25   2,838.8     2,914.2     214    ₹16,136       +2.7%       -0.1%     
  UNOMINDA    30     AUTO       01-Sep-25   1,311.6     1,321.4     465    ₹4,547        +0.7%       +2.6%     
  COROMANDEL  46     MFG        02-Jun-25   2,265.8     2,242.2     270    ₹-6,390       -1.0%       -0.5%     
  GODFRYPHLP  41     FMCG       01-Sep-25   3,503.9     3,357.6     174    ₹-25,463      -4.2%       -1.7%     
  METROPOLIS  53     HEALTH     01-Sep-25   535.0       508.2       1141   ₹-30,541      -5.0%       -1.5%     
  BOSCHLTD    23     AUTO       01-Aug-25   40,390.0    38,320.0    15     ₹-31,050      -5.1%       -2.3%     
  MEDANTA     32     HEALTH     01-Sep-25   1,396.0     1,304.6     437    ₹-39,942      -6.5%       -2.5%     

  AFTER: Invested ₹17,732,859 | Cash ₹26,031 | Total ₹17,758,890 | Positions 28/30 | Slot ₹592,174

========================================================================
  REBALANCE #131  —  03 Nov 2025
  NAV: ₹17,968,712  |  Slot: ₹598,957  |  Cash: ₹26,031
========================================================================

  EXITS (11)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  KIMS        109    HEALTH     01-Nov-24   541.5       726.8       1275   ₹236,257      +34.2%    367d  
  MFSL        97     FIN SVC    02-Jun-25   1,522.3     1,559.9     403    ₹15,153       +2.5%     154d  
  ERIS        206    HEALTH     02-Jun-25   1,557.7     1,588.8     393    ₹12,196       +2.0%     154d  
  JKCEMENT    169    MFG        01-Jul-25   6,121.3     5,899.5     103    ₹-22,842      -3.6%     125d  
  COROMANDEL  245    MFG        02-Jun-25   2,265.8     2,135.8     270    ₹-35,113      -5.7%     154d  
  NLCINDIA    158    ENERGY     01-Oct-25   273.2       255.9       2167   ₹-37,390      -6.3%     33d   
  METROPOLIS  243    HEALTH     01-Sep-25   535.0       499.1       1141   ₹-40,939      -6.7%     63d   
  BOSCHLTD    334    AUTO       01-Aug-25   40,390.0    37,025.0    15     ₹-50,475      -8.3%     94d   
  MEDANTA     178    HEALTH     01-Sep-25   1,396.0     1,274.6     437    ₹-53,052      -8.7%     63d   
  GODFRYPHLP  122    FMCG       01-Sep-25   3,503.9     3,091.0     174    ₹-71,851      -11.8%    63d   
  TATAINVEST  85     FIN SVC    01-Oct-25   1,052.6     794.6       562    ₹-144,979     -24.5%    33d   

  ENTRIES (11)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  TVSMOTOR    5      AUTO       3.048    1.17   +43.2%    +25.3%    3,498.3     171    ₹598,205      -0.7%     
  CANBK       7      PSU BNK    3.024    1.09   +43.6%    +30.2%    135.1       4432   ₹598,949      +8.6%     
  NAVINFLUOR  8      MFG        2.909    0.63   +75.1%    +17.0%    5,892.1     101    ₹595,102      +16.1%    
  SBIN        11     PSU BNK    2.708    0.97   +22.3%    +19.2%    932.9       642    ₹598,912      +4.8%     
  BANKINDIA   13     PSU BNK    2.781    1.12   +47.0%    +27.6%    137.6       4351   ₹598,846      +7.1%     
  RBLBANK     14     PVT BNK    3.190    1.09   +98.1%    +23.7%    328.8       1821   ₹598,654      +6.2%     
  MSUMI       15     AUTO       2.565    0.78   +16.5%    +28.4%    47.7        12569  ₹598,913      +2.1%     
  FEDERALBNK  18     PVT BNK    2.465    0.70   +29.4%    +18.2%    237.9       2517   ₹598,769      +6.8%     
  CUMMINSIND  19     INFRA      2.439    1.14   +30.1%    +23.2%    4,359.9     137    ₹597,303      +5.4%     
  BEL         20     DEFENCE    2.409    1.15   +57.6%    +10.5%    420.5       1424   ₹598,806      +2.3%     
  SAMMAANCAP  21     FIN SVC    2.591    1.16   +35.4%    +49.7%    189.9       3153   ₹598,849      +8.9%     

  HOLDS (17)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ECLERX      21     IT         01-Aug-25   1,899.1     2,366.5     321    ₹150,027      +24.6%      +6.7%     
  MUTHOOTFIN  7      FIN SVC    01-Aug-25   2,571.0     3,163.8     237    ₹140,514      +23.1%      +0.3%     
  DELHIVERY   76     INFRA      01-Jul-25   384.0       472.2       1657   ₹146,313      +23.0%      +0.7%     
  FORTIS      22     HEALTH     01-Aug-25   859.2       1,030.7     709    ₹121,558      +20.0%      -1.2%     
  INDIANB     2      PSU BNK    01-Oct-25   721.7       862.0       820    ₹115,055      +19.4%      +8.6%     
  ANANDRATHI  25     FIN SVC    01-Aug-25   1,295.5     1,546.5     470    ₹117,980      +19.4%      +0.6%     
  RADICO      43     FMCG       01-Aug-25   2,838.8     3,193.6     214    ₹75,927       +12.5%      +2.6%     
  EICHERMOT   3      AUTO       01-Sep-25   6,280.0     7,023.5     97     ₹72,120       +11.8%      +1.3%     
  CHOICEIN    16     FIN SVC    01-Oct-25   766.2       830.5       772    ₹49,678       +8.4%       +1.9%     
  BAJFINANCE  12     FIN SVC    01-Oct-25   981.7       1,036.7     603    ₹33,202       +5.6%       -0.5%     
  HEROMOTOCO  19     AUTO       01-Sep-25   5,143.7     5,433.1     118    ₹34,144       +5.6%       +0.0%     
  USHAMART    56     METAL      01-Oct-25   452.8       478.2       1307   ₹33,328       +5.6%       +4.4%     
  MARUTI      4      AUTO       01-Sep-25   14,887.0    15,651.0    41     ₹31,324       +5.1%       -2.9%     
  NYKAA       33     CONSUMP    01-Oct-25   241.3       250.1       2454   ₹21,644       +3.7%       -1.2%     
  ADANIPOWER  58     ENERGY     01-Oct-25   152.5       156.7       3882   ₹16,382       +2.8%       -0.8%     
  UNOMINDA    46     AUTO       01-Sep-25   1,311.6     1,261.7     465    ₹-23,197      -3.8%       +2.2%     
  MANAPPURAM  27     FIN SVC    01-Oct-25   284.9       266.5       2078   ₹-38,247      -6.5%       -4.3%     

  AFTER: Invested ₹17,939,252 | Cash ₹21,646 | Total ₹17,960,897 | Positions 28/30 | Slot ₹598,957

========================================================================
  REBALANCE #132  —  01 Dec 2025
  NAV: ₹17,930,384  |  Slot: ₹597,679  |  Cash: ₹21,646
========================================================================

  EXITS (7)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  ECLERX      82     IT         01-Aug-25   1,899.1     2,344.9     321    ₹143,094      +23.5%    122d  
  DELHIVERY   212    INFRA      01-Jul-25   384.0       417.6       1657   ₹55,758       +8.8%     153d  
  FORTIS      154    HEALTH     01-Aug-25   859.2       904.8       709    ₹32,330       +5.3%     122d  
  UNOMINDA    138    AUTO       01-Sep-25   1,311.6     1,306.1     465    ₹-2,552       -0.4%     91d   
  MSUMI       88     AUTO       03-Nov-25   47.7        46.7        12569  ₹-12,443      -2.1%     28d   
  USHAMART    132    METAL      01-Oct-25   452.8       422.9       1307   ₹-39,079      -6.6%     61d   
  SAMMAANCAP  147    FIN SVC    03-Nov-25   189.9       152.5       3153   ₹-117,922     -19.7%    28d   

  ENTRIES (7)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  CUB         3      PVT BNK    3.505    0.84   +59.4%    +44.3%    212.2       2816   ₹597,590      +7.1%     
  M&MFIN      5      FIN SVC    3.355    1.20   +39.5%    +44.9%    367.9       1624   ₹597,470      +9.0%     
  AUBANK      6      PVT BNK    3.130    0.74   +60.9%    +32.4%    950.5       628    ₹596,914      +4.1%     
  ASAHIINDIA  16     AUTO       2.657    0.95   +59.8%    +26.8%    1,057.9     564    ₹596,656      +6.3%     
  GMRAIRPORT  17     INFRA      2.710    0.81   +34.2%    +25.1%    107.6       5552   ₹597,617      +6.1%     
  AIAENG      19     MFG        2.423    0.77   +11.0%    +26.8%    3,854.1     155    ₹597,386      +5.1%     
  RELIANCE    20     OIL&GAS    2.319    1.16   +21.4%    +15.4%    1,558.9     383    ₹597,056      +2.7%     

  HOLDS (21)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  2      FIN SVC    01-Aug-25   2,571.0     3,779.1     237    ₹286,340      +47.0%      +6.6%     
  INDIANB     5      PSU BNK    01-Oct-25   721.7       868.8       820    ₹120,675      +20.4%      +2.6%     
  HEROMOTOCO  18     AUTO       01-Sep-25   5,143.7     6,175.1     118    ₹121,705      +20.1%      +7.1%     
  EICHERMOT   13     AUTO       01-Sep-25   6,280.0     7,125.5     97     ₹82,014       +13.5%      +1.6%     
  RADICO      62     FMCG       01-Aug-25   2,838.8     3,217.0     214    ₹80,935       +13.3%      -0.4%     
  ANANDRATHI  74     FIN SVC    01-Aug-25   1,295.5     1,459.6     470    ₹77,124       +12.7%      -1.3%     
  NYKAA       29     CONSUMP    01-Oct-25   241.3       264.9       2454   ₹58,013       +9.8%       +0.6%     
  MARUTI      19     AUTO       01-Sep-25   14,887.0    16,097.0    41     ₹49,610       +8.1%       +1.2%     
  FEDERALBNK  10     PVT BNK    03-Nov-25   237.9       256.6       2517   ₹47,093       +7.9%       +4.9%     
  CANBK       3      PSU BNK    03-Nov-25   135.1       145.7       4432   ₹46,766       +7.8%       +3.7%     
  TVSMOTOR    27     AUTO       03-Nov-25   3,498.3     3,649.0     171    ₹25,782       +4.3%       +4.5%     
  CUMMINSIND  41     INFRA      03-Nov-25   4,359.9     4,523.6     137    ₹22,434       +3.8%       +4.5%     
  CHOICEIN    67     FIN SVC    01-Oct-25   766.2       794.9       772    ₹22,156       +3.7%       -0.4%     
  BANKINDIA   14     PSU BNK    03-Nov-25   137.6       142.6       4351   ₹21,530       +3.6%       +1.8%     
  BAJFINANCE  17     FIN SVC    01-Oct-25   981.7       1,014.9     603    ₹20,017       +3.4%       -0.3%     
  SBIN        15     PSU BNK    03-Nov-25   932.9       955.9       642    ₹14,757       +2.5%       +1.1%     
  MANAPPURAM  25     FIN SVC    01-Oct-25   284.9       281.5       2078   ₹-7,099       -1.2%       +0.7%     
  BEL         51     DEFENCE    03-Nov-25   420.5       415.5       1424   ₹-7,161       -1.2%       +0.3%     
  NAVINFLUOR  11     MFG        03-Nov-25   5,892.1     5,743.1     101    ₹-15,050      -2.5%       -0.3%     
  ADANIPOWER  53     ENERGY     01-Oct-25   152.5       147.3       3882   ₹-20,070      -3.4%       -2.2%     
  RBLBANK     24     PVT BNK    03-Nov-25   328.8       307.0       1821   ₹-39,516      -6.6%       -1.9%     

  AFTER: Invested ₹17,775,805 | Cash ₹149,615 | Total ₹17,925,420 | Positions 28/30 | Slot ₹597,679

========================================================================
  REBALANCE #133  —  01 Jan 2026
  NAV: ₹18,160,087  |  Slot: ₹605,336  |  Cash: ₹149,615
========================================================================

  EXITS (6)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  RADICO      72     FMCG       01-Aug-25   2,838.8     3,253.9     214    ₹88,831       +14.6%    153d  
  M&MFIN      5      FIN SVC    01-Dec-25   367.9       404.1       1624   ₹58,870       +9.9%     31d   
  TVSMOTOR    29     AUTO       03-Nov-25   3,498.3     3,781.2     171    ₹48,378       +8.1%     59d   
  BAJFINANCE  92     FIN SVC    01-Oct-25   981.7       967.2       603    ₹-8,750       -1.5%     92d   
  ADANIPOWER  116    ENERGY     01-Oct-25   152.5       148.8       3882   ₹-14,558      -2.5%     92d   
  BEL         105    DEFENCE    03-Nov-25   420.5       396.0       1424   ₹-34,882      -5.8%     59d   

  ENTRIES (6)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  UPL         6      MFG        3.160    1.01   +61.9%    +22.8%    805.3       751    ₹604,818      +4.8%     
  SBILIFE     7      FIN SVC    2.965    0.85   +45.5%    +14.0%    2,037.6     297    ₹605,158      +1.2%     
  KARURVYSYA  9      PVT BNK    2.827    0.94   +46.6%    +27.7%    269.4       2246   ₹605,072      +6.2%     
  TITAN       13     CON DUR    2.617    0.74   +22.7%    +20.3%    4,049.3     149    ₹603,346      +2.9%     
  ASIANPAINT  18     CONSUMP    2.465    0.69   +22.5%    +17.3%    2,728.3     221    ₹602,961      -1.3%     
  CANFINHOME  19     FIN SVC    2.401    0.91   +26.0%    +23.0%    931.5       649    ₹604,511      +1.7%     

  HOLDS (22)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  10     FIN SVC    01-Aug-25   2,571.0     3,806.8     237    ₹292,897      +48.1%      +1.8%     
  ANANDRATHI  47     FIN SVC    01-Aug-25   1,295.5     1,544.4     470    ₹116,995      +19.2%      +2.6%     
  EICHERMOT   39     AUTO       01-Sep-25   6,280.0     7,348.0     97     ₹103,596      +17.0%      +1.8%     
  INDIANB     41     PSU BNK    01-Oct-25   721.7       815.2       820    ₹76,717       +13.0%      +3.5%     
  MARUTI      28     AUTO       01-Sep-25   14,887.0    16,708.0    41     ₹74,661       +12.2%      +1.6%     
  FEDERALBNK  3      PVT BNK    03-Nov-25   237.9       266.2       2517   ₹71,382       +11.9%      +1.6%     
  HEROMOTOCO  49     AUTO       01-Sep-25   5,143.7     5,729.8     118    ₹69,157       +11.4%      +0.4%     
  CANBK       16     PSU BNK    03-Nov-25   135.1       149.3       4432   ₹62,812       +10.5%      +2.9%     
  NYKAA       24     CONSUMP    01-Oct-25   241.3       265.8       2454   ₹60,098       +10.2%      +3.2%     
  MANAPPURAM  31     FIN SVC    01-Oct-25   284.9       313.1       2078   ₹58,558       +9.9%       +5.8%     
  CHOICEIN    45     FIN SVC    01-Oct-25   766.2       838.2       772    ₹55,584       +9.4%       +4.9%     
  AUBANK      4      PVT BNK    01-Dec-25   950.5       999.5       628    ₹30,741       +5.1%       +2.4%     
  AIAENG      20     MFG        01-Dec-25   3,854.1     4,030.1     155    ₹27,280       +4.6%       +4.4%     
  SBIN        32     PSU BNK    03-Nov-25   932.9       967.3       642    ₹22,104       +3.7%       +1.5%     
  BANKINDIA   36     PSU BNK    03-Nov-25   137.6       142.3       4351   ₹20,477       +3.4%       +3.3%     
  CUB         9      PVT BNK    01-Dec-25   212.2       216.8       2816   ₹12,778       +2.1%       +2.4%     
  CUMMINSIND  46     INFRA      03-Nov-25   4,359.9     4,450.4     137    ₹12,397       +2.1%       +0.7%     
  NAVINFLUOR  11     MFG        03-Nov-25   5,892.1     5,925.4     101    ₹3,361        +0.6%       +0.5%     
  RELIANCE    22     OIL&GAS    01-Dec-25   1,558.9     1,568.3     383    ₹3,622        +0.6%       +1.5%     
  GMRAIRPORT  38     INFRA      01-Dec-25   107.6       105.5       5552   ₹-11,881      -2.0%       +2.7%     
  RBLBANK     26     PVT BNK    03-Nov-25   328.8       315.3       1821   ₹-24,492      -4.1%       +2.7%     
  ASAHIINDIA  64     AUTO       01-Dec-25   1,057.9     997.6       564    ₹-34,009      -5.7%       -0.3%     

  AFTER: Invested ₹17,912,476 | Cash ₹243,305 | Total ₹18,155,781 | Positions 28/30 | Slot ₹605,336

========================================================================
  REBALANCE #134  —  02 Feb 2026
  NAV: ₹17,325,214  |  Slot: ₹577,507  |  Cash: ₹243,305
========================================================================

  [REGIME OFF] Nifty 500 22,837.0 < SMA200 23,150.1 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (28)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  23     FIN SVC    01-Aug-25   2,571.0     3,505.5     237    ₹221,476      +36.3%      -8.4%     
  FEDERALBNK  9      PVT BNK    03-Nov-25   237.9       281.3       2517   ₹109,263      +18.2%      +2.9%     
  INDIANB     38     PSU BNK    01-Oct-25   721.7       817.4       820    ₹78,483       +13.3%      -3.0%     
  EICHERMOT   52     AUTO       01-Sep-25   6,280.0     6,985.5     97     ₹68,434       +11.2%      -2.8%     
  ANANDRATHI  58     FIN SVC    01-Aug-25   1,295.5     1,432.7     470    ₹64,505       +10.6%      -4.2%     
  KARURVYSYA  10     PVT BNK    01-Jan-26   269.4       295.8       2246   ₹59,182       +9.8%       +7.0%     
  SBIN        11     PSU BNK    03-Nov-25   932.9       1,010.5     642    ₹49,820       +8.3%       -0.3%     
  HEROMOTOCO  45     AUTO       01-Sep-25   5,143.7     5,515.5     118    ₹43,867       +7.2%       -0.1%     
  BANKINDIA   39     PSU BNK    03-Nov-25   137.6       146.9       4351   ₹40,195       +6.7%       -3.1%     
  CANBK       22     PSU BNK    03-Nov-25   135.1       141.8       4432   ₹29,476       +4.9%       -3.6%     
  AIAENG      27     MFG        01-Dec-25   3,854.1     4,016.2     155    ₹25,125       +4.2%       +2.9%     
  AUBANK      16     PVT BNK    01-Dec-25   950.5       965.2       628    ₹9,232        +1.5%       -1.9%     
  CUB         6      PVT BNK    01-Dec-25   212.2       214.1       2816   ₹5,280        +0.9%       +0.6%     
  NAVINFLUOR  14     MFG        03-Nov-25   5,892.1     5,893.9     101    ₹183          +0.0%       -0.7%     
  NYKAA       83     CONSUMP    01-Oct-25   241.3       237.6       2454   ₹-8,933       -1.5%       -3.3%     
  SBILIFE     28     FIN SVC    01-Jan-26   2,037.6     1,998.2     297    ₹-11,686      -1.9%       -1.7%     
  MANAPPURAM  48     FIN SVC    01-Oct-25   284.9       278.5       2078   ₹-13,313      -2.2%       -6.0%     
  CANFINHOME  44     FIN SVC    01-Jan-26   931.5       909.8       649    ₹-14,051      -2.3%       -0.2%     
  CHOICEIN    91     FIN SVC    01-Oct-25   766.2       748.5       772    ₹-13,626      -2.3%       -4.8%     
  TITAN       59     CON DUR    01-Jan-26   4,049.3     3,953.2     149    ₹-14,319      -2.4%       -2.2%     
  MARUTI      186    AUTO       01-Sep-25   14,887.0    14,384.0    41     ₹-20,623      -3.4%       -7.8%     
  ASAHIINDIA  43     AUTO       01-Dec-25   1,057.9     992.2       564    ₹-37,055      -6.2%       +2.4%     
  CUMMINSIND  67     INFRA      03-Nov-25   4,359.9     4,073.8     137    ₹-39,196      -6.6%       -0.1%     
  RBLBANK     51     PVT BNK    03-Nov-25   328.8       297.0       1821   ₹-57,908      -9.7%       -1.6%     
  RELIANCE    164    OIL&GAS    01-Dec-25   1,558.9     1,384.0     383    ₹-66,983      -11.2%      -3.1%     
  GMRAIRPORT  75     INFRA      01-Dec-25   107.6       94.0        5552   ₹-75,674      -12.7%      -3.1%     
  ASIANPAINT  168    CONSUMP    01-Jan-26   2,728.3     2,381.3     221    ₹-76,685      -12.7%      -9.3%     
  UPL         103    MFG        01-Jan-26   805.3       698.5       751    ₹-80,207      -13.3%      -5.2%     

  AFTER: Invested ₹17,081,909 | Cash ₹243,305 | Total ₹17,325,214 | Positions 28/30 | Slot ₹577,507

========================================================================
  REBALANCE #135  —  02 Mar 2026
  NAV: ₹17,848,740  |  Slot: ₹594,958  |  Cash: ₹243,305
========================================================================

  [REGIME OFF] Nifty 500 22,835.9 < SMA200 23,303.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (28)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  116    FIN SVC    01-Aug-25   2,571.0     3,442.8     237    ₹206,623      +33.9%      -2.9%     
  INDIANB     14     PSU BNK    01-Oct-25   721.7       956.3       820    ₹192,374      +32.5%      +4.4%     
  SBIN        1      PSU BNK    03-Nov-25   932.9       1,168.8     642    ₹151,478      +25.3%      +1.6%     
  EICHERMOT   17     AUTO       01-Sep-25   6,280.0     7,826.0     97     ₹149,962      +24.6%      +0.5%     
  FEDERALBNK  10     PVT BNK    03-Nov-25   237.9       295.0       2517   ₹143,620      +24.0%      +1.5%     
  ANANDRATHI  44     FIN SVC    01-Aug-25   1,295.5     1,571.2     470    ₹129,567      +21.3%      +3.8%     
  BANKINDIA   18     PSU BNK    03-Nov-25   137.6       166.8       4351   ₹127,117      +21.2%      +1.7%     
  KARURVYSYA  9      PVT BNK    01-Jan-26   269.4       315.3       2246   ₹103,091      +17.0%      -0.9%     
  CUMMINSIND  24     INFRA      03-Nov-25   4,359.9     4,816.8     137    ₹62,599       +10.5%      +4.1%     
  CANBK       28     PSU BNK    03-Nov-25   135.1       148.7       4432   ₹59,895       +10.0%      +0.6%     
  HEROMOTOCO  117    AUTO       01-Sep-25   5,143.7     5,591.5     118    ₹52,836       +8.7%       -0.2%     
  NYKAA       91     CONSUMP    01-Oct-25   241.3       259.1       2454   ₹43,779       +7.4%       -2.0%     
  NAVINFLUOR  49     MFG        03-Nov-25   5,892.1     6,261.5     101    ₹37,306       +6.3%       -0.9%     
  TITAN       35     CON DUR    01-Jan-26   4,049.3     4,270.3     149    ₹32,929       +5.5%       +0.9%     
  AUBANK      43     PVT BNK    01-Dec-25   950.5       951.2       628    ₹471          +0.1%       -3.7%     
  SBILIFE     57     FIN SVC    01-Jan-26   2,037.6     2,029.4     297    ₹-2,432       -0.4%       -0.9%     
  MANAPPURAM  101    FIN SVC    01-Oct-25   284.9       281.6       2078   ₹-6,935       -1.2%       -5.4%     
  CUB         53     PVT BNK    01-Dec-25   212.2       208.0       2816   ₹-11,827      -2.0%       -2.8%     
  AIAENG      182    MFG        01-Dec-25   3,854.1     3,738.9     155    ₹-17,856      -3.0%       -4.3%     
  MARUTI      220    AUTO       01-Sep-25   14,887.0    14,388.0    41     ₹-20,459      -3.4% ⚠     -4.5%     
  RBLBANK     59     PVT BNK    03-Nov-25   328.8       313.1       1821   ₹-28,408      -4.7%       -1.4%     
  CHOICEIN    145    FIN SVC    01-Oct-25   766.2       709.2       772    ₹-43,965      -7.4%       -7.1%     
  GMRAIRPORT  166    INFRA      01-Dec-25   107.6       96.5        5552   ₹-61,905      -10.4%      -2.6%     
  CANFINHOME  137    FIN SVC    01-Jan-26   931.5       821.4       649    ₹-71,422      -11.8%      -7.6%     
  RELIANCE    279    OIL&GAS    01-Dec-25   1,558.9     1,351.8     383    ₹-79,336      -13.3% ⚠    -4.3%     
  ASIANPAINT  358    CONSUMP    01-Jan-26   2,728.3     2,287.3     221    ₹-97,477      -16.2% ⚠    -5.1%     
  ASAHIINDIA  247    AUTO       01-Dec-25   1,057.9     850.8       564    ₹-116,776     -19.6% ⚠    -9.4%     
  UPL         349    MFG        01-Jan-26   805.3       622.8       751    ₹-137,058     -22.7% ⚠    -10.2%    
  ⚠  WAZ < 0 (momentum below universe mean): MARUTI, ASAHIINDIA, RELIANCE, UPL, ASIANPAINT

  AFTER: Invested ₹17,605,435 | Cash ₹243,305 | Total ₹17,848,740 | Positions 28/30 | Slot ₹594,958

========================================================================
  REBALANCE #136  —  01 Apr 2026
  NAV: ₹16,456,041  |  Slot: ₹548,535  |  Cash: ₹243,305
========================================================================

  [REGIME OFF] Nifty 500 20,935.2 < SMA200 23,188.4 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (28)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  MUTHOOTFIN  108    FIN SVC    01-Aug-25   2,571.0     3,228.7     237    ₹155,884      +25.6%      -1.7%     
  INDIANB     8      PSU BNK    01-Oct-25   721.7       869.5       820    ₹121,197      +20.5%      -0.2%     
  ANANDRATHI  10     FIN SVC    01-Aug-25   1,295.5     1,554.3     470    ₹121,616      +20.0%      +2.2%     
  FEDERALBNK  42     PVT BNK    03-Nov-25   237.9       267.6       2517   ₹74,780       +12.5%      -1.1%     
  EICHERMOT   85     AUTO       01-Sep-25   6,280.0     6,825.5     97     ₹52,914       +8.7%       -3.2%     
  KARURVYSYA  7      PVT BNK    01-Jan-26   269.4       291.9       2246   ₹50,423       +8.3%       +0.4%     
  SBIN        22     PSU BNK    03-Nov-25   932.9       999.8       642    ₹42,946       +7.2%       -4.7%     
  CUMMINSIND  15     INFRA      03-Nov-25   4,359.9     4,609.1     137    ₹34,144       +5.7%       -0.3%     
  NAVINFLUOR  45     MFG        03-Nov-25   5,892.1     6,026.7     101    ₹13,600       +2.3%       -2.7%     
  TITAN       37     CON DUR    01-Jan-26   4,049.3     4,065.5     149    ₹2,414        +0.4%       -0.2%     
  BANKINDIA   64     PSU BNK    03-Nov-25   137.6       137.2       4351   ₹-1,812       -0.3%       -6.2%     
  HEROMOTOCO  59     AUTO       01-Sep-25   5,143.7     5,122.0     118    ₹-2,565       -0.4%       -3.5%     
  NYKAA       94     CONSUMP    01-Oct-25   241.3       240.0       2454   ₹-3,166       -0.5%       -2.1%     
  AIAENG      129    MFG        01-Dec-25   3,854.1     3,681.0     155    ₹-26,831      -4.5%       +1.6%     
  AUBANK      65     PVT BNK    01-Dec-25   950.5       874.9       628    ₹-47,477      -8.0%       -3.5%     
  RBLBANK     40     PVT BNK    03-Nov-25   328.8       301.6       1821   ₹-49,349      -8.2%       +0.4%     
  CANBK       110    PSU BNK    03-Nov-25   135.1       123.2       4432   ₹-52,773      -8.8%       -6.8%     
  MANAPPURAM  215    FIN SVC    01-Oct-25   284.9       255.0       2078   ₹-62,122      -10.5% ⚠    -3.0%     
  SBILIFE     160    FIN SVC    01-Jan-26   2,037.6     1,790.5     297    ₹-73,379      -12.1%      -5.5%     
  CANFINHOME  123    FIN SVC    01-Jan-26   931.5       814.3       649    ₹-75,998      -12.6%      -2.5%     
  RELIANCE    204    OIL&GAS    01-Dec-25   1,558.9     1,362.9     383    ₹-75,066      -12.6% ⚠    -1.6%     
  CUB         111    PVT BNK    01-Dec-25   212.2       179.8       2816   ₹-91,386      -15.3%      -4.4%     
  MARUTI      324    AUTO       01-Sep-25   14,887.0    12,509.0    41     ₹-97,498      -16.0% ⚠    -4.6%     
  CHOICEIN    176    FIN SVC    01-Oct-25   766.2       640.2       772    ₹-97,233      -16.4%      -1.9%     
  GMRAIRPORT  171    INFRA      01-Dec-25   107.6       89.3        5552   ₹-101,879     -17.0%      -2.0%     
  ASIANPAINT  308    CONSUMP    01-Jan-26   2,728.3     2,206.7     221    ₹-115,290     -19.1% ⚠    -1.0%     
  ASAHIINDIA  135    AUTO       01-Dec-25   1,057.9     822.7       564    ₹-132,653     -22.2%      -2.5%     
  UPL         323    MFG        01-Jan-26   805.3       594.5       751    ₹-158,348     -26.2% ⚠    -4.4%     
  ⚠  WAZ < 0 (momentum below universe mean): RELIANCE, MANAPPURAM, ASIANPAINT, UPL, MARUTI

  AFTER: Invested ₹16,212,736 | Cash ₹243,305 | Total ₹16,456,041 | Positions 28/30 | Slot ₹548,535

========================================================================
  REBALANCE #137  —  01 May 2026
  NAV: ₹17,583,903  |  Slot: ₹586,130  |  Cash: ₹243,305
========================================================================

  [REGIME OFF] Nifty 500 22,683.6 < SMA200 23,107.5 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (28)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ANANDRATHI  4      FIN SVC    01-Aug-25   1,295.5     1,797.3     470    ₹235,837      +38.7%      +2.6%     
  MUTHOOTFIN  119    FIN SVC    01-Aug-25   2,571.0     3,424.2     237    ₹202,218      +33.2%      -1.0%     
  CUMMINSIND  2      INFRA      03-Nov-25   4,359.9     5,266.4     137    ₹124,194      +20.8%      +3.6%     
  FEDERALBNK  93     PVT BNK    03-Nov-25   237.9       287.0       2517   ₹123,484      +20.6%      +0.1%     
  NAVINFLUOR  57     MFG        03-Nov-25   5,892.1     6,812.8     101    ₹92,991       +15.6%      +6.0%     
  INDIANB     111    PSU BNK    01-Oct-25   721.7       834.1       820    ₹92,172       +15.6%      -5.6%     
  EICHERMOT   132    AUTO       01-Sep-25   6,280.0     7,109.0     97     ₹80,413       +13.2%      -0.1%     
  SBIN        96     PSU BNK    03-Nov-25   932.9       1,049.5     642    ₹74,888       +12.5%      -1.1%     
  NYKAA       101    CONSUMP    01-Oct-25   241.3       264.8       2454   ₹57,669       +9.7%       +1.4%     
  KARURVYSYA  89     PVT BNK    01-Jan-26   269.4       293.5       2246   ₹54,129       +8.9%       +1.3%     
  TITAN       73     CON DUR    01-Jan-26   4,049.3     4,385.2     149    ₹50,049       +8.3%       +0.1%     
  AUBANK      87     PVT BNK    01-Dec-25   950.5       1,016.0     628    ₹41,103       +6.9%       +1.9%     
  MANAPPURAM  144    FIN SVC    01-Oct-25   284.9       293.9       2078   ₹18,688       +3.2%       +4.7%     
  AIAENG      118    MFG        01-Dec-25   3,854.1     3,949.6     155    ₹14,802       +2.5%       +1.4%     
  RBLBANK     58     PVT BNK    03-Nov-25   328.8       336.5       1821   ₹14,204       +2.4%       +5.1%     
  HEROMOTOCO  133    AUTO       01-Sep-25   5,143.7     5,099.0     118    ₹-5,279       -0.9%       -1.2%     
  BANKINDIA   246    PSU BNK    03-Nov-25   137.6       135.4       4351   ₹-9,522       -1.6% ⚠     -4.3%     
  CANBK       169    PSU BNK    03-Nov-25   135.1       130.4       4432   ₹-21,238      -3.5%       -2.7%     
  CUB         123    PVT BNK    01-Dec-25   212.2       202.6       2816   ₹-27,160      -4.5%       +2.6%     
  CANFINHOME  171    FIN SVC    01-Jan-26   931.5       865.2       649    ₹-42,964      -7.1% ⚠     -0.4%     
  RELIANCE    158    OIL&GAS    01-Dec-25   1,558.9     1,424.2     383    ₹-51,581      -8.6%       +3.9%     
  GMRAIRPORT  180    INFRA      01-Dec-25   107.6       96.4        5552   ₹-62,238      -10.4%      +1.0%     
  MARUTI      292    AUTO       01-Sep-25   14,887.0    13,314.0    41     ₹-64,493      -10.6% ⚠    +0.7%     
  SBILIFE     297    FIN SVC    01-Jan-26   2,037.6     1,819.0     297    ₹-64,915      -10.7% ⚠    -2.3%     
  ASIANPAINT  294    CONSUMP    01-Jan-26   2,728.3     2,423.5     221    ₹-67,373      -11.2% ⚠    +1.1%     
  CHOICEIN    315    FIN SVC    01-Oct-25   766.2       663.5       772    ₹-79,323      -13.4% ⚠    -3.3%     
  UPL         348    MFG        01-Jan-26   805.3       641.8       751    ₹-122,788     -20.3% ⚠    +0.2%     
  ASAHIINDIA  289    AUTO       01-Dec-25   1,057.9     836.2       564    ₹-125,011     -21.0% ⚠    -1.2%     
  ⚠  WAZ < 0 (momentum below universe mean): CANFINHOME, BANKINDIA, ASAHIINDIA, MARUTI, ASIANPAINT, SBILIFE, CHOICEIN, UPL

  AFTER: Invested ₹17,340,598 | Cash ₹243,305 | Total ₹17,583,903 | Positions 28/30 | Slot ₹586,130

========================================================================
  REBALANCE #138  —  01 Jun 2026
  NAV: ₹17,465,556  |  Slot: ₹582,185  |  Cash: ₹243,305
========================================================================

  [REGIME OFF] Nifty 500 22,437.9 < SMA200 23,053.2 — holding all, skipping exits & entries

  EXITS (0)
    —

  ENTRIES (0)
    —

  HOLDS (28)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ANANDRATHI  15     FIN SVC    01-Aug-25   1,295.5     1,751.2     470    ₹214,182      +35.2%      -1.3%     
  CUMMINSIND  9      INFRA      03-Nov-25   4,359.9     5,680.5     137    ₹180,926      +30.3%      +3.4%     
  MUTHOOTFIN  92     FIN SVC    01-Aug-25   2,571.0     3,246.4     237    ₹160,079      +26.3%      -3.3%     
  FEDERALBNK  75     PVT BNK    03-Nov-25   237.9       288.2       2517   ₹126,630      +21.1%      +0.1%     
  NAVINFLUOR  42     MFG        03-Nov-25   5,892.1     6,990.1     101    ₹110,897      +18.6%      -0.6%     
  AIAENG      40     MFG        01-Dec-25   3,854.1     4,522.7     155    ₹103,633      +17.3%      +9.8%     
  EICHERMOT   150    AUTO       01-Sep-25   6,280.0     7,100.5     97     ₹79,588       +13.1%      -0.9%     
  MANAPPURAM  73     FIN SVC    01-Oct-25   284.9       316.0       2078   ₹64,748       +10.9%      +0.5%     
  NYKAA       114    CONSUMP    01-Oct-25   241.3       266.7       2454   ₹62,430       +10.5%      -0.3%     
  INDIANB     181    PSU BNK    01-Oct-25   721.7       792.9       820    ₹58,371       +9.9%       -3.3%     
  KARURVYSYA  132    PVT BNK    01-Jan-26   269.4       286.3       2246   ₹37,957       +6.3%       -1.8%     
  RBLBANK     70     PVT BNK    03-Nov-25   328.8       338.8       1821   ₹18,301       +3.1%       +0.8%     
  SBIN        268    PSU BNK    03-Nov-25   932.9       954.1       642    ₹13,620       +2.3% ⚠     -2.5%     
  AUBANK      96     PVT BNK    01-Dec-25   950.5       969.2       628    ₹11,712       +2.0%       -2.4%     
  TITAN       210    CON DUR    01-Jan-26   4,049.3     4,024.6     149    ₹-3,680       -0.6% ⚠     -3.3%     
  BANKINDIA   235    PSU BNK    03-Nov-25   137.6       136.7       4351   ₹-3,890       -0.6% ⚠     -1.3%     
  ASIANPAINT  89     CONSUMP    01-Jan-26   2,728.3     2,609.8     221    ₹-26,204      -4.3%       +1.3%     
  HEROMOTOCO  262    AUTO       01-Sep-25   5,143.7     4,819.9     118    ₹-38,213      -6.3% ⚠     -4.1%     
  CANBK       261    PSU BNK    03-Nov-25   135.1       123.9       4432   ₹-49,941      -8.3% ⚠     -2.8%     
  GMRAIRPORT  220    INFRA      01-Dec-25   107.6       97.1        5552   ₹-58,740      -9.8% ⚠     +0.1%     
  CANFINHOME  203    FIN SVC    01-Jan-26   931.5       837.0       649    ₹-61,266      -10.1% ⚠    -1.1%     
  CUB         172    PVT BNK    01-Dec-25   212.2       190.6       2816   ₹-60,931      -10.2%      -0.2%     
  SBILIFE     321    FIN SVC    01-Jan-26   2,037.6     1,812.5     297    ₹-66,845      -11.0% ⚠    -2.5%     
  CHOICEIN    316    FIN SVC    01-Oct-25   766.2       668.5       772    ₹-75,386      -12.7% ⚠    -1.2%     
  MARUTI      312    AUTO       01-Sep-25   14,887.0    12,946.0    41     ₹-79,581      -13.0% ⚠    -1.8%     
  ASAHIINDIA  167    AUTO       01-Dec-25   1,057.9     898.8       564    ₹-89,732      -15.0%      +4.0%     
  RELIANCE    291    OIL&GAS    01-Dec-25   1,558.9     1,313.9     383    ₹-93,823      -15.7% ⚠    -2.8%     
  UPL         192    MFG        01-Jan-26   805.3       645.2       751    ₹-120,235     -19.9%      +0.2%     
  ⚠  WAZ < 0 (momentum below universe mean): CANFINHOME, TITAN, GMRAIRPORT, BANKINDIA, CANBK, HEROMOTOCO, SBIN, RELIANCE, MARUTI, CHOICEIN, SBILIFE

  AFTER: Invested ₹17,222,251 | Cash ₹243,305 | Total ₹17,465,556 | Positions 28/30 | Slot ₹582,185

========================================================================
  REBALANCE #139  —  01 Jul 2026
  NAV: ₹18,387,622  |  Slot: ₹612,921  |  Cash: ₹243,305
========================================================================

  EXITS (22)
  Ticker      Rank   Sector     Entry       Entry₹      Exit₹       Qty    Gross P&L     P&L%      Hold  
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ────────  ──────
  INDIANB     207    PSU BNK    01-Oct-25   721.7       821.0       820    ₹81,404       +13.8%    273d  
  EICHERMOT   147    AUTO       01-Sep-25   6,280.0     7,139.0     97     ₹83,323       +13.7%    303d  
  MUTHOOTFIN  310    FIN SVC    01-Aug-25   2,571.0     2,914.9     237    ₹81,513       +13.4%    334d  
  SBIN        111    PSU BNK    03-Nov-25   932.9       1,047.4     642    ₹73,519       +12.3%    240d  
  KARURVYSYA  137    PVT BNK    01-Jan-26   269.4       300.1       2246   ₹68,952       +11.4%    181d  
  MANAPPURAM  118    FIN SVC    01-Oct-25   284.9       316.6       2078   ₹65,995       +11.1%    273d  
  AUBANK      94     PVT BNK    01-Dec-25   950.5       1,051.4     628    ₹63,365       +10.6%    212d  
  RBLBANK     85     PVT BNK    03-Nov-25   328.8       361.8       1821   ₹60,184       +10.1%    240d  
  TITAN       150    CON DUR    01-Jan-26   4,049.3     4,398.6     149    ₹52,046       +8.6%     181d  
  GMRAIRPORT  68     INFRA      01-Dec-25   107.6       113.5       5552   ₹32,701       +5.5%     212d  
  BANKINDIA   184    PSU BNK    03-Nov-25   137.6       141.6       4351   ₹17,430       +2.9%     240d  
  CUB         95     PVT BNK    01-Dec-25   212.2       216.1       2816   ₹10,806       +1.8%     212d  
  CHOICEIN    135    FIN SVC    01-Oct-25   766.2       763.8       772    ₹-1,814       -0.3%     273d  
  ASIANPAINT  71     CONSUMP    01-Jan-26   2,728.3     2,716.4     221    ₹-2,636       -0.4%     181d  
  MARUTI      127    AUTO       01-Sep-25   14,887.0    14,395.0    41     ₹-20,172      -3.3%     303d  
  CANFINHOME  177    FIN SVC    01-Jan-26   931.5       882.3       649    ₹-31,866      -5.3%     181d  
  HEROMOTOCO  260    AUTO       01-Sep-25   5,143.7     4,834.9     118    ₹-36,443      -6.0%     303d  
  CANBK       210    PSU BNK    03-Nov-25   135.1       126.3       4432   ₹-39,276      -6.6%     240d  
  SBILIFE     305    FIN SVC    01-Jan-26   2,037.6     1,790.9     297    ₹-73,260      -12.1%    181d  
  RELIANCE    358    OIL&GAS    01-Dec-25   1,558.9     1,308.0     383    ₹-96,092      -16.1%    212d  
  ASAHIINDIA  185    AUTO       01-Dec-25   1,057.9     855.0       564    ₹-114,407     -19.2%    212d  
  UPL         385    MFG        01-Jan-26   805.3       565.0       751    ₹-180,465     -29.8%    181d  

  ENTRIES (22)
  Ticker      Rank   Sector     Score    Beta   Ret12m    Ret3m     Entry₹      Qty    Capital       Ext%EMA20 
  ──────────  ─────  ─────────  ───────  ─────  ────────  ────────  ──────────  ─────  ────────────  ──────────
  HFCL        1      IT         5.688    0.07   +142.7%   +192.7%   212.1       2889   ₹612,728      +8.0%     
  ACUTAAS     2      FIN SVC    4.178    -0.17  +223.0%   +53.0%    3,567.5     171    ₹610,042      +9.4%     
  SYRMA       3      MFG        4.056    -0.24  +164.1%   +78.2%    1,421.6     431    ₹612,710      +7.8%     
  KIRLOSENG   4      MFG        3.779    -0.18  +175.9%   +71.3%    2,341.1     261    ₹611,027      +7.2%     
  LAURUSLABS  5      HEALTH     3.815    -0.10  +119.9%   +43.9%    1,493.3     410    ₹612,253      +4.4%     
  ATHERENERG  6      ENERGY     3.736    0.05   +236.4%   +46.6%    1,130.4     542    ₹612,677      +10.6%    
  CEMPRO      7      MFG        3.582    -0.04  +56.1%    +156.5%   1,363.3     449    ₹612,122      +14.3%    
  AEGISLOG    8      INFRA      3.466    -0.14  +58.9%    +107.9%   1,254.7     488    ₹612,294      +22.9%    
  RRKABEL     9      ENERGY     3.434    -0.26  +78.5%    +79.1%    2,381.7     257    ₹612,097      +3.8%     
  WELCORP     10     METAL      3.294    -0.02  +58.9%    +79.1%    1,477.6     414    ₹611,726      +4.2%     
  ADANIENSOL  12     ENERGY     2.823    0.09   +75.9%    +59.0%    1,521.4     402    ₹611,603      +1.7%     
  BHEL        13     ENERGY     2.839    0.01   +56.9%    +64.4%    414.0       1480   ₹612,646      +3.0%     
  ADANIGREEN  14     ENERGY     2.775    -0.03  +54.7%    +80.2%    1,535.0     399    ₹612,465      +2.9%     
  HONASA      15     CONSUMP    2.797    0.39   +50.0%    +56.4%    466.5       1313   ₹612,514      +11.1%    
  CAPLIPOINT  16     HEALTH     2.687    -0.26  +22.0%    +70.7%    2,564.0     239    ₹612,796      +7.0%     
  IDEA        17     CONSUMP    2.728    0.02   +102.2%   +69.4%    14.6        41866  ₹612,918      +2.1%     
  MAHABANK    18     PSU BNK    2.669    -0.04  +74.6%    +42.7%    91.2        6720   ₹612,864      +4.3%     
  POLYCAB     19     MFG        2.595    -0.19  +50.9%    +40.9%    9,713.5     63     ₹611,950      +0.6%     
  OFSS        20     IT         2.575    -0.19  +25.0%    +62.3%    10,863.0    56     ₹608,328      +6.9%     
  VIJAYA      21     HEALTH     2.572    0.22   +42.1%    +54.9%    1,369.9     447    ₹612,345      +3.9%     
  THERMAX     22     ENERGY     2.537    0.18   +49.3%    +53.1%    5,100.5     120    ₹612,060      +5.9%     
  ADANIENT    26     METAL      2.550    -0.01  +21.6%    +70.7%    3,143.6     194    ₹609,858      +5.4%     

  HOLDS (6)
  Ticker      Rank   Sector     Since       Entry₹      Now₹        Qty    Unreal P&L    P&L%        Ext%EMA20 
  ──────────  ─────  ─────────  ──────────  ──────────  ──────────  ─────  ────────────  ──────────  ──────────
  ANANDRATHI  11     FIN SVC    01-Aug-25   1,295.5     1,964.9     470    ₹314,621      +51.7%      +5.0%     
  FEDERALBNK  23     PVT BNK    03-Nov-25   237.9       331.4       2517   ₹235,365      +39.3%      +4.2%     
  CUMMINSIND  24     INFRA      03-Nov-25   4,359.9     5,663.0     137    ₹178,528      +29.9%      +0.1%     
  AIAENG      25     MFG        01-Dec-25   3,854.1     4,996.0     155    ₹176,994      +29.6%      +6.4%     
  NAVINFLUOR  46     MFG        03-Nov-25   5,892.1     7,585.0     101    ₹170,983      +28.7%      +2.5%     
  NYKAA       41     CONSUMP    01-Oct-25   241.3       308.6       2454   ₹165,375      +27.9%      +5.9%     

  AFTER: Invested ₹18,293,384 | Cash ₹78,253 | Total ₹18,371,637 | Positions 28/30 | Slot ₹612,921

========================================================================
  FINAL SUMMARY
========================================================================
  Period        : 2015-01-01 → 2026-07-01  (11.5 years)
  Starting Cap  : ₹2,000,000
  Final Value   : ₹18,371,637
  Total Return  : +818.6%  (on total invested)
  CAGR          : +21.3%

  Closed Trades : 628  |  Open: 28
  Win Rate      : 49.7%  (312W / 316L)
  Profit Factor : 2.25
  Avg hold      : 154 days
  Total charges : ₹833,610
  Closed net P&L: ₹14,891,595
  Open unreal   : ₹1,241,866

  YEAR-BY-YEAR:
  2015  -  1.8%  ░
  2016  +  0.1%  
  2017  + 19.2%  ███████████████████
  2018  -  3.7%  ░░░
  2019  +  6.8%  ██████
  2020  + 97.8%  ████████████████████████████████████████
  2021  + 78.8%  ████████████████████████████████████████
  2022  + 28.2%  ████████████████████████████
  2023  + 49.6%  ████████████████████████████████████████
  2024  + 22.4%  ██████████████████████
  2025  - 10.3%  ░░░░░░░░░░
  2026  +  2.6%  ██

  Rebalance NAV exported → mq500_rebal.csv (139 rows)
