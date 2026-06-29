"""Canonical AI ecosystem universe — single source of truth.

Shared by ai_universe_backtest.py (pure backtest) and ui/app.py (live screener).
92 stocks · 18 sectors · max 2/sector in portfolio.

Filter rule: >$5B market cap. Borderline / QQQ-only stocks live in
app.py's TECHMO_UNIVERSE extension, not here.
"""

AI_UNIVERSE = {

    # ── 1. Compute Silicon / AI Accelerators ─────────────────────────────────
    "NVDA":  "Compute Silicon",    # Nvidia — GPUs, AI accelerators
    "AMD":   "Compute Silicon",    # AMD — GPUs, CPUs
    "AVGO":  "Compute Silicon",    # Broadcom — AI ASICs, networking chips
    "INTC":  "Compute Silicon",    # Intel — CPUs, Gaudi AI accelerator
    "ARM":   "Compute Silicon",    # Arm Holdings — IP architecture licensor
    "MRVL":  "Compute Silicon",    # Marvell — custom AI silicon, DPUs
    "SNPS":  "Compute Silicon",    # Synopsys — EDA tools (chip design software)

    # ── 2. Memory & Storage ───────────────────────────────────────────────────
    "MU":    "Memory & Storage",   # Micron — HBM3E, DRAM, NAND
    "WDC":   "Memory & Storage",   # Western Digital
    "SNDK":  "Memory & Storage",   # SanDisk (WDC flash spinoff)
    "STX":   "Memory & Storage",   # Seagate — enterprise HDDs
    "NTAP":  "Memory & Storage",   # NetApp — storage software

    # ── 3. Semiconductor Equipment ────────────────────────────────────────────
    "ASML":  "Semicon Equipment",  # ASML — EUV/DUV lithography
    "LRCX":  "Semicon Equipment",  # Lam Research — etch & deposition
    "KLAC":  "Semicon Equipment",  # KLA Corporation — process control
    "KEYS":  "Semicon Equipment",  # Keysight — test & measurement
    "CAMT":  "Semicon Equipment",  # Camtek — advanced packaging inspection
    "AMAT":  "Semicon Equipment",  # Applied Materials — deposition, etch

    # ── 4. Packaging & Foundry ────────────────────────────────────────────────
    "TSM":   "Packaging & Foundry", # TSMC — leading-edge foundry, CoWoS
    "ASX":   "Packaging & Foundry", # ASE Technology — OSAT
    "AMKR":  "Packaging & Foundry", # Amkor Technology — OSAT

    # ── 5. Photonics / Optical ────────────────────────────────────────────────
    "COHR":  "Photonics / Optical", # Coherent Corp — optical transceivers
    "LITE":  "Photonics / Optical", # Lumentum — optical components
    "GLW":   "Photonics / Optical", # Corning — fiber optic cable
    "FN":    "Photonics / Optical", # Fabrinet — optical module contract mfg
    "NOK":   "Photonics / Optical", # Nokia — optical networking
    "CIEN":  "Photonics / Optical", # Ciena — optical networking
    "AAOI":  "Photonics / Optical", # Applied Optoelectronics — transceivers

    # ── 6. Networking & Connectivity ─────────────────────────────────────────
    "ANET":  "Networking",         # Arista Networks — data center switching
    "CSCO":  "Networking",         # Cisco Systems
    "CRDO":  "Networking",         # Credo Technology — high-speed SerDes
    "APH":   "Networking",         # Amphenol — connectors & interconnects

    # ── 7. Server OEMs & Contract Manufacturing ───────────────────────────────
    "SMCI":  "Server OEMs",        # Super Micro Computer — GPU servers
    "DELL":  "Server OEMs",        # Dell Technologies
    "HPE":   "Server OEMs",        # Hewlett Packard Enterprise
    "JBL":   "Server OEMs",        # Jabil — EMS / contract manufacturing
    "FLEX":  "Server OEMs",        # Flex Ltd — contract electronics

    # ── 8. AI Data Centers / Neoclouds ───────────────────────────────────────
    "CRWV":  "AI Neoclouds",       # CoreWeave — dedicated AI cloud
    "NBIS":  "AI Neoclouds",       # Nebius — AI cloud, ex-Yandex
    "IREN":  "AI Neoclouds",       # Iris Energy — AI compute data centers
    "APLD":  "AI Neoclouds",       # Applied Digital — AI data center hosting
    "WULF":  "AI Neoclouds",       # TeraWulf — AI compute
    "CORZ":  "AI Neoclouds",       # Core Scientific — AI HPC data centers
    "CIFR":  "AI Neoclouds",       # Cipher Mining — AI compute
    "ORCL":  "AI Neoclouds",       # Oracle — OCI cloud, AI infrastructure

    # ── 9. Power & Cooling Infrastructure ────────────────────────────────────
    "VRT":   "Power & Cooling",    # Vertiv — power/thermal mgmt for data centers
    "ETN":   "Power & Cooling",    # Eaton — power management, UPS
    "GEV":   "Power & Cooling",    # GE Vernova — power generation, grid equipment
    "PWR":   "Power & Cooling",    # Quanta Services — electrical infrastructure
    "HUBB":  "Power & Cooling",    # Hubbell — electrical products
    "MOD":   "Power & Cooling",    # Modine Manufacturing — thermal management

    # ── 10. Energy / AI Power Supply ─────────────────────────────────────────
    "CEG":   "Energy / AI Power",  # Constellation Energy — nuclear
    "VST":   "Energy / AI Power",  # Vistra — gas + nuclear
    "NEE":   "Energy / AI Power",  # NextEra Energy — wind + solar
    "OKLO":  "Energy / AI Power",  # Oklo — microreactors
    "EQT":   "Energy / AI Power",  # EQT Corporation — natural gas
    "CCJ":   "Energy / AI Power",  # Cameco — uranium supply
    "BWXT":  "Energy / AI Power",  # BWX Technologies — nuclear components

    # ── 11. Power Electronics ─────────────────────────────────────────────────
    "STM":   "Power Electronics",  # STMicroelectronics
    "ADI":   "Power Electronics",  # Analog Devices
    "MPWR":  "Power Electronics",  # Monolithic Power Systems
    "ON":    "Power Electronics",  # ON Semiconductor
    "TXN":   "Power Electronics",  # Texas Instruments — analog/embedded

    # ── 12. Robotics & Autonomy ───────────────────────────────────────────────
    "TSLA":  "Robotics & Autonomy", # Tesla — autonomous vehicles, Optimus
    "PATH":  "Robotics & Autonomy", # UiPath — RPA, enterprise automation
    "SYM":   "Robotics & Autonomy", # Symbotic — warehouse robotics
    "TER":   "Robotics & Autonomy", # Teradyne — industrial robotics + test
    "ISRG":  "Robotics & Autonomy", # Intuitive Surgical — surgical robotics

    # ── 13. Defense & Drones ──────────────────────────────────────────────────
    "KTOS":  "Defense & Drones",   # Kratos Defense — drones, defense AI
    "AVAV":  "Defense & Drones",   # AeroVironment — tactical drones
    "LMT":   "Defense & Drones",   # Lockheed Martin
    "NOC":   "Defense & Drones",   # Northrop Grumman

    # ── 14. Space & Satellites ────────────────────────────────────────────────
    "ASTS":  "Space & Satellites", # AST SpaceMobile — space-based cellular
    "RKLB":  "Space & Satellites", # Rocket Lab — launch vehicles
    "LUNR":  "Space & Satellites", # Intuitive Machines — lunar landers
    "PL":    "Space & Satellites", # Planet Labs — earth observation
    "IRDM":  "Space & Satellites", # Iridium Communications — satellite IoT

    # ── 15. Materials & Metals ────────────────────────────────────────────────
    "MP":    "Materials",          # MP Materials — rare earth mining
    "FCX":   "Materials",          # Freeport-McMoRan — copper
    "AA":    "Materials",          # Alcoa — aluminum
    "TECK":  "Materials",          # Teck Resources — base metals

    # ── 16. Frontier AI Models ────────────────────────────────────────────────
    "MSFT":  "Frontier AI Models", # Microsoft — Azure + OpenAI partnership
    "GOOGL": "Frontier AI Models", # Alphabet — Gemini / DeepMind
    "META":  "Frontier AI Models", # Meta — LLaMA, AI research
    "AMZN":  "Frontier AI Models", # Amazon — AWS Bedrock + AGI lab

    # ── 17. Enterprise AI Software ────────────────────────────────────────────
    "SNOW":  "Enterprise AI Software", # Snowflake — AI data cloud, Cortex AI
    "NOW":   "Enterprise AI Software", # ServiceNow — AI workflow automation
    "CRM":   "Enterprise AI Software", # Salesforce — Einstein AI, Agentforce
    "PLTR":  "Enterprise AI Software", # Palantir — AI platform, AIP
    "DDOG":  "Enterprise AI Software", # Datadog — AI observability
    "NET":   "Enterprise AI Software", # Cloudflare — AI inference edge

    # ── 18. Quantum Computing ─────────────────────────────────────────────────
    "IONQ":  "Quantum Computing",  # IonQ — trapped-ion quantum computers
}

SECTOR_ORDER = [
    "Compute Silicon", "Memory & Storage", "Semicon Equipment",
    "Packaging & Foundry", "Photonics / Optical", "Networking",
    "Server OEMs", "AI Neoclouds", "Power & Cooling",
    "Energy / AI Power", "Power Electronics", "Robotics & Autonomy",
    "Defense & Drones", "Space & Satellites", "Materials",
    "Frontier AI Models", "Enterprise AI Software", "Quantum Computing",
]

# Abbreviated display names used in the UI screener
SECTOR_ABBREV = {
    "Compute Silicon":        "Compute",
    "Memory & Storage":       "Memory",
    "Semicon Equipment":      "Semicon Equip",
    "Packaging & Foundry":    "Packaging",
    "Photonics / Optical":    "Photonics",
    "Networking":             "Networking",
    "Server OEMs":            "Server OEM",
    "AI Neoclouds":           "Neocloud",
    "Power & Cooling":        "Power/Cool",
    "Energy / AI Power":      "AI Energy",
    "Power Electronics":      "Power Elec",
    "Robotics & Autonomy":    "Robotics",
    "Defense & Drones":       "Defense",
    "Space & Satellites":     "Space",
    "Materials":              "Materials",
    "Frontier AI Models":     "Frontier AI",
    "Enterprise AI Software": "AI Software",
    "Quantum Computing":      "Quantum",
}
