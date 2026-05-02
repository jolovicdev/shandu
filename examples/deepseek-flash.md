# The Global Race for AI Sovereignty: National Strategies, Supply Chains, and the Bifurcating Stack (2023–2026)

## Executive Summary

The global AI landscape is fragmenting along geopolitical lines as nations pursue distinct models of sovereignty—controlling their own data, compute infrastructure, regulatory frameworks, and talent. By early 2026, the evidence shows not a unified global AI ecosystem but **three emerging poles**: the US-led market-driven approach with massive private investment and export controls, the EU’s regulatory-first model with the world’s first comprehensive AI law, and China’s state-directed system combining tight governance with military-civil fusion. A fourth category of “wildcard” nations—the UAE, India, and the UK—are pursuing agile, pro-innovation strategies, each leveraging unique advantages. The semiconductor supply chain remains the single point of vulnerability: **TSMC makes 99% of the world’s AI accelerator chips** [1], and the US export control regime has unintentionally accelerated Chinese self-reliance while failing to halt access to advanced hardware [2]. Regulatory philosophies diverge sharply, from the EU’s binding risk-based rules to the UK’s non-statutory principles, creating costly compliance fragmentation. Long-term forecasts from IDC and Stanford HAI point toward **a bifurcated global AI stack by 2028**, with 60% of multinational firms splitting operations across sovereign zones, tripling integration costs [3]. The next 3–5 years will likely see deepening fragmentation rather than convergence, driven by geopolitics, supply chain dependencies, and incompatible regulatory regimes.

---

## Key Findings

1. **US leads in private AI investment but relies on a fragile hardware supply chain.** In 2024, US private AI investment reached ~$109.1B, 12× China’s $9.3B [4]; by 2025, US investment surged to ~$285.9B against China’s ~$12.4B [5]. Yet the CHIPS Act ($280B authorized) [6] has yet to secure domestic advanced logic manufacturing; TSMC remains the sole source of cutting-edge AI chips, and Taiwan’s government prevents 2nm production abroad [7].

2. **The EU AI Act establishes a global regulatory benchmark but creates compliance costs.** Entering into force August 2024, the Act bans eight unacceptable-risk practices, imposes strict obligations on high-risk systems, and requires transparency for generative AI [8][9]. Its phased timeline pushes most rules to 2026–2027 [10], and its extraterritorial reach is already shaping non-EU AI development.

3. **China’s AI governance is a layered, four-regulation system with the CAC as central enforcer.** Effective 2022–2023, these cover algorithmic recommendation, deep synthesis (mandating real-identity verification and visible labeling of AI-generated content), generative AI broadly, and ethical review [11][12]. The regime prioritizes content control and national security, contrasting with the EU’s rights-based approach.

4. **The semiconductor supply chain is a geopolitical chokepoint.** Nvidia controls >80% of AI training GPUs and held 92% of the discrete GPU market (Q1 2025) [13]. ASML’s EUV lithography gives the West an estimated five-year lead in chipmaking tools; China’s SMIC lags five years behind TSMC [1]. Despite this, Jensen Huang called US export controls “a failure,” noting Nvidia’s China market share dropped from 95% to 50% [2], spurring a domestic push.

5. **The UK’s agile framework offers a counterpoint to EU rigidity.** The March 2023 White Paper adopts five non-statutory principles (safety, transparency, fairness, accountability, contestability) implemented by sector regulators, avoiding binding horizontal legislation [14][15]. This pro-innovation stance aims to attract AI talent and investment, but lacks enforcement teeth.

6. **The UAE and India are emerging as sovereign AI hubs.** The UAE launched the AI Charter (2024), the ACCESS framework, and MGX (>$100B in AI assets); the Stargate UAE project (1 GW cluster with G42, OpenAI, NVIDIA) targets Dh335B economic output by 2031, with a goal of becoming the first AI-native government by 2027 [16][17]. India’s AI Mission ($1.14B) funds sovereign LLMs (Sarvam AI) and the Bhashini translation platform [17].

7. **Military AI is accelerating, especially in China.** The PLA is pursuing “intelligentized warfare” with systems like Norinco’s Intelligent Precision Strike System (autonomous drone dispatch and target tracking), the “War Skull” wargaming platform, and the “Aiwu LLM+” for command and control [18][19]. The US DoD operates under Directive 3000.09 for lethal autonomous weapons, but no domestic or international prohibitions on LAWS exist [20].

8. **Regulatory fragmentation is already imposing costs on global firms.** IDC predicts that by 2028, 60% of multinationals will split AI stacks across sovereign zones, tripling integration costs [3]. The Stanford 2026 AI Index notes that foundation model transparency declined from 58 to 40 out of 100, and 80 of 95 notable models released in 2025 had no published training code [5].

9. **The US-China model performance gap has nearly closed.** As of March 2026, the top US model leads the top Chinese competitor by just 2.7% on benchmark comparisons [5]. DeepSeek-R1 briefly matched or surpassed leading US systems in February 2025 [5], signaling that software-side leadership is no longer a given for the US.

---

## Detailed Analysis

### 1. National and Regional Strategies: Four Distinct Models

The six nations under review fall into four broad strategic archetypes. The table below compares core dimensions.

| Dimension | United States | European Union | China | United Kingdom | UAE | India |
|-----------|---------------|----------------|-------|----------------|-----|-------|
| **Primary strategy** | Market-driven, state-enabled (CHIPS Act, export controls) | Regulatory-first, risk-based | State-directed, military-civil fusion | Pro-innovation, principles-based | Sovereign AI hub, partnership-driven | Compute-access & sovereign LLM development |
| **Key legislation/framework** | CHIPS Act (2022, $280B auth.) [6]; Biden EO on AI (2023, DPA basis) [21] | AI Act (Reg. 2024/1689) [8]; phased implementation to 2027 | Four regulations (2022–2023): Algorithm, Deep Synthesis, GenAI, Ethical Review [11] | AI White Paper (2023); five non-statutory principles [14][15] | AI Charter (2024); ACCESS framework (2024) [16] | IndiaAI Mission (₹10,000 cr / ~$1.14B) [17] |
| **Private AI investment (2024)** | ~$109.1B [4] | ~$12.8B (Europe collectively) [22] | ~$9.3B VC; massive tech capex (Alibaba ~$52B/3yr, ByteDance ~$20B/yr) [23] | ~$4.5B [4] | UAE MGX fund >$100B; Stargate 1GW cluster [16] | $679.8M Q1 2026 [24] |
| **Notable AI models (2024)** | 40 [4] | 3 [4] | 15 [4] | N/A | N/A | Sarvam AI (sovereign LLM) [17] |
| **Governance body** | White House AI Council, NIST | European AI Office, AI Board, national authorities [9] | Cyberspace Administration of China (CAC) [11][12] | Sector regulators (MHRA, Ofcom, ICO, etc.) | UAE Council for AI [16] | Ministry of Electronics & IT |
| **Unique positioning** | Largest AI ecosystem, deepest capital markets | First comprehensive AI regulation; GDPR-like global influence | Content control & national security; huge scale of data | Agile, sector-specific; talent per capita 30% higher than US [22] | Sovereign cloud with Microsoft/Core42; first AI-native govt by 2027 [17] | Multilingual & low-cost model deployment; Bhashini platform |

The US model combines massive private capital with selective state intervention: the CHIPS Act aims to onshore semiconductor manufacturing, while the 2023 Executive Order uses the Defense Production Act to mandate safety testing for dual-use foundation models (>10B parameters) [21]. Critics argue this stretches DPA authority designed for tangible supply chain crises [25]. The result is a hybrid of market-led innovation and strategic industrial policy.

The EU model is exclusively regulatory: rather than investing directly in AI infrastructure, the bloc sets rules that aim to become a global standard. The AI Act’s risk categories—unacceptable, high, limited, minimal—create a compliance burden that critics say could stifle startups but advantages large incumbents with legal resources [9]. The AI Office and codes of practice for general-purpose AI are still under development as of mid-2025.

China’s state-directed model operates through central planning and party oversight. The New Generation AI Development Plan (2017) set goals of global AI leadership by 2030 and a top-tier military by 2049 [26]. The four AI regulations issued 2022–2023 impose pre-market security assessments, real-identity verification for deep synthesis users, and explicit prohibitions on content that harms national security [12]. The CAC, with powers to order suspension of algorithm updates, acts as a powerful gatekeeper.

The UK’s approach is deliberately non-statutory and sector-specific. The 2023 White Paper established five cross-cutting principles but tasked existing regulators (e.g., MHRA for medical devices, Ofcom for communications) with implementation, avoiding new legislation [15]. A consultation closed in June 2023, and the government has signaled it will “wait and see” before moving to statutory rules. This flexibility contrasts sharply with the EU and is designed to retain London’s position as a hub for AI talent and venture capital.

The UAE has positioned itself as a neutral AI hub through massive capital deployment and multi-jurisdictional partnerships. MGX, backed by Mubadala and G42, manages >$100B in AI assets; the Stargate project (1 GW) brings together G42, OpenAI, and Nvidia [16]. The country’s AI Charter and ACCESS framework signal a desire to be seen as ethically governed while attracting business. India’s approach is more modest: a $1.14B mission to build sovereign compute infrastructure and multilingual models (Sarvam AI, Bhashini) that serve its diverse linguistic population [17].

---

### 2. Supply Chain & Hardware Leverage: Who Controls Compute

The concentration of AI compute in a few hands creates both vulnerability and weaponization opportunity. The table below summarizes the key choke points.

| Asset | Dominant Player | Market Share / Dependence | Geopolitical Leverage |
|-------|----------------|--------------------------|------------------------|
| **Advanced logic foundry** | TSMC (Taiwan) | ~99% of AI accelerator chips [1] | Taiwan’s government prohibits 2nm production abroad [7]; single point of failure for all AI progress |
| **AI training GPUs** | Nvidia | >80% of AI training market; 92% discrete GPU market Q1 2025 [13] | US export controls limit China access; Nvidia’s China share dropped from 95% to 50% [2] |
| **EUV lithography** | ASML (Netherlands) | Sole supplier of EUV machines for ≤5nm nodes | Western lead “so vast” China has little chance to close gap within five years if restrictions hold [1] |
| **Chinese foundry** | SMIC | Most advanced chips on par with TSMC five years ago [1] | Limited by lack of EUV tools; progress dependent on domestic tooling breakthroughs |
| **US onshoring attempts** | TSMC Arizona, Intel Ohio | TSMC Phoenix delayed; Intel Ohio paused | $280B CHIPS Act has not yet produced advanced logic on US soil [6] |

The export control regime has had mixed effects. The US Department of Commerce ordered TSMC to halt 7nm+ shipments to Chinese clients targeting Huawei, and also restricted AI chip sales to China through a “diffusion rule” [1][7]. However, Nvidia CEO Jensen Huang declared the controls “a failure” at Computex 2025, noting that Chinese firms turned to domestic designers like Huawei and are aggressively building a self-reliant supply chain [2]. A thriving black market and transshipment through Southeast Asia and the Gulf have undermined the policy.

China’s counter-leverage includes its massive market (Nvidia designs chips compliant with Chinese regulations to stay competitive), control over rare earths and critical minerals, and geographic proximity to Taiwan [1]. Beijing launched antitrust investigations into Nvidia and Intel and banned Micron chips from sensitive sectors. However, using mineral supply as a weapon risks self-harm due to China’s export dependence.

For the wildcard nations, supply chain strategy focuses on access rather than independence. The UAE invested in MGX and Stargate to secure GPU capacity via partnerships with Nvidia and OpenAI [16]. India’s AI Mission includes plans to procure high-end GPUs through diplomatic channels. The UK relies on proximity to US hyperscalers. None of these three is building domestic fabs.

---

### 3. Regulatory Philosophy: Safety, Copyright, Open-Source, and Frontier Oversight

The table below compares how each jurisdiction handles key regulatory dimensions. Note that evidence is densest for the EU and China; the UK and wildcard nations have less formalized or non-binding frameworks.

| Dimension | United States | EU (AI Act) | China | UK | UAE / India |
|-----------|---------------|-------------|-------|-----|-------------|
| **Safety testing** | Mandatory red-team reporting for dual-use foundation models (>10B params) via DPA [21] | High-risk systems require risk management, data governance, human oversight [9] | Pre-market security assessments for high-risk functions (e.g., face/voice editing) [12] | Non-statutory principle of safety/security/robustness [14] | No formal safety testing mandate; ethical principles (UAE Charter) [16] |
| **Copyright** | No federal AI copyright law; lawsuits pending (NYT v. OpenAI, artist class actions) [27] | Art. 53 mandates copyright compliance policy and training data summary; Recital 105 notes authorization requirement [28] — **but** AIPPI 2024 panel stated “EU AI Act contains nothing about copyright” [29] | Deep Synthesis Reg. protects training data (personal/biometric); no specific copyright provision [12] | UK Copyright Act (S.9(3)) assigns authorship to person making arrangements; UKIPO reviewing for AI [27] | No specific provisions |
| **Open-source treatment** | No formal exemption; DPA requirements apply regardless of license | Art. 2(12) exempts FOSS AI unless high-risk/prohibited; GPAI exception if params public and training <10^25 FLOPs [30] | No open-source carve-out; all providers must comply with regulations [11] | Non-statutory; sector regulators apply principles proportionally | No specific provisions |
| **Liability** | No federal AI liability framework; product liability and tort law apply | High-risk systems subject to strict obligations; liability for safety defects tied to product liability directive amendments | Regulations impose penalties on providers, supporters, users [11] | Common law liability; no AI-specific legislation | Not yet codified |
| **Frontier model oversight** | EO requires reporting for dual-use models; NIST to develop red-teaming standards [21] | GPAI with systemic risk (>10^25 FLOPs) subject to codes of practice, oversight by European AI Office [9][30] | No explicit frontier model category; all generative AI regulated under GenAI Reg. [11] | No frontier-specific oversight | None |

The EU AI Act contains a notable internal contradiction on copyright. The IAPP analysis confirms that **Article 53 obliges GPAI providers to implement a copyright compliance policy** and publish a detailed summary of training data [28]. Yet a 2024 AIPPI conference report states explicitly that the EU AI Act “contains nothing about copyright” [29]. This discrepancy may stem from the fact that Article 53 imposes a procedural transparency obligation—to have a policy and publish a summary—rather than substantive copyright rules. The actual copyright liability for training data flows from existing EU directives (Copyright Directive, DSM Directive), not the AI Act itself. The AI Act effectively requires providers to demonstrate good faith compliance, but does not create new copyright exceptions or liability rules.

China’s regulations are the most prescriptive on content control. The Deep Synthesis Regulation mandates **real-identity verification for publishers**, visible technical marks on all AI-generated content, training data protection (especially personal and biometric data), and mandatory security assessments for functions that edit faces or voices [12]. The regulation prohibits creating or disseminating information that violates laws, including false news and content harming national security. The Generative AI Regulation (effective August 2023) extends these rules broadly to all AI technologies providing services in China [11]. Enforcement is multi-agency, with the CAC as lead.

The UK’s approach—outcome rather than rule-based—is the lightest touch of any major economy. The five principles are not legally binding, and regulators apply them according to sector-specific context. The government has stated that a “heavy-handed and rigid approach can stifle innovation and slow AI adoption” [14]. This framework is designed to be adaptable, but its effectiveness in preventing harm is untested.

---

### 4. Economic and Security Implications

#### Startup Ecosystems and Big Tech Moat Dynamics

The AI startup funding landscape is starkly concentrated. In 2024, global AI VC hit $110B, with the US commanding **74% share** ($81B+), Europe $12.8B (12%), and China ~$9.3B [22]. Generative AI absorbed the bulk of funding, with top rounds including $10B for data/AI infrastructure in the Bay Area and $6.6B for GenAI companies [22]. AI now accounts for one-third of total global VC, more than double its share from two years prior.

Europe’s AI startup ecosystem benefits from a **30% higher per-capita concentration of AI experts among software engineers** than the US, and nearly three times that of China; London leads in absolute AI engineer count [22]. However, European startups face a fragmented regulatory landscape and smaller domestic markets. The UK’s pro-innovation stance may help London retain talent, but the Brexit-induced separation from EU markets complicates scaling.

India’s AI startup funding surged to $679.8M in Q1 2026, the highest quarterly total [24], reflecting a shift from experimentation to deployment. Indian AI startups focus on cost-efficient models and vernacular language services. The UAE’s AI ecosystem is less startup-driven and more top-down, leveraging sovereign wealth funds to attract global giants; the $100B MGX fund dwarfs local VC.

Big Tech moats are widening. US hyperscalers (Microsoft, Amazon, Google, Meta, Oracle) planned ~$450B in AI-specific capex in 2026, financing via debt (Big Five raised $108B in 2025) [31]. These investments create barriers to entry for smaller competitors. Nvidia’s GPU dominance remains entrenched; cloud providers tie GPU access to their own AI services (e.g., Microsoft’s Azure OpenAI Service). Foundation model competition is also consolidating: the top few US labs (OpenAI, Anthropic, Google DeepMind) absorb majority funding.

#### Military and Defense AI Applications

The military dimension of AI sovereignty is most advanced in the US and China. The US DoD’s Data, Analytics, and AI Adoption Strategy (November 2023) and the Political Declaration on Responsible Military Use of AI (endorsed by 47 states as of February 2024) emphasize ethical principles, legal review, and senior oversight [32]. The CRS report notes that narrow AI is already used for ISR, logistics, cyber operations, and semi-autonomous vehicles; however, no domestic or international prohibitions on lethal autonomous weapon systems (LAWS) exist, and DoD Directive 3000.09 governs compliance with law of war [20].

China’s military AI push is more aggressive and less constrained by public accountability. The PLA’s “intelligentized warfare” doctrine aims for cognitive dominance through integration of physical, virtual, and cognitive domains [19]. Concrete systems include Norinco’s **Intelligent Precision Strike System** (autonomously dispatches drones, tracks targets, assigns strikes—human only required for firing authorization) [18], the “War Skull” second-generation wargaming platform, and the “Aiwu LLM+” developed by the People’s Armed Police Engineering University for command and control [18]. Under military-civil fusion, civilian firms like U-Tenet contribute tools such as the Tianji (decision-making brain) and Tianwang (real-time repository) trained on over a million documents and 300 TB of military imagery [18]. China also explores the “battleverse” concept—a persistent virtual-real fused environment for operations [19].

For other nations, military AI data is sparse. The UAE and India are developing AI for defense but have not disclosed specific programs. The UK’s military AI strategy is integrated with the US and NATO.

#### Data Sovereignty Requirements

The push for data sovereignty—keeping data within national borders—is reshaping cloud and AI architecture. The EU’s GDPR already required data localization for personal data, and the AI Act adds layers of compliance. The IDC forecast notes that **63% of organizations are more likely to adopt sovereign cloud services** due to recent geopolitical events [3]. AWS launched the AWS European Sovereign Cloud in Germany in January 2026, with €7.8B planned investment through 2040, featuring physically and logically separate infrastructure [3].

China’s Data Security Law and Personal Information Protection Law (2021) impose strict data localization requirements, with mandatory security assessments for cross-border data transfers. The AI regulations reinforce these rules for training data. For UAE, the sovereign cloud partnership between Microsoft and Core42 aims to create a fully AI-native government by 2027 with data residing in UAE [17]. India’s upcoming Digital Personal Data Protection Act (2023) requires consent and data localization for sensitive data, affecting how AI models trained abroad can be deployed in India.

---

### 5. Synthesis and Forecast: Where the Global AI Stack Is Heading

#### The Evidence for Fragmentation

Three credible forecasting streams point toward fragmentation rather than convergence by 2028–2030.

**IDC FutureScape 2026** predicts that by 2028, **60% of multinational firms will split AI stacks across sovereign zones**, tripling integration costs [3]. Two layers of fragmentation are emerging: infrastructure (hardware segmentation by region) and platform (distinct “East vs. West” AI stacks). The AWS European Sovereign Cloud is a concrete signal.

**Stabilarity Hub analysis** argues that the global AI infrastructure stack is bifurcating into a Western stack (US hyperscalers—Microsoft, AWS, Google, Meta, Oracle—investing $450B in AI-specific capex in 2026) and a Chinese stack (Alibaba, Huawei) built on Chinese governance norms [31]. The Atlantic Council’s January 2026 analysis cited therein identifies control over compute power, cloud storage, microchips, and regulation as the defining axes of AI competition.

**Stanford HAI AI Index 2026** does not offer an explicit forecast, but its data strongly implies fragmentation: governance approaches are diverging, foundation model transparency is declining, and the performance gap between US and Chinese models has narrowed to just **2.7%**, meaning Chinese models are now competitive globally [5]. However, the supply chain remains concentrated in Taiwan, creating a single point of failure.

The Gartner forecast (October 2025) notes that achieving AI sovereignty requires “decision-making authority across the entire AI stack with geographical constraints,” and that regulatory compliance leads to increased operational costs and interoperability challenges [33].

#### Convergence Forces

Countervailing forces suggest some convergence. Hardware dependencies (TSMC, ASML, Nvidia) are global by nature; no nation can fully decouple. Open-source models like DeepSeek-R1 and Meta’s Llama provide a common substrate. The EU AI Act may become a global baseline, as GDPR did for privacy—several Latin American and African countries have already cited it [10]. The Political Declaration on Responsible Military Use of AI, endorsed by 47 states, shows limited normative convergence.

#### The Likely Outcome (2026–2029)

The balance of evidence favors **deep fragmentation with limited interoperability**. The AI stack appears to be splitting into two broad ecosystems: one centered on US hyperscalers and Western regulatory norms (with EU compliance layers), and another centered on Chinese infrastructure and Soviet-style state control. The UAE, India, and the UK represent middle-ground nodes that can bridge both ecosystems but may be forced to choose if decoupling intensifies. A pure “full convergence” scenario—one global AI market with common rules—is improbable given current geopolitical trajectories. A “full bifurcation” (completely separate stacks with no interoperability) is also unlikely due to hardware and open-source interdependencies. The most plausible scenario is **controlled fragmentation**: sovereign zones with localized data, models, and governance, linked by thin interoperability layers (e.g., open-weight models, standardized APIs) but incurring high integration costs.

---

## Risks & Counterpoints

Several important caveats and uncertainties temper the analysis.

**Contradiction on EU AI Act copyright provisions.** The IAPP source [28] states Article 53 requires a copyright compliance policy and training data summary; the AIPPI conference report [29] states the AI Act “contains nothing about copyright.” This likely reflects a nuanced reality: the Act mandates a *policy and summary* rather than substantive copyright rules. Until the final Code of Practice is published and court interpretations emerge, the exact burden remains uncertain. This contradiction means confidence on EU copyright treatment is moderate at best.

**Weak evidence for several dimensions.** Military AI and data sovereignty regulations for the UAE, India, and the UK are poorly documented in the evidence corpus. For the UK, only general white paper principles are available; for India, no specific military AI programs were captured. The forecast section relies heavily on an IDC blog post (though IDC is a reputable firm) and a self-published Stabilarity analysis, both of which lack the rigor of peer-reviewed think tank reports. Stanford HAI offers trend data but stops short of explicit future scenarios.

**The “failure” of US export controls is contested.** Jensen Huang’s statement [2] reflects Nvidia’s commercial interest in the China market. Other analysis suggests that while China has been pushed to develop domestic alternatives, those alternatives still lag five years behind and face tooling shortages [1]. The effectiveness of controls cannot be judged by market share alone; the long-term drag on China’s capabilities may be significant even if short-term evasion occurs.

**Open-source dynamics are ambiguous.** The EU AI Act’s open-source exemptions (Art. 2(12) and GPAI threshold of 10^25 FLOPs [30]) could be a double-edged sword: they protect small innovators but may allow large frontier labs to structure releases as “open source” while maintaining control. China’s regulations impose obligations on all providers regardless of license, potentially dampening open-source adoption. The UK’s non-statutory approach provides the most freedom but the least legal certainty.

**AI incidents are rising faster than governance adaptation.** The Stanford Index documents a 55% increase in AI incidents (from 233 in 2024 to 362 in 2025) [5], while foundation model transparency dropped from 58 to 40 out of 100 [5]. This widening oversight gap suggests that regulatory frameworks, however designed, are struggling to keep pace with deployment.

**Military AI data is US- and China-heavy.** The evidence does not cover EU member states’ military AI (e.g., France’s defense AI strategy, Germany’s investments), nor detailed programs for the UAE, India, or UK beyond general policy statements. This limits the comparative depth of the security analysis.

**Big Tech moats are assumed but not directly measured.** The evidence provides aggregate capex and market share figures (Nvidia 92% GPU, TSMC 99% AI foundry) but no systematic comparison of cloud market share by region (e.g., AWS vs. Alibaba Cloud) or foundation model API revenue. Thus the “moat” discussion is inferred from investment data rather than directly measured.

---

## Open Questions

1. **How will open-source AI models affect sovereignty dynamics?** The EU AI Act’s 10^25 FLOPs threshold exempts many open-weight models, but frontier labs could design models to stay just below the threshold, or conversely, open-sourcing critical models could democratize access. The trajectory is unclear.

2. **Will India and the UAE remain neutral bridges, or be forced to align?** Both nations currently maintain partnerships with US and Chinese tech giants. As the US-China technology war deepens, they may face pressure to choose sides, particularly over data sovereignty and hardware procurement.

3. **What are the exact liability mechanisms for frontier models under the EU AI Act?** The Act assigns obligations to providers of high-risk and GPAI models, but the liability for downstream harm (e.g., misinformation, bias, safety failures) is still untested. The interplay with the EU’s proposed AI Liability Directive remains under negotiation.

4. **Can any nation break TSMC’s dominance in advanced logic?** The US CHIPS Act has not yet yielded advanced fabs; Intel’s foundry business is struggling; Japanese and European initiatives (Rapidus, joint ventures) are nascent. Without a breakthrough, all AI sovereignty depends on Taiwan’s stability.

5. **How will the US political transition affect regulatory coherence?** The 2024 election resulted in a new administration whose stance on AI regulation, export controls, and international cooperation may differ from the Biden approach. The Mercatus Center brief suggests previous EOs may be repealed [25]; uncertainty remains.

6. **What data sovereignty regulations will the UAE and India actually enforce?** Both countries have announced ambitious AI plans but have not yet passed comprehensive data protection laws with extraterritorial scope comparable to GDPR or China’s PIPL. Their actual enforcement posture is untested.

7. **Are alternative compute architectures (e.g., neuromorphic, optical, quantum) likely to disrupt the semiconductor bottleneck within 5 years?** No evidence addresses this. Current roadmaps suggest they are at least 5–10 years from material impact.

---

The global AI sovereignty race is not a single contest but a set of overlapping competitions over compute, data, talent, norms, and military advantage. The next 3–5 years will likely see the current fragmentation harden into two semi-interoperable stacks, with high costs for straddling both sides. Nations will face increasing pressure to choose their alliances, and the design of regulatory frameworks will become as strategically important as the design of chips. The United States retains advantages in capital, talent, and model performance, but the gap with China is closing faster than many anticipated. The EU’s regulatory approach may become a global template but at the cost of slower innovation. The wildcard nations—UAE, India, UK—have the most flexibility but also the most to lose if the stacks fully separate. The semiconductor supply chain, rooted in a single island, remains the ultimate vulnerability for both camps.

## References

[1] www.axios.com. "Biden ratchets up AI chip war with China". https://www.axios.com/2024/12/18/china-ai-chip-export-controls (accessed 2026-05-02)
[2] international.astroawani.com. "Nvidia says US export controls on AI chips to China were 'a failure'". https://international.astroawani.com/global-news/nvidia-says-us-export-controls-ai-chips-china-were-failure-521683 (accessed 2026-05-02)
[3] www.idc.com. "The high cost of sovereignty in the age of AI". https://www.idc.com/resource-center/blog/the-high-cost-of-sovereignty-in-the-age-of-ai/ (accessed 2026-05-02)
[4] hai.stanford.edu. "The 2025 AI Index Report". https://hai.stanford.edu/ai-index/2025-ai-index-report (accessed 2026-05-02)
[5] complexdiscovery.com. "Stanford’s 2026 AI Index highlights rapid growth and widening governance gaps". https://complexdiscovery.com/stanfords-2026-ai-index-highlights-rapid-growth-and-widening-governance-gaps/ (accessed 2026-05-02)
[6] en.wikipedia.org. "CHIPS and Science Act - Wikipedia". https://en.wikipedia.org/wiki/CHIPS_and_Science_Act (accessed 2026-05-02)
[7] www.linkedin.com. "US Tightens Export Controls on AI Chips to China". https://www.linkedin.com/pulse/us-tightens-export-controls-ai-chips-china-dusan-simic-urqvf (accessed 2026-05-02)
[8] digital-strategy.ec.europa.eu. "AI Act". https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai (accessed 2026-05-02)
[9] regulations.ai. "Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence and amending certain Union legislative acts". https://regulations.ai/regulations/european-union-2024-8-ai-act (accessed 2026-05-02)
[10] artificialintelligenceact.eu. "The EU Artificial Intelligence Act". https://artificialintelligenceact.eu/ (accessed 2026-05-02)
[11] www.lw.com. "China’s New AI Regulations". https://www.lw.com/admin/upload/SiteAttachments/Chinas-New-AI-Regulations.pdf (accessed 2026-05-02)
[12] regulations.ai. "Provisions on the Administration of Deep Synthesis of Internet-based Information Services". https://regulations.ai/regulations/china-2022-11-deep-synthesis (accessed 2026-05-02)
[13] en.wikipedia.org. "Nvidia - Wikipedia". https://en.wikipedia.org/wiki/Nvidia (accessed 2026-05-02)
[14] www.gov.uk. "A pro-innovation approach to AI regulation". https://www.gov.uk/government/publications/ai-regulation-a-pro-innovation-approach/white-paper (accessed 2026-05-02)
[15] www.braithwate.com. "UK Government White Paper on AI: “A pro-innovation approach to AI regulation” — Braithwate - Specialist Advisors in Financial Services". https://www.braithwate.com/insights/uk-pro-innovation-approach-to-ai-regulation (accessed 2026-05-02)
[16] en.aletihad.ae. "Behind UAE’s AI growth is a governance model built to safeguard ethics, public trust: NEP expert". https://en.aletihad.ae/news/uae/4654783/behind-uae-s-ai-growth-is-a-governance-model-built-to-safegu (accessed 2026-05-02)
[17] arxiv.org. "Sovereign AI: Rethinking Autonomy in the Age of Global Interdependence". https://arxiv.org/html/2511.15734v1 (accessed 2026-05-02)
[18] www.defenseone.com. "New products show China’s quest to automate battle". https://www.defenseone.com/threats/2025/03/new-products-show-chinas-quest-automate-battle/403387/ (accessed 2026-05-02)
[19] blog.roninsgrips.com. "Enter the Battleverse: China's Pursuit of Intelligentized Warfare in the Metaverse - Ronin's Grips". https://blog.roninsgrips.com/enter-the-battleverse-chinas-pursuit-of-intelligentized-warfare-in-the-metaverse/ (accessed 2026-05-02)
[20] www.congress.gov. "https://crsreports.congress.gov". https://www.congress.gov/crs_external_products/IF/PDF/IF11105/IF11105.12.pdf (accessed 2026-05-02)
[21] www.pillsburylaw.com. "Executive Order on Safe, Secure, Trustworthy Artificial Intelligence". https://www.pillsburylaw.com/en/news-and-insights/biden-executive-order-safe-ai.html (accessed 2026-05-02)
[22] www.techmonitor.ai. "Global AI venture capital reaches $110bn in 2024, driven by foundational models". https://www.techmonitor.ai/digital-economy/ai-and-automation/global-ai-venture-capital-110bn-2024-driven-foundational-models (accessed 2026-05-02)
[23] www.yicaiglobal.com. "RELATED". https://www.yicaiglobal.com/news/chinas-vc-investment-in-ai-lags-far-behind-us-qiming-managing-partner-says (accessed 2026-05-02)
[24] analyticsindiamag.com. "With $680 Mn in Q1, Indian AI Startups ... | Analytics India Magazine". https://analyticsindiamag.com/ai-startups/with-680-mn-in-q1-indian-ai-startups-have-found-their-mojo-but-for-how-long (accessed 2026-05-02)
[25] www.mercatus.org. "Executive Orders on AI: How to (Lawfully) Apply the Defense Production Act". https://www.mercatus.org/research/policy-briefs/executive-orders-ai-how-lawfully-apply-defense-production-act (accessed 2026-05-02)
[26] thechinabriefing.substack.com. "Briefing: China's Growing Use of Artificial Intelligence in Millitary Applications". https://thechinabriefing.substack.com/p/briefing-chinas-growing-use-of-artificial (accessed 2026-05-02)
[27] kandspartners.com. "Mitigating liability while copyright law catches up with Artificial Intelligence". https://kandspartners.com/mitigating-liability-while-copyright-law-catches-up-with-artificial-intelligence-2/ (accessed 2026-05-02)
[28] iapp.org. "The EU AI Act and copyrights compliance | IAPP". https://iapp.org/news/a/the-eu-ai-act-and-copyrights-compliance (accessed 2026-05-02)
[29] www.asiaiplaw.com. "AIPPI 2024: The copyright dilemma: Trained to infringe?". https://www.asiaiplaw.com/sector/copyright/aippi-2024-the-copyright-dilemma-trained-to-infringe (accessed 2026-05-02)
[30] www.jdsupra.com. "The EU AI Act: Open-Source Exceptions and Considerations for Your AI Strategy | JD Supra". https://www.jdsupra.com/legalnews/the-eu-ai-act-open-source-exceptions-9085314/ (accessed 2026-05-02)
[31] hub.stabilarity.com. "Tech Cold War 2026 — Microsoft, AWS, and the Geopolitics of AI Infrastructure - Stabilarity Hub". https://hub.stabilarity.com/tech-cold-war-2026-microsoft-aws-and-the-geopolitics-of-ai-infrastructure/ (accessed 2026-05-02)
[32] defense.info. "The U.S. Department of Defense and AI | Defense.info". https://defense.info/defense-decisions/2024/01/the-u-s-department-of-defense-and-ai/ (accessed 2026-05-02)
[33] www.gartner.com. "Predicts 2026: AI Sovereignty". https://www.gartner.com/en/documents/7077898 (accessed 2026-05-02)