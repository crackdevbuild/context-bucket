from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = BENCHMARK_ROOT / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

LEGAL_BUSINESS_DOMAINS = [
    "merger_acquisition", "contract_review", "compliance_audit",
    "regulatory_filing", "ip_patent", "corporate_governance",
    "employment_law", "real_estate", "tax_strategy", "litigation",
    "data_privacy", "antitrust", "banking_finance", "securities",
    "environmental_reg", "trade_secret", "franchise_law",
    "insurance_claim", "bankruptcy", "immigration_law",
]

MEDICAL_DOMAINS = [
    "patient_care", "drug_interaction", "clinical_trial",
    "hipaa_compliance", "medical_device", "radiology",
    "pathology", "pharmacology", "surgical_protocol",
    "emergency_medicine", "mental_health", "vaccination",
    "infection_control", "obstetrics", "pediatrics",
    "oncology", "cardiology", "neurology",
    "rehabilitation", "palliative_care",
]

KINDS = [
    "research_finding", "research_report", "evidence_summary",
    "decision_outcome", "topic_pattern", "user_profile_note",
]

SCOPES = ["session", "user", "global"]

SHORT_TEMPLATES = {
    "merger_acquisition": [
        "ACME Corp acquisition of TargetCo proceeding at 3.2x revenue multiple.",
        "Board approved the merger proposal pending regulatory clearance.",
        "Merger talks between AlphaCo and BetaCorp stalled over valuation gap.",
        "Acquisition of CloudSync completed for $450M cash consideration.",
        "Hostile bid from MegaCorp rejected by TargetBoard unanimously.",
    ],
    "contract_review": [
        "Contract amendment requires legal sign-off before execution.",
        "Service agreement renewal terms updated for FY2026.",
        "Vendor contract dispute escalated to arbitration panel.",
        "Non-compete clause found unenforceable in current jurisdiction.",
        "Master services agreement drafted and sent for client review.",
    ],
    "compliance_audit": [
        "Annual compliance audit identified three material gaps.",
        "SOX controls testing revealed deficiency in access management.",
        "GDPR audit trail requirements met for Q4 reporting period.",
        "Compliance training completion rate dropped to 72 percent.",
        "External audit confirmed full regulatory compliance status.",
    ],
    "regulatory_filing": [
        "SEC 10-K filing deadline extended to March 31.",
        "FDA pre-market approval submitted for medical device class II.",
        "EPA emissions report filed ahead of December deadline.",
        "FTC merger notification filed under HSR Act waiting period.",
        "Quarterly regulatory filing completed with no deficiencies.",
    ],
    "ip_patent": [
        "Patent application filed for proprietary compression algorithm.",
        "Trademark opposition filed against competing brand application.",
        "Trade secret misappropriation claim filed in federal court.",
        "Patent portfolio review identified 12 expiring patents by 2027.",
        "Licensing agreement negotiated for cross-patent technology access.",
    ],
    "corporate_governance": [
        "Board charter amendment proposed to expand committee oversight.",
        "Shareholder proposal for ESG disclosure received majority vote.",
        "Director independence assessment completed per exchange rules.",
        "Proxy statement filed with updated executive compensation data.",
        "Governance framework updated to align with new SEC guidance.",
    ],
    "employment_law": [
        "Non-compete enforcement varies significantly by state jurisdiction.",
        "Employee classification audit reclassified 15 contractors.",
        "Workplace harassment policy updated with new reporting channel.",
        "Mass layoff notice requirements triggered under WARN Act.",
        "Remote work policy must comply with multi-state tax obligations.",
    ],
    "real_estate": [
        "Commercial lease negotiation completed for downtown headquarters.",
        "Zoning variance approved for mixed-use development project.",
        "Environmental site assessment revealed contamination requiring remediation.",
        "Property tax appeal reduced assessed value by 18 percent.",
        "Title insurance claim denied due to undisclosed easement.",
    ],
    "tax_strategy": [
        "Transfer pricing documentation required for intercompany transactions.",
        "R&D tax credit claim filed for $2.3M in qualifying expenses.",
        "State tax nexus analysis triggered by remote employee presence.",
        "International tax treaty benefits applied to cross-border royalty.",
        "Tax loss carryforward utilization strategy for acquired entity.",
    ],
    "litigation": [
        "Class action certification hearing scheduled for next month.",
        "Settlement offer of $12M rejected by plaintiff counsel.",
        "E-discovery production completed with 1.2M documents reviewed.",
        "Summary judgment motion filed on statute of limitations grounds.",
        "Expert witness testimony challenged under Daubert standard.",
    ],
    "data_privacy": [
        "Data breach notification sent to 50,000 affected individuals.",
        "Privacy impact assessment required before system deployment.",
        "Consent management platform updated for CCPA compliance.",
        "Cross-border data transfer mechanism validated under SCCs.",
        "Biometric data retention policy updated per BIPA requirements.",
    ],
    "antitrust": [
        "Market concentration analysis shows HHI above 2500 threshold.",
        "Price-fixing investigation resulted in criminal indictment.",
        "Merger remedy package proposed with behavioral commitments.",
        "Exclusive dealing arrangement reviewed for competitive effects.",
        "Antitrust compliance training rolled out to sales organization.",
    ],
    "banking_finance": [
        "Stress testing results show adequate capital under adverse scenario.",
        "Loan loss provision increased by 15 percent for Q4.",
        "Basel III capital requirements met with 2% buffer above minimum.",
        "Wire transfer monitoring system flagged suspicious activity patterns.",
        "Consumer lending compliance review identified APR disclosure errors.",
    ],
    "securities": [
        "Insider trading surveillance system flagged unusual options activity.",
        "Regulation D private placement exemption relied upon for offering.",
        "Proxy solicitation rules applied to shareholder communication plan.",
        "Registration statement effectiveness delayed by SEC comment letter.",
        "Beneficial ownership reporting updated per new corporate transparency act.",
    ],
    "environmental_reg": [
        "Carbon emission reduction targets set at 40% by 2030.",
        "Hazardous waste disposal permit renewed with stricter conditions.",
        "Environmental impact statement required for pipeline expansion.",
        "Clean Water Act violation resulted in $500K civil penalty.",
        "Sustainability reporting framework aligned with TCFD recommendations.",
    ],
    "trade_secret": [
        "NDA breach allegation filed against departing senior engineer.",
        "Trade secret identification audit cataloged 340 protected assets.",
        "Employee exit interview protocol updated for IP protection.",
        "Inevitable disclosure doctrine invoked in preliminary injunction.",
        "Competitive intelligence gathering must avoid trade secret boundary.",
    ],
    "franchise_law": [
        "Franchise disclosure document updated with 2026 financials.",
        "Territory dispute between franchisees escalated to mediation.",
        "Franchise agreement renewal terms modified with performance standards.",
        "State franchise registration required before offering in new market.",
        "Franchisee default notice issued for operational non-compliance.",
    ],
    "insurance_claim": [
        "Business interruption claim disputed over policy period interpretation.",
        "Directors and officers liability coverage responded to shareholder suit.",
        "Property damage claim adjusted for actual cash value versus replacement.",
        "Cyber insurance policy activated following ransomware incident.",
        "Workers compensation claim frequency analysis reveals upward trend.",
    ],
    "bankruptcy": [
        "Chapter 11 reorganization plan filed with creditor committee support.",
        "Preference action recovery targeted $4.2M in avoidable transfers.",
        "Debtor-in-possession financing approved with super-priority status.",
        "Automatic stay enforcement action filed against foreclosing lender.",
        "Plan confirmation hearing scheduled after objection resolution period.",
    ],
    "immigration_law": [
        "H-1B visa petition filed for senior software engineer position.",
        "PERM labor certification process initiated for permanent residency.",
        "I-9 compliance audit identified documentation gaps in remote hires.",
        "E-2 treaty investor visa approved for $500K capital commitment.",
        "Immigration policy change impacts seasonal worker staffing plan.",
    ],
}

MEDIUM_TEMPLATES = {
    "merger_acquisition": [
        "ACME Corp has entered into a definitive merger agreement with TargetCo at a purchase price of $2.1B, representing a 3.2x revenue multiple. The transaction is subject to regulatory approval under the HSR Act, with an expected closing date in Q3 2026. Due diligence identified key risks in target IP portfolio and outstanding litigation exposure of approximately $45M.",
        "The board of directors unanimously approved the acquisition proposal after extensive review of the fairness opinion provided by independent financial advisor. Key conditions precedent include regulatory clearance, third-party consent for assignment of material contracts, and absence of material adverse change prior to closing.",
        "Merger integration planning has identified 180-day synergy targets including $85M in cost reductions from headcount optimization and $42M in revenue synergies from cross-selling opportunities. Cultural integration risk assessment flagged leadership style differences between organizations.",
    ],
    "contract_review": [
        "The master service agreement with GlobalTech requires amendment to address liability cap reductions from $10M to $5M and updated data protection addendum for GDPR compliance. Legal review identified three non-standard indemnification clauses requiring negotiation. Counterparty has requested acceleration of timeline to close by end of quarter.",
        "Vendor contract portfolio review completed: 47 active agreements analyzed, 12 flagged for renewal within 90 days, 5 with unfavorable auto-renewal provisions, and 3 with missing force majeure clauses. Recommendation to standardize contract template with improved termination for convenience provisions.",
        "Service level agreement dispute with CloudHost Inc. centers on uptime guarantee interpretation. Customer claims 99.99% availability commitment was breached during Q4 outage events. Vendor argues planned maintenance windows are excluded from calculation. Escalation to arbitration per dispute resolution clause.",
    ],
}

LONG_TEMPLATES = {
    "merger_acquisition": [
        "ACME Corporation Merger Integration Plan - Phase 1 Assessment. Executive Summary: The proposed merger between ACME Corporation and TargetCo represents a transformative combination valued at approximately $2.1 billion. This document outlines the comprehensive integration strategy across all functional areas. Financial Integration: Combined revenue of $4.8B with projected EBITDA margins of 22%. Cost synergy target of $85M annually by end of year two, primarily through consolidation of redundant corporate functions and procurement optimization. Revenue synergy estimate of $42M through cross-selling ACME enterprise solutions to TargetCo mid-market customer base. Technology Integration: Both organizations utilize cloud-native architectures, facilitating relatively straightforward system integration. Key challenges include data migration from TargetCo legacy ERP system and harmonization of CRM platforms. Estimated technology integration budget of $18M over 18 months. Human Capital: Combined workforce of 12,400 employees with overlapping functions in sales, marketing, and operations representing approximately 340 positions potentially affected. Retention packages recommended for 85 key personnel across both organizations. Cultural assessment reveals moderate integration risk due to differences in decision-making velocity and organizational hierarchy. Regulatory: HSR Act filing submitted; expected 60-day review period. State-level regulatory approvals required in 4 jurisdictions. International competition clearance needed in EU, UK, and Japan. Risk assessment indicates moderate regulatory risk with potential remedy requirements in EU markets. Timeline: Target closing Q3 2026 with Phase 1 integration activities commencing upon regulatory clearance. Full integration expected within 24 months of closing.",
    ],
}

STRUCTURED_DATA_TEMPLATES = [
    {"subject": "Contract amendment review", "from_name": "Legal Team", "body_text": "Please review the attached contract amendments by end of week. Key changes include updated liability caps and revised indemnification language.", "priority": "high", "deadline": "2026-05-15"},
    {"subject": "Q4 financial results", "from_name": "CFO Office", "body_text": "Quarterly earnings exceeded expectations with 15% revenue growth. Board presentation scheduled for next Tuesday.", "priority": "medium", "deadline": "2026-05-20"},
    {"incident": "Server outage", "severity": "high", "duration_min": 45, "affected_users": 1200, "root_cause": "Database connection pool exhaustion", "resolution": "Increased pool size and added circuit breaker"},
    {"title": "Merger Risk Assessment", "status": "active", "team": {"lead": "Jane Smith", "members": 8, "budget": {"allocated": 500000, "spent": 320000, "currency": "USD"}}, "milestones": [{"phase": "due_diligence", "complete": True}, {"phase": "negotiation", "complete": False}]},
    {"policy_name": "Data Retention Standard", "version": "3.1", "sections": [{"title": "General Principles", "content": "All data must be classified before retention decisions are made.", "mandatory": True}, {"title": "Retention Periods", "content": "Financial records: 7 years. Employee records: 5 years post-separation.", "mandatory": True}]},
    {"project": "ACME Integration", "phase": "planning", "risks": [{"category": "regulatory", "likelihood": "medium", "impact": "high", "mitigation": "Engage external counsel for pre-filing strategy"}, {"category": "technology", "likelihood": "high", "impact": "medium", "mitigation": "Parallel run period before cutover"}]},
    {"deal_name": "TargetCo Acquisition", "valuation": {"enterprise_value": 2100000000, "revenue_multiple": 3.2, "ev_ebitda": 14.5}, "advisors": {"legal": "BigLaw LLP", "financial": "Investment Bank Corp"}, "timeline": {"loi_date": "2026-01-15", "signing_target": "2026-06-30", "closing_target": "2026-09-30"}},
]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _source_key(domain: str, idx: int, prefix: str = "") -> str:
    return f"{prefix}{domain}:{idx:04d}" if prefix else f"{domain}:{idx:04d}"


def _pick_scope(rng: random.Random, idx: int, scope_weights: tuple[float, float, float] | None = None) -> tuple[str, dict]:
    w = scope_weights or (0.40, 0.35, 0.25)
    scope = rng.choices(SCOPES, weights=w, k=1)[0]
    extras: dict = {}
    if scope == "session":
        extras["session_id"] = f"s_{domain_hash(idx)}"
    elif scope == "user":
        extras["user_id"] = f"u_{idx % 20 + 1:02d}"
    return scope, extras


def domain_hash(idx: int) -> str:
    return hashlib.md5(str(idx).encode()).hexdigest()[:6]


def generate_base_dataset(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    idx = 0
    for domain in LEGAL_BUSINESS_DOMAINS:
        short_pool = SHORT_TEMPLATES.get(domain, SHORT_TEMPLATES["merger_acquisition"])
        medium_pool = MEDIUM_TEMPLATES.get(domain, short_pool)
        long_pool = LONG_TEMPLATES.get(domain, short_pool)
        records_per_domain = 1000 // len(LEGAL_BUSINESS_DOMAINS)
        for di in range(records_per_domain):
            scope, extras = _pick_scope(rng, idx)
            kind = rng.choice(KINDS)
            length_type = rng.choices(["short", "medium", "long"], weights=[0.60, 0.30, 0.10], k=1)[0]
            if length_type == "short":
                text = rng.choice(short_pool)
            elif length_type == "medium":
                text = rng.choice(medium_pool)
            else:
                text = rng.choice(long_pool)
            rec: dict = {
                "source_key": _source_key(domain, di),
                "text": text,
                "kind": kind,
                "scope": scope,
                "tags": rng.sample([domain, "legal", "finance", "compliance", "risk", "strategy", "board", "review", "quarterly", "policy"], k=rng.randint(1, 4)),
                "metadata": {"domain": domain, "importance": rng.choice(["low", "medium", "high"])},
                **extras,
            }
            if rng.random() < 0.20:
                sd = rng.choice(STRUCTURED_DATA_TEMPLATES)
                rec["structured_data"] = sd
                if rng.random() < 0.5:
                    rec["data_schema"] = {
                        "schema_name": rng.choice(["email_source", "incident_report", "merger_assessment", "policy_document", "deal_summary"]),
                        "schema_mode": "declared",
                        "root_type": "object",
                        "primary_text_paths": ["subject", "body_text", "incident", "title", "policy_name", "deal_name"],
                    }
            records.append(rec)
            idx += 1
    while len(records) < 1000:
        domain = rng.choice(LEGAL_BUSINESS_DOMAINS)
        scope, extras = _pick_scope(rng, idx)
        short_pool = SHORT_TEMPLATES.get(domain, SHORT_TEMPLATES["merger_acquisition"])
        rec = {
            "source_key": _source_key(domain, len(records)),
            "text": rng.choice(short_pool),
            "kind": rng.choice(KINDS),
            "scope": scope,
            "tags": [domain, "supplemental"],
            **extras,
        }
        records.append(rec)
        idx += 1
    for run_num in range(1, 51):
        for di, domain in enumerate(rng.sample(LEGAL_BUSINESS_DOMAINS, k=5)):
            short_pool = SHORT_TEMPLATES.get(domain, SHORT_TEMPLATES["merger_acquisition"])
            records.append({
                "source_key": f"session_run{run_num:02d}:{di:04d}",
                "text": rng.choice(short_pool),
                "kind": rng.choice(KINDS),
                "scope": "session",
                "session_id": f"s_run{run_num:02d}",
                "tags": [domain, "session_benchmark"],
                "metadata": {"domain": domain, "benchmark_session": True, "run_num": run_num},
            })
    return records


def generate_perturbed_dataset(base: list[dict], seed: int = SEED + 1) -> list[dict]:
    rng = random.Random(seed)
    shuffled_texts = [r["text"] for r in rng.sample(base, len(base))]
    rng.shuffle(shuffled_texts)
    records: list[dict] = []
    for i, rec in enumerate(base):
        new_rec = dict(rec)
        new_rec["text"] = shuffled_texts[i]
        if rng.random() < 0.20:
            old_scope = new_rec["scope"]
            if old_scope == "session":
                new_rec["scope"] = "user"
                new_rec.pop("session_id", None)
                new_rec["user_id"] = f"u_perturbed_{i % 15 + 1:02d}"
            elif old_scope == "user":
                new_rec["scope"] = "global"
                new_rec.pop("user_id", None)
            else:
                new_rec["scope"] = "session"
                new_rec["session_id"] = f"s_perturbed_{domain_hash(i)}"
        if rng.random() < 0.30:
            original_tags = list(new_rec.get("tags", []))
            synonym_map = {"legal": "lawful", "finance": "financial", "compliance": "conformity", "risk": "hazard", "strategy": "plan", "review": "assessment", "quarterly": "q_period", "policy": "guideline", "board": "directors"}
            new_tags = [synonym_map.get(t, t) for t in original_tags]
            new_rec["tags"] = new_tags
        new_rec["metadata"] = dict(new_rec.get("metadata", {}))
        new_rec["metadata"]["perturbed"] = True
        records.append(new_rec)
    near_dup_count = 50
    for ndi in range(near_dup_count):
        src = rng.choice(base)
        dup_rec = dict(src)
        dup_rec["source_key"] = f"near_dup:{ndi:04d}"
        text_words = dup_rec["text"].split()
        if len(text_words) > 5:
            pos = rng.randint(0, len(text_words) - 1)
            text_words[pos] = rng.choice(["modified", "updated", "revised", "adjusted", "alternative"]) + text_words[pos][-2:]
        dup_rec["text"] = " ".join(text_words)
        dup_rec["metadata"] = dict(dup_rec.get("metadata", {}))
        dup_rec["metadata"]["near_duplicate"] = True
        records.append(dup_rec)
    return records


MEDICAL_SHORT_TEMPLATES = {
    "patient_care": [
        "Patient discharge protocol updated for post-surgical monitoring requirements.",
        "Care coordination meeting identified three patients needing specialist referrals.",
        "Patient satisfaction scores improved by 12 points after workflow changes.",
        "Medication reconciliation completed for all transitional care patients.",
        "Patient escalation criteria revised for early warning score triggers.",
    ],
    "drug_interaction": [
        "Warfarin-NSAID interaction warning added to prescribing decision support.",
        "CYP3A4 inhibitor co-administration requires dose adjustment protocol.",
        "Drug-drug interaction alert fired on 340 orders last month.",
        "Serotonin syndrome risk flagged for concurrent SSRI and MAOI therapy.",
        "Pharmacogenomic testing recommended before initiating thiopurine therapy.",
    ],
    "clinical_trial": [
        "Phase III trial enrollment reached 80% of target patient population.",
        "Data safety monitoring board reviewed interim analysis results.",
        "Protocol amendment filed for expanded inclusion criteria.",
        "Adverse event reporting timeline updated per FDA guidance.",
        "Clinical trial site audit identified documentation gaps at two locations.",
    ],
    "hipaa_compliance": [
        "PHI access audit flagged 15 unauthorized record views last quarter.",
        "Business associate agreement updated for cloud EHR vendor.",
        "HIPAA training completion rate reached 98% organization-wide.",
        "Breach notification procedure updated for 60-day deadline compliance.",
        "Minimum necessary standard applied to research data access requests.",
    ],
    "medical_device": [
        "FDA 510(k) clearance obtained for updated surgical stapler model.",
        "Device recall initiated for insulin pump firmware version 2.3.",
        "Post-market surveillance data submitted to regulatory authority.",
        "Unique device identifier implementation completed for Class III devices.",
        "Adverse event report filed for catheter fragmentation incident.",
    ],
    "radiology": [
        "AI-assisted imaging reduced diagnostic turnaround by 35 minutes.",
        "Radiation dose optimization protocol updated for pediatric CT scans.",
        "MRI safety screening checklist revised for implantable device compatibility.",
        "Teleradiology service level agreement maintained at 30-minute response.",
        "Contrast agent reaction protocol updated with new premedication guidelines.",
    ],
    "pathology": [
        "Laboratory accreditation renewal completed with zero deficiencies.",
        "Specimen labeling error rate decreased to 0.02% after barcode implementation.",
        "Molecular pathology panel expanded to include 50 additional biomarkers.",
        "Anatomic pathology turnaround time target met for 94% of cases.",
        "Quality control review identified two cases requiring amended reports.",
    ],
    "pharmacology": [
        "Therapeutic drug monitoring protocol updated for vancomycin dosing.",
        "Formulary review committee added three new biosimilar products.",
        "Adverse drug reaction reporting system integrated with EHR alerts.",
        "Compounding pharmacy inspection revealed two minor violations.",
        "Medication error root cause analysis identified look-alike packaging issue.",
    ],
    "surgical_protocol": [
        "Surgical safety checklist compliance reached 99.5% across all ORs.",
        "Minimally invasive procedure conversion rate tracked at 8% last quarter.",
        "Preoperative antibiotic timing protocol updated for SCIP compliance.",
        "Surgical site infection rate decreased 22% after bundle implementation.",
        "Robotic-assisted surgery credentialing requirements revised by committee.",
    ],
    "emergency_medicine": [
        "Triage protocol updated with refined vital sign threshold criteria.",
        "ED boarding time reduction initiative achieved 15% improvement.",
        "Stroke alert activation time median reduced to 12 minutes.",
        "Trauma activation criteria revised for penetrating mechanism threshold.",
        "ED throughput metrics dashboard deployed with real-time monitoring.",
    ],
    "mental_health": [
        "Involuntary hold procedure updated per revised state statute requirements.",
        "Suicide risk screening tool validated for emergency department use.",
        "Telehealth psychiatry coverage expanded to 24/7 availability.",
        "Substance use disorder treatment protocol aligned with ASAM criteria.",
        "Behavioral health integration pilot reduced no-show rate by 30%.",
    ],
    "vaccination": [
        "Vaccine storage temperature excursion reported for refrigerator unit 4.",
        "Immunization registry data exchange completed with state health department.",
        "COVID-19 booster uptake reached 45% among eligible patient population.",
        "Vaccine contraindication screening tool updated with latest ACIP guidance.",
        "Community vaccination event administered 1,200 doses over two days.",
    ],
    "infection_control": [
        "Hand hygiene compliance rate improved to 92% after campaign relaunch.",
        "MDRO surveillance identified cluster of CRE cases in ICU unit 3.",
        "Personal protective equipment inventory stabilized after supply chain fix.",
        "Environmental cleaning audit revealed deficiencies in terminal cleaning.",
        "Antibiotic stewardship program reduced broad-spectrum prescribing by 18%.",
    ],
    "obstetrics": [
        "Maternal hemorrhage protocol activated three times in Q4 with good outcomes.",
        "Elective delivery before 39 weeks policy enforced with hard stop in EHR.",
        "Postpartum depression screening rate reached 88% at two-week follow-up.",
        "Obstetric emergency simulation drill completed with full team participation.",
        "Neonatal intensive care unit capacity planning updated for projected volumes.",
    ],
    "pediatrics": [
        "Pediatric medication dosing calculator integrated into order entry system.",
        "Child abuse reporting protocol updated with streamlined documentation.",
        "Asthma action plan completion rate improved to 76% for pediatric patients.",
        "Pediatric early warning score implementation reduced rapid response calls.",
        "Well-child visit schedule adherence tracked at 65% for age 0-3 cohort.",
    ],
    "oncology": [
        "Tumor board review completed for 45 new cases this month.",
        "Chemotherapy regimen standardization reduced preparation errors by 40%.",
        "Palliative care referral rate for advanced cancer patients reached 55%.",
        "Clinical trial matching algorithm identified 12 eligible patients.",
        "Cancer registry data submission completed per state reporting deadline.",
    ],
    "cardiology": [
        "Door-to-balloon time median maintained at 58 minutes for STEMI patients.",
        "Heart failure readmission rate decreased to 18% after discharge program.",
        "Anticoagulation management clinic expanded to serve 500 active patients.",
        "Cardiac rehabilitation referral rate improved to 62% post-MI.",
        "Electrophysiology lab accreditation renewed with distinction.",
    ],
    "neurology": [
        "Stroke pathway compliance reached 94% for ischemic stroke admissions.",
        "EEG reporting turnaround time reduced to 24 hours with remote reading.",
        "Multiple sclerosis treatment protocol updated with new DMT options.",
        "Epilepsy monitoring unit utilization rate improved to 85%.",
        "Neurology consultation response time median at 2.3 hours.",
    ],
    "rehabilitation": [
        "Physical therapy outcome measures collected for 90% of discharged patients.",
        "Rehabilitation length of stay benchmarked at 12 days for stroke patients.",
        "Occupational therapy home assessment completion rate reached 78%.",
        "Speech therapy waitlist reduced by 25% after staffing adjustment.",
        "Functional independence measure gains tracked per diagnosis category.",
    ],
    "palliative_care": [
        "Advance care planning documentation rate increased to 45% for eligible patients.",
        "Hospice referral timing improved with earlier goals-of-care conversations.",
        "Symptom management protocol updated for refractory pain scenarios.",
        "Palliative care consultation volume grew 20% year over year.",
        "Bereavement support program expanded to include 13-month follow-up.",
    ],
}

MEDICAL_MEDIUM_TEMPLATES = {
    "patient_care": [
        "Care coordination committee reviewed 23 complex patients requiring multi-specialty involvement. Key findings include: 8 patients with delayed specialist appointments, 5 with medication reconciliation gaps at transitions of care, and 3 with incomplete diagnostic workups. Action items assigned to case managers with 2-week follow-up deadlines. Updated care pathways to be implemented for high-risk patient populations.",
        "Post-discharge follow-up program data shows 30-day readmission rate decreased from 14.2% to 11.8% after implementation of telephonic outreach for heart failure and COPD patients. Patient engagement rate at 72% for initial follow-up calls. Medication adherence assessment identified 18 patients requiring pharmacy intervention for refill synchronization.",
    ],
    "drug_interaction": [
        "Pharmacy and therapeutics committee reviewed quarterly drug interaction data. Total interaction alerts: 12,400 (up 15% from prior quarter). Breakdown: 340 critical (contraindicated combinations), 2,100 major (dose modification required), 10,060 moderate (monitoring recommended). Override rate for critical alerts decreased from 8% to 4.5% after clinician education campaign. New alert categories added for direct oral anticoagulant interactions with concurrent NSAID and antiplatelet therapy.",
    ],
}


def generate_medical_dataset(seed: int = SEED + 2) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    idx = 0
    for domain in MEDICAL_DOMAINS:
        short_pool = MEDICAL_SHORT_TEMPLATES.get(domain, MEDICAL_SHORT_TEMPLATES["patient_care"])
        medium_pool = MEDICAL_MEDIUM_TEMPLATES.get(domain, short_pool)
        records_per_domain = 1000 // len(MEDICAL_DOMAINS)
        for di in range(records_per_domain):
            scope, extras = _pick_scope(rng, idx, scope_weights=(0.35, 0.40, 0.25))
            kind = rng.choice(KINDS)
            length_type = rng.choices(["short", "medium", "long"], weights=[0.55, 0.35, 0.10], k=1)[0]
            text = rng.choice(short_pool if length_type == "short" else medium_pool)
            rec: dict = {
                "source_key": _source_key(domain, di, prefix="med_"),
                "text": text,
                "kind": kind,
                "scope": scope,
                "tags": rng.sample([domain, "medical", "healthcare", "clinical", "patient_safety", "regulatory", "hipaa", "quality", "protocol", "pharmacy"], k=rng.randint(1, 4)),
                "metadata": {"domain": domain, "medical_category": rng.choice(["clinical", "administrative", "regulatory"])},
                **extras,
            }
            if rng.random() < 0.20:
                sd = rng.choice([
                    {"patient_id": f"P{idx:06d}", "diagnosis_code": f"ICD-{rng.randint(10,99)}.{rng.randint(1,9)}", "treatment_plan": "Standard protocol", "status": rng.choice(["active", "resolved", "chronic"])},
                    {"medication": rng.choice(["warfarin", "metformin", "lisinopril", "atorvastatin"]), "dose": f"{rng.randint(1,50)}mg", "frequency": rng.choice(["daily", "BID", "TID"]), "indication": "chronic management"},
                    {"procedure_name": rng.choice(["appendectomy", "knee replacement", "cardiac catheterization", "endoscopy"]), "status": rng.choice(["scheduled", "completed", "cancelled"]), "surgeon_id": f"DR{rng.randint(100,999)}"},
                ])
                rec["structured_data"] = sd
                if rng.random() < 0.5:
                    rec["data_schema"] = {
                        "schema_name": rng.choice(["patient_record", "medication_order", "procedure_note"]),
                        "schema_mode": "declared",
                        "root_type": "object",
                        "primary_text_paths": ["diagnosis_code", "treatment_plan", "medication", "indication", "procedure_name", "status"],
                    }
            records.append(rec)
            idx += 1
    while len(records) < 1000:
        domain = rng.choice(MEDICAL_DOMAINS)
        scope, extras = _pick_scope(rng, idx, scope_weights=(0.35, 0.40, 0.25))
        short_pool = MEDICAL_SHORT_TEMPLATES.get(domain, MEDICAL_SHORT_TEMPLATES["patient_care"])
        rec = {
            "source_key": _source_key(domain, len(records), prefix="med_"),
            "text": rng.choice(short_pool),
            "kind": rng.choice(KINDS),
            "scope": scope,
            "tags": [domain, "supplemental"],
            **extras,
        }
        records.append(rec)
        idx += 1
    for run_num in range(1, 51):
        for di, domain in enumerate(rng.sample(MEDICAL_DOMAINS, k=5)):
            short_pool = MEDICAL_SHORT_TEMPLATES.get(domain, MEDICAL_SHORT_TEMPLATES["patient_care"])
            records.append({
                "source_key": f"med_session_run{run_num:02d}:{di:04d}",
                "text": rng.choice(short_pool),
                "kind": rng.choice(KINDS),
                "scope": "session",
                "session_id": f"s_run{run_num:02d}",
                "tags": [domain, "medical", "session_benchmark"],
                "metadata": {"domain": domain, "benchmark_session": True, "run_num": run_num},
            })
    return records


def generate_adversarial_dataset(base: list[dict], seed: int = SEED + 3) -> list[dict]:
    rng = random.Random(seed)
    records = list(base)
    adversarial_additions: list[dict] = []
    domains = LEGAL_BUSINESS_DOMAINS
    near_miss_pairs = [
        ("ACME merger acquisition approved by board unanimously.", "ACME merger acquisition blocked by board minority veto."),
        ("Contract amendment signed and executed by both parties.", "Contract amendment rejected and returned for revision."),
        ("Compliance audit passed with zero findings.", "Compliance audit failed with three critical findings."),
        ("Regulatory filing accepted on first submission.", "Regulatory filing rejected requiring resubmission."),
        ("Patent application approved by examiner without objection.", "Patent application rejected based on prior art objections."),
        ("Tax strategy reduced effective rate to 18 percent.", "Tax strategy challenge resulted in additional $2M assessment."),
        ("Settlement reached for $5M with all parties.", "Settlement negotiations collapsed after mediation failure."),
        ("Data breach contained within 24 hours no exfiltration.", "Data breach exfiltrated 200K records over 30 days."),
        ("Franchise territory expansion approved for new market.", "Franchise territory expansion blocked by existing operator."),
        ("Insurance claim approved at full policy limit.", "Insurance claim denied based on policy exclusion clause."),
        ("Clinical trial Phase III shows statistically significant improvement.", "Clinical trial Phase III fails to meet primary endpoint."),
        ("Drug interaction alert appropriately prevented contraindicated order.", "Drug interaction alert overridden resulting in adverse event."),
        ("Patient safety event near-miss reported and addressed.", "Patient safety event sentinel event reported to regulators."),
        ("Hospital accreditation renewed with full compliance.", "Hospital accreditation conditional with corrective action required."),
        ("Vaccine efficacy confirmed at 95% in real-world data.", "Vaccine efficacy questioned after breakthrough cluster identified."),
    ]
    for i, (positive, negative) in enumerate(near_miss_pairs):
        domain = domains[i % len(domains)]
        for polarity, text in [("positive", positive), ("negative", negative)]:
            scope, extras = _pick_scope(rng, 1000 + i * 2 + (0 if polarity == "positive" else 1))
            rec: dict = {
                "source_key": f"adversarial:{polarity}:{i:04d}",
                "text": text,
                "kind": rng.choice(KINDS),
                "scope": scope,
                "tags": [domain, "adversarial", polarity],
                "metadata": {"domain": domain, "adversarial": True, "polarity": polarity, "pair_id": i},
                **extras,
            }
            adversarial_additions.append(rec)
    distractor_templates = [
        "Market analysis reveals strong growth potential in emerging sector.",
        "Strategic planning session identified key priorities for next quarter.",
        "Operational review highlights efficiency gains from process improvement.",
        "Financial projections indicate steady performance through year end.",
        "Risk assessment framework updated with new evaluation criteria.",
        "Industry benchmarking study compares organizational metrics to peers.",
        "Technology roadmap aligns infrastructure investments with business goals.",
        "Talent acquisition strategy targets specialized skill sets in demand.",
        "Customer satisfaction survey shows improvement across all segments.",
        "Regulatory landscape analysis reveals evolving compliance requirements.",
    ]
    for i in range(100):
        domain = domains[i % len(domains)]
        scope, extras = _pick_scope(rng, 2000 + i)
        text = rng.choice(distractor_templates)
        rec: dict = {
            "source_key": f"distractor:{i:04d}",
            "text": text,
            "kind": rng.choice(KINDS),
            "scope": scope,
            "tags": [domain, "distractor"],
            "metadata": {"domain": domain, "distractor": True, "originality_score": round(rng.random() * 0.3, 2)},
            **extras,
        }
        adversarial_additions.append(rec)
    for i in range(70):
        src = rng.choice(base)
        confusing_rec = dict(src)
        confusing_rec["source_key"] = f"confusing:{i:04d}"
        confusing_rec["metadata"] = dict(confusing_rec.get("metadata", {}))
        confusing_rec["metadata"]["misleading_kind"] = confusing_rec["kind"]
        wrong_kind = rng.choice([k for k in KINDS if k != confusing_rec["kind"]])
        confusing_rec["kind"] = wrong_kind
        if rng.random() < 0.5:
            confusing_rec["tags"] = [t + "_v2" for t in confusing_rec.get("tags", [])]
        adversarial_additions.append(confusing_rec)
    records.extend(adversarial_additions)
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    print("Generating base dataset (1000 legal/business records)...")
    base = generate_base_dataset()
    write_jsonl(base, DATASETS_DIR / "base_dataset.jsonl")
    print(f"  -> {len(base)} records written to {DATASETS_DIR / 'base_dataset.jsonl'}")

    print("Generating perturbed dataset...")
    perturbed = generate_perturbed_dataset(base)
    write_jsonl(perturbed, DATASETS_DIR / "perturbed_dataset.jsonl")
    print(f"  -> {len(perturbed)} records written to {DATASETS_DIR / 'perturbed_dataset.jsonl'}")

    print("Generating medical dataset (1000 healthcare records)...")
    medical = generate_medical_dataset()
    write_jsonl(medical, DATASETS_DIR / "medical_dataset.jsonl")
    print(f"  -> {len(medical)} records written to {DATASETS_DIR / 'medical_dataset.jsonl'}")

    print("Generating adversarial dataset (1000 base + 200 adversarial)...")
    adversarial = generate_adversarial_dataset(base)
    write_jsonl(adversarial, DATASETS_DIR / "adversarial_dataset.jsonl")
    print(f"  -> {len(adversarial)} records written to {DATASETS_DIR / 'adversarial_dataset.jsonl'}")

    print("All datasets generated.")


if __name__ == "__main__":
    main()
