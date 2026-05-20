from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parent
CASES_DIR = BENCHMARK_ROOT / "cases"
DATASETS_DIR = BENCHMARK_ROOT / "datasets"

SEED = 42

logger = logging.getLogger(__name__)

BASE_QUERIES_BY_DIFFICULTY: dict[str, list[dict[str, Any]]] = {
    "trivial": [
        {"query_text": "merger acquisition ACME", "expected_source_keys": ["merger_acquisition:0000"], "expected_terms": ["merger proposal", "acquisition proceeding"]},
        {"query_text": "contract review amendment", "expected_source_keys": ["contract_review:0000"], "expected_terms": ["contract amendment", "legal sign-off"]},
        {"query_text": "compliance audit gaps", "expected_source_keys": ["compliance_audit:0000"], "expected_terms": ["compliance audit", "material gaps"]},
        {"query_text": "regulatory filing SEC", "expected_source_keys": ["regulatory_filing:0000"], "expected_terms": ["regulatory filing", "SEC 10-K"]},
        {"query_text": "patent application filed", "expected_source_keys": ["ip_patent:0000"], "expected_terms": ["patent application", "compression algorithm"]},
        {"query_text": "corporate governance board", "expected_source_keys": ["corporate_governance:0000"], "expected_terms": ["board charter", "committee oversight"]},
        {"query_text": "employment law non-compete", "expected_source_keys": ["employment_law:0000"], "expected_terms": ["non-compete enforcement", "state jurisdiction"]},
        {"query_text": "real estate lease", "expected_source_keys": ["real_estate:0000"], "expected_terms": ["lease negotiation", "downtown headquarters"]},
        {"query_text": "tax strategy transfer pricing", "expected_source_keys": ["tax_strategy:0000"], "expected_terms": ["transfer pricing", "intercompany transactions"]},
        {"query_text": "litigation class action", "expected_source_keys": ["litigation:0000"], "expected_terms": ["class action", "certification hearing"]},
        {"query_text": "data privacy breach", "expected_source_keys": ["data_privacy:0000"], "expected_terms": ["data breach", "affected individuals"]},
        {"query_text": "antitrust market concentration", "expected_source_keys": ["antitrust:0000"], "expected_terms": ["market concentration", "HHI above"]},
        {"query_text": "banking finance stress test", "expected_source_keys": ["banking_finance:0000"], "expected_terms": ["stress testing", "adequate capital"]},
        {"query_text": "securities insider trading", "expected_source_keys": ["securities:0000"], "expected_terms": ["insider trading", "unusual options"]},
        {"query_text": "environmental carbon emissions", "expected_source_keys": ["environmental_reg:0000"], "expected_terms": ["carbon emission", "reduction targets"]},
    ],
    "easy": [
        {"query_text": "ACME merger regulatory approval timeline", "expected_source_keys": ["merger_acquisition:0000", "regulatory_filing:0000"], "expected_terms": ["merger proposal", "regulatory clearance"]},
        {"query_text": "contract vendor compliance requirements", "expected_source_keys": ["contract_review:0000", "compliance_audit:0000"], "expected_terms": ["contract dispute", "compliance audit"]},
        {"query_text": "IP patent portfolio strategy review", "expected_source_keys": ["ip_patent:0000", "corporate_governance:0000"], "expected_terms": ["patent portfolio", "expiring patents"]},
        {"query_text": "employment workplace harassment policy governance", "expected_source_keys": ["employment_law:0000", "corporate_governance:0000"], "expected_terms": ["harassment policy", "reporting channel"]},
        {"query_text": "data privacy GDPR audit filing", "expected_source_keys": ["data_privacy:0000", "compliance_audit:0000"], "expected_terms": ["GDPR audit trail", "Q4 reporting"]},
        {"query_text": "real estate environmental remediation assessment", "expected_source_keys": ["real_estate:0000", "environmental_reg:0000"], "expected_terms": ["site assessment", "environmental remediation"]},
        {"query_text": "tax strategy international treaty banking", "expected_source_keys": ["tax_strategy:0000", "banking_finance:0000"], "expected_terms": ["international tax treaty", "loan loss provision"]},
        {"query_text": "securities antitrust competitive effects merger", "expected_source_keys": ["securities:0000", "antitrust:0000"], "expected_terms": ["insider trading surveillance", "competitive effects"]},
        {"query_text": "litigation insurance claim dispute resolution", "expected_source_keys": ["litigation:0000", "insurance_claim:0000"], "expected_terms": ["settlement offer", "policy period interpretation"]},
        {"query_text": "franchise bankruptcy reorganization plan", "expected_source_keys": ["franchise_law:0000", "bankruptcy:0000"], "expected_terms": ["franchise disclosure document", "creditor committee support"]},
    ],
    "moderate_easy": [
        {"query_text": "cross-border merger compliance with data privacy and antitrust requirements", "expected_source_keys": ["merger_acquisition:0000", "data_privacy:0000", "antitrust:0000"], "expected_terms": ["merger remedy package", "privacy impact assessment", "antitrust compliance"]},
        {"query_text": "corporate governance securities regulation insider trading controls", "expected_source_keys": ["corporate_governance:0000", "securities:0000"], "expected_terms": ["governance framework", "insider trading surveillance", "SEC guidance"]},
        {"query_text": "employment law immigration work authorization compliance", "expected_source_keys": ["employment_law:0000", "immigration_law:0000"], "expected_terms": ["non-compete enforcement", "H-1B visa petition", "I-9 compliance"]},
        {"query_text": "real estate franchise law territory dispute lease", "expected_source_keys": ["real_estate:0000", "franchise_law:0000"], "expected_terms": ["commercial lease negotiation", "territory dispute", "franchisee escalation"]},
        {"query_text": "environmental compliance regulatory filing carbon tax strategy", "expected_source_keys": ["environmental_reg:0000", "regulatory_filing:0000", "tax_strategy:0000"], "expected_terms": ["carbon emission reduction", "regulatory filing deadline", "transfer pricing documentation"]},
        {"query_text": "banking stress testing securities capital requirements Basel", "expected_source_keys": ["banking_finance:0000", "securities:0000"], "expected_terms": ["Basel III capital", "stress testing results", "adequate capital"]},
        {"query_text": "trade secret protection employment NDA intellectual property", "expected_source_keys": ["trade_secret:0000", "employment_law:0000", "ip_patent:0000"], "expected_terms": ["NDA breach allegation", "trade secret misappropriation", "patent portfolio review"]},
        {"query_text": "insurance claim litigation settlement dispute coverage", "expected_source_keys": ["insurance_claim:0000", "litigation:0000"], "expected_terms": ["policy period interpretation", "settlement offer", "arbitration panel"]},
        {"query_text": "contract review compliance audit vendor risk management", "expected_source_keys": ["contract_review:0000", "compliance_audit:0000"], "expected_terms": ["vendor contract dispute", "compliance audit identified", "arbitration panel"]},
        {"query_text": "bankruptcy creditor committee franchise agreement termination", "expected_source_keys": ["bankruptcy:0000", "franchise_law:0000"], "expected_terms": ["creditor committee support", "Chapter 11 reorganization", "franchisee default notice"]},
    ],
    "moderate": [
        {"query_text": "multi-jurisdictional merger review with antitrust remedies and data privacy compliance framework", "expected_source_keys": ["merger_acquisition:0000", "antitrust:0000", "data_privacy:0000", "regulatory_filing:0000"], "expected_terms": ["merger remedy package", "HSR Act waiting period", "data breach notification", "regulatory filing deadline"]},
        {"query_text": "corporate restructuring bankruptcy implications for franchise agreements and employment contracts", "expected_source_keys": ["bankruptcy:0000", "franchise_law:0000", "employment_law:0000"], "expected_terms": ["Chapter 11 reorganization plan", "creditor committee support", "franchise disclosure document", "mass layoff notice"]},
        {"query_text": "ESG disclosure requirements environmental compliance securities governance", "expected_source_keys": ["environmental_reg:0000", "securities:0000", "corporate_governance:0000"], "expected_terms": ["ESG disclosure received majority", "carbon emission reduction targets", "insider trading surveillance"]},
        {"query_text": "international tax strategy transfer pricing with banking regulatory considerations", "expected_source_keys": ["tax_strategy:0000", "banking_finance:0000", "regulatory_filing:0000"], "expected_terms": ["transfer pricing documentation", "intercompany transactions", "Basel III capital requirements", "SEC 10-K filing deadline"]},
        {"query_text": "IP patent litigation trade secret misappropriation with employment departures", "expected_source_keys": ["ip_patent:0000", "litigation:0000", "trade_secret:0000", "employment_law:0000"], "expected_terms": ["patent portfolio review", "trade secret misappropriation claim", "NDA breach allegation", "class action certification"]},
        {"query_text": "insurance coverage for cyber breach data privacy regulatory penalties", "expected_source_keys": ["insurance_claim:0000", "data_privacy:0000", "compliance_audit:0000"], "expected_terms": ["cyber insurance policy", "data breach notification", "compliance audit identified", "affected individuals"]},
        {"query_text": "real estate portfolio environmental remediation liability and insurance recovery", "expected_source_keys": ["real_estate:0000", "environmental_reg:0000", "insurance_claim:0000"], "expected_terms": ["environmental site assessment", "remediation requirement", "property tax appeal", "policy period interpretation"]},
        {"query_text": "immigration compliance audit employment verification I-9 requirements across states", "expected_source_keys": ["immigration_law:0000", "employment_law:0000", "compliance_audit:0000"], "expected_terms": ["I-9 compliance audit", "H-1B visa petition", "compliance audit identified", "multi-state tax obligations"]},
    ],
    "moderate_hard": [
        {"query_text": "cross-border M&A with overlapping antitrust, data privacy, and securities regulation in multiple jurisdictions", "expected_source_keys": ["merger_acquisition:0000", "antitrust:0000", "data_privacy:0000", "securities:0000", "regulatory_filing:0000"], "expected_terms": ["merger remedy package", "HSR Act waiting period", "privacy impact assessment", "insider trading surveillance", "SEC 10-K filing deadline"]},
        {"query_text": "post-merger integration employment workforce reduction with trade secret protection and non-compete enforcement", "expected_source_keys": ["merger_acquisition:0000", "employment_law:0000", "trade_secret:0000"], "expected_terms": ["board approved the merger", "non-compete enforcement", "NDA breach allegation", "trade secret identification audit"]},
        {"query_text": "bankruptcy remote work arrangement franchise territory and insurance claims intersection", "expected_source_keys": ["bankruptcy:0000", "employment_law:0000", "franchise_law:0000", "insurance_claim:0000"], "expected_terms": ["Chapter 11 reorganization plan", "multi-state tax obligations", "territory dispute between franchisees", "business interruption claim"]},
        {"query_text": "ambiguous regulatory filing deadline interpretation with conflicting compliance audit standards", "expected_source_keys": ["regulatory_filing:0000", "compliance_audit:0000", "corporate_governance:0000"], "expected_terms": ["SEC 10-K filing deadline extended", "compliance audit identified three", "governance framework updated", "regulatory compliance status"]},
        {"query_text": "ESG environmental carbon disclosure versus securities materiality assessment conflict", "expected_source_keys": ["environmental_reg:0000", "securities:0000", "corporate_governance:0000", "compliance_audit:0000"], "expected_terms": ["carbon emission reduction targets", "ESG disclosure received majority", "insider trading surveillance", "SOX controls testing"]},
        {"query_text": "patent validity challenge with trade secret alternative protection and litigation exposure", "expected_source_keys": ["ip_patent:0000", "trade_secret:0000", "litigation:0000"], "expected_terms": ["patent portfolio review identified", "trade secret misappropriation claim", "expert witness testimony challenged", "12 expiring patents"]},
    ],
    "hard": [
        {"query_text": "distinguishing approved merger from blocked merger in same domain with similar terminology", "expected_source_keys": ["merger_acquisition:0000"], "expected_terms": ["board approved the merger proposal", "rejected by TargetBoard unanimously"]},
        {"query_text": "contract amendment status whether signed or rejected based on contextual evidence", "expected_source_keys": ["contract_review:0000"], "expected_terms": ["legal sign-off before execution", "escalated to arbitration panel"]},
        {"query_text": "identifying the correct compliance outcome from audit with near-identical distractor records", "expected_source_keys": ["compliance_audit:0000"], "expected_terms": ["three material gaps", "full regulatory compliance status"]},
        {"query_text": "disambiguating patent approval from patent rejection with overlapping vocabulary", "expected_source_keys": ["ip_patent:0000"], "expected_terms": ["proprietary compression algorithm", "12 expiring patents by 2027"]},
        {"query_text": "settlement outcome versus negotiation collapse retrieval with minimal distinguishing terms", "expected_source_keys": ["litigation:0000"], "expected_terms": ["$12M rejected by plaintiff", "expert witness testimony challenged under Daubert"]},
        {"query_text": "data breach severity assessment contained versus exfiltrated with similar incident descriptions", "expected_source_keys": ["data_privacy:0000"], "expected_terms": ["50,000 affected individuals", "biometric data retention policy"]},
        {"query_text": "insurance claim acceptance or denial from overlapping policy language", "expected_source_keys": ["insurance_claim:0000"], "expected_terms": ["policy period interpretation", "directors and officers liability"]},
        {"query_text": "regulatory filing acceptance versus rejection requiring resubmission", "expected_source_keys": ["regulatory_filing:0000"], "expected_terms": ["no deficiencies", "SEC 10-K filing deadline"]},
    ],
    "harder": [
        {"query_text": "disambiguate which specific domain's merger was approved when multiple merger records exist across session and global scopes", "expected_source_keys": ["merger_acquisition:0000"], "expected_terms": ["Board approved the merger proposal", "regulatory clearance"], "session_id": "s_disambig_1"},
        {"query_text": "find the session-scoped contract that was signed, not the global policy about contracts", "expected_source_keys": ["contract_review:0000"], "expected_terms": ["legal sign-off before execution", "Master services agreement"], "session_id": "s_disambig_2"},
        {"query_text": "locate the user-specific compliance preference that differs from the global compliance standard", "expected_source_keys": ["compliance_audit:0000"], "expected_terms": ["compliance training completion rate", "72 percent"], "user_id": "u_01"},
        {"query_text": "retrieve the exact regulatory filing accepted on first attempt excluding rejected filings and amendments", "expected_source_keys": ["regulatory_filing:0000"], "expected_terms": ["completed with no deficiencies", "filing deadline extended to March 31"]},
        {"query_text": "identify the patent that was specifically approved by examiner without objection among multiple patent records", "expected_source_keys": ["ip_patent:0000"], "expected_terms": ["proprietary compression algorithm", "12 expiring patents by 2027"]},
        {"query_text": "find only the tax strategy that reduced effective rate excluding ones that resulted in additional assessment", "expected_source_keys": ["tax_strategy:0000"], "expected_terms": ["$2.3M in qualifying expenses", "R&D tax credit claim"]},
    ],
    "very_hard": [
        {"query_text": "outcome of the board decision on the merger with specific vote details", "expected_source_keys": ["merger_acquisition:0000"], "expected_terms": ["rejected by TargetBoard unanimously", "3.2x revenue multiple"]},
        {"query_text": "precise medication reconciliation gap count at care transitions", "expected_source_keys": [], "expected_terms": ["medication reconciliation completed", "transitional care patients"]},
        {"query_text": "which specific interaction alert category had the highest override rate", "expected_source_keys": [], "expected_terms": ["interaction alert fired on 340 orders", "override rate for critical"]},
        {"query_text": "exact stress testing capital buffer percentage above minimum requirement", "expected_source_keys": ["banking_finance:0000"], "expected_terms": ["2% buffer above minimum", "Basel III capital requirements"]},
        {"query_text": "specific carbon emission reduction target percentage and year", "expected_source_keys": ["environmental_reg:0000"], "expected_terms": ["40% by 2030", "carbon emission reduction targets"]},
        {"query_text": "vaccine efficacy percentage in real-world confirmation data", "expected_source_keys": [], "expected_terms": ["1,200 doses over two days", "community vaccination event"]},
    ],
    "extreme": [
        {"query_text": "integration synergy", "expected_source_keys": ["merger_acquisition:0000"], "expected_terms": ["3.2x revenue multiple", "hostile bid from MegaCorp"]},
        {"query_text": "adverse event reporting timeline modification", "expected_source_keys": [], "expected_terms": ["adverse event reporting timeline updated per FDA guidance"]},
        {"query_text": "DOE credentialing", "expected_source_keys": [], "expected_terms": ["credentialing requirements revised by committee"]},
        {"query_text": "HHI above threshold antitrust", "expected_source_keys": ["antitrust:0000"], "expected_terms": ["HHI above 2500 threshold", "market concentration analysis"]},
        {"query_text": "look-alike packaging", "expected_source_keys": [], "expected_terms": ["look-alike packaging issue"]},
        {"query_text": "BIPA biometric retention", "expected_source_keys": ["data_privacy:0000"], "expected_terms": ["BIPA requirements", "biometric data retention policy"]},
        {"query_text": "SCIP compliance antibiotic timing", "expected_source_keys": [], "expected_terms": ["SCIP compliance", "antibiotic timing protocol"]},
        {"query_text": "Daubert expert challenge", "expected_source_keys": ["litigation:0000"], "expected_terms": ["Daubert standard", "expert witness testimony challenged"]},
    ],
    "maximum": [
        {"query_text": "synergy target", "expected_source_keys": [], "expected_terms": ["3.2x revenue multiple", "$450M cash consideration"]},
        {"query_text": "override rate", "expected_source_keys": [], "expected_terms": ["override rate for critical alerts"]},
        {"query_text": "readmission rate", "expected_source_keys": [], "expected_terms": ["heart failure readmission rate decreased"]},
        {"query_text": "door-to-balloon", "expected_source_keys": [], "expected_terms": ["door-to-balloon time median maintained"]},
        {"query_text": "HHI threshold", "expected_source_keys": [], "expected_terms": ["HHI above 2500 threshold"]},
        {"query_text": "HSR waiting", "expected_source_keys": [], "expected_terms": ["HSR Act waiting period"]},
        {"query_text": "SCIP", "expected_source_keys": [], "expected_terms": ["SCIP compliance"]},
        {"query_text": "BIPA", "expected_source_keys": [], "expected_terms": ["BIPA requirements"]},
        {"query_text": "Daubert", "expected_source_keys": [], "expected_terms": ["Daubert standard"]},
        {"query_text": "CCPA consent", "expected_source_keys": [], "expected_terms": ["CCPA compliance"]},
        {"query_text": "WARN Act", "expected_source_keys": [], "expected_terms": ["WARN Act"]},
    ],
}

MEDICAL_QUERIES_BY_DIFFICULTY: dict[str, list[dict[str, Any]]] = {
    "trivial": [
        {"query_text": "patient care discharge protocol", "expected_source_keys": ["med_patient_care:0000"], "expected_terms": ["discharge protocol updated", "post-surgical monitoring"]},
        {"query_text": "drug interaction warfarin", "expected_source_keys": ["med_drug_interaction:0000"], "expected_terms": ["warfarin interaction warning", "prescribing decision support"]},
        {"query_text": "clinical trial enrollment Phase III", "expected_source_keys": ["med_clinical_trial:0000"], "expected_terms": ["Phase III trial enrollment", "target patient population"]},
        {"query_text": "HIPAA compliance PHI access audit", "expected_source_keys": ["med_hipaa_compliance:0000"], "expected_terms": ["PHI access audit flagged", "unauthorized record views"]},
        {"query_text": "medical device FDA 510k clearance", "expected_source_keys": ["med_medical_device:0000"], "expected_terms": ["FDA 510(k) clearance", "surgical stapler model"]},
        {"query_text": "radiology AI imaging diagnostic", "expected_source_keys": ["med_radiology:0000"], "expected_terms": ["AI-assisted imaging reduced", "diagnostic turnaround"]},
        {"query_text": "pathology laboratory accreditation", "expected_source_keys": ["med_pathology:0000"], "expected_terms": ["laboratory accreditation renewal", "zero deficiencies"]},
        {"query_text": "pharmacology therapeutic drug monitoring", "expected_source_keys": ["med_pharmacology:0000"], "expected_terms": ["therapeutic drug monitoring", "vancomycin dosing"]},
        {"query_text": "surgical protocol safety checklist", "expected_source_keys": ["med_surgical_protocol:0000"], "expected_terms": ["surgical safety checklist", "compliance reached"]},
        {"query_text": "emergency medicine triage vital signs", "expected_source_keys": ["med_emergency_medicine:0000"], "expected_terms": ["triage protocol updated", "vital sign threshold"]},
        {"query_text": "mental health involuntary hold procedure", "expected_source_keys": ["med_mental_health:0000"], "expected_terms": ["involuntary hold procedure", "revised state statute"]},
        {"query_text": "vaccination storage temperature excursion", "expected_source_keys": ["med_vaccination:0000"], "expected_terms": ["vaccine storage temperature", "temperature excursion"]},
        {"query_text": "infection control hand hygiene compliance", "expected_source_keys": ["med_infection_control:0000"], "expected_terms": ["hand hygiene compliance rate", "campaign relaunch"]},
        {"query_text": "obstetrics maternal hemorrhage protocol", "expected_source_keys": ["med_obstetrics:0000"], "expected_terms": ["maternal hemorrhage protocol", "good outcomes"]},
        {"query_text": "pediatric medication dosing calculator", "expected_source_keys": ["med_pediatrics:0000"], "expected_terms": ["pediatric medication dosing", "order entry system"]},
    ],
    "easy": [
        {"query_text": "patient care coordination drug interaction alert management", "expected_source_keys": ["med_patient_care:0000", "med_drug_interaction:0000"], "expected_terms": ["care coordination meeting", "interaction warning added"]},
        {"query_text": "clinical trial adverse event HIPAA reporting requirements", "expected_source_keys": ["med_clinical_trial:0000", "med_hipaa_compliance:0000"], "expected_terms": ["adverse event reporting timeline", "HIPAA training completion rate"]},
        {"query_text": "medical device recall pharmacology formulary review", "expected_source_keys": ["med_medical_device:0000", "med_pharmacology:0000"], "expected_terms": ["device recall initiated", "formulary review committee"]},
        {"query_text": "surgical safety infection control antibiotic stewardship", "expected_source_keys": ["med_surgical_protocol:0000", "med_infection_control:0000"], "expected_terms": ["surgical site infection rate", "antibiotic stewardship program"]},
        {"query_text": "emergency psychiatry mental health crisis evaluation", "expected_source_keys": ["med_emergency_medicine:0000", "med_mental_health:0000"], "expected_terms": ["ED boarding time reduction", "suicide risk screening tool"]},
        {"query_text": "obstetrics neonatal ICU pediatric care coordination", "expected_source_keys": ["med_obstetrics:0000", "med_pediatrics:0000"], "expected_terms": ["neonatal intensive care unit", "pediatric medication dosing"]},
        {"query_text": "vaccination infection control immunization registry data", "expected_source_keys": ["med_vaccination:0000", "med_infection_control:0000"], "expected_terms": ["immunization registry data exchange", "hand hygiene compliance rate"]},
        {"query_text": "radiology pathology diagnostic turnaround time improvement", "expected_source_keys": ["med_radiology:0000", "med_pathology:0000"], "expected_terms": ["diagnostic turnaround by 35 minutes", "specimen labeling error rate"]},
        {"query_text": "cardiology stroke pathway neurology consultation response", "expected_source_keys": ["med_cardiology:0000", "med_neurology:0000"], "expected_terms": ["heart failure readmission rate", "EEG reporting turnaround time"]},
        {"query_text": "oncology palliative care advance directive documentation", "expected_source_keys": ["med_oncology:0000", "med_palliative_care:0000"], "expected_terms": ["palliative care referral rate", "advance care planning documentation"]},
    ],
    "moderate_easy": [
        {"query_text": "post-surgical patient care with drug interaction monitoring and pharmacology review", "expected_source_keys": ["med_patient_care:0000", "med_drug_interaction:0000", "med_pharmacology:0000"], "expected_terms": ["post-surgical monitoring requirements", "CYP3A4 inhibitor co-administration", "therapeutic drug monitoring protocol"]},
        {"query_text": "HIPAA compliance for clinical trial data privacy breach notification", "expected_source_keys": ["med_hipaa_compliance:0000", "med_clinical_trial:0000", "med_hipaa_compliance:0000"], "expected_terms": ["PHI access audit flagged", "adverse event reporting timeline", "breach notification procedure"]},
        {"query_text": "infection control surgical site infection antibiotic stewardship protocol", "expected_source_keys": ["med_infection_control:0000", "med_surgical_protocol:0000"], "expected_terms": ["antibiotic stewardship program reduced", "surgical site infection rate decreased", "broad-spectrum prescribing"]},
        {"query_text": "pediatric emergency triage with mental health screening and vaccination status", "expected_source_keys": ["med_pediatrics:0000", "med_emergency_medicine:0000", "med_mental_health:0000"], "expected_terms": ["pediatric early warning score", "triage protocol updated", "involuntary hold procedure"]},
        {"query_text": "obstetric emergency simulation with neonatal ICU and pediatric coordination", "expected_source_keys": ["med_obstetrics:0000", "med_pediatrics:0000"], "expected_terms": ["obstetric emergency simulation drill", "neonatal intensive care unit", "maternal hemorrhage protocol"]},
        {"query_text": "oncology tumor board with pathology review and clinical trial matching", "expected_source_keys": ["med_oncology:0000", "med_pathology:0000", "med_clinical_trial:0000"], "expected_terms": ["tumor board review completed", "specimen labeling error rate", "clinical trial matching algorithm"]},
        {"query_text": "cardiology heart failure rehabilitation and palliative care referral criteria", "expected_source_keys": ["med_cardiology:0000", "med_rehabilitation:0000", "med_palliative_care:0000"], "expected_terms": ["heart failure readmission rate decreased", "physical therapy outcome measures", "palliative care referral rate"]},
        {"query_text": "neurology stroke pathway with radiology imaging and emergency activation", "expected_source_keys": ["med_neurology:0000", "med_radiology:0000", "med_emergency_medicine:0000"], "expected_terms": ["stroke pathway compliance reached", "AI-assisted imaging reduced", "stroke alert activation time"]},
        {"query_text": "clinical trial adverse event with medical device and pathology correlation", "expected_source_keys": ["med_clinical_trial:0000", "med_medical_device:0000", "med_pathology:0000"], "expected_terms": ["adverse event report filed", "FDA 510(k) clearance", "anatomic pathology turnaround time"]},
    ],
    "moderate": [
        {"query_text": "multi-drug interaction management across clinical trial patients with HIPAA data handling", "expected_source_keys": ["med_drug_interaction:0000", "med_clinical_trial:0000", "med_hipaa_compliance:0000", "med_pharmacology:0000"], "expected_terms": ["warfarin interaction warning", "Phase III trial enrollment reached 80%", "HIPAA training completion rate reached 98%", "therapeutic drug monitoring protocol"]},
        {"query_text": "hospital-wide infection control with surgical antibiotic prophylaxis and pharmacy stewardship", "expected_source_keys": ["med_infection_control:0000", "med_surgical_protocol:0000", "med_pharmacology:0000"], "expected_terms": ["antibiotic stewardship program reduced broad-spectrum", "surgical site infection rate decreased 22%", "formulary review committee"]},
        {"query_text": "complex obstetric hemorrhage with blood bank pharmacology and neonatal emergency coordination", "expected_source_keys": ["med_obstetrics:0000", "med_emergency_medicine:0000", "med_pediatrics:0000"], "expected_terms": ["maternal hemorrhage protocol activated", "stroke alert activation time median reduced", "pediatric early warning score implementation"]},
        {"query_text": "advance care planning integration across oncology cardiology and palliative care settings", "expected_source_keys": ["med_oncology:0000", "med_cardiology:0000", "med_palliative_care:0000"], "expected_terms": ["advance care planning documentation rate", "heart failure readmission rate decreased", "palliative care referral rate for advanced"]},
        {"query_text": "medical device adverse event reporting with FDA compliance and pathology correlation", "expected_source_keys": ["med_medical_device:0000", "med_pathology:0000"], "expected_terms": ["adverse event report filed for catheter", "FDA 510(k) clearance obtained", "specimen labeling error rate decreased"]},
        {"query_text": "radiation dose optimization across radiology with pediatric and emergency protocols", "expected_source_keys": ["med_radiology:0000", "med_pediatrics:0000", "med_emergency_medicine:0000"], "expected_terms": ["radiation dose optimization protocol", "pediatric CT scans", "ED boarding time reduction initiative"]},
    ],
    "moderate_hard": [
        {"query_text": "disambiguating clinical trial Phase III success versus failure with overlapping terminology", "expected_source_keys": ["med_clinical_trial:0000"], "expected_terms": ["Phase III trial enrollment reached 80%", "adverse event reporting timeline updated"]},
        {"query_text": "drug interaction alert that prevented harm versus alert that was overridden with adverse outcome", "expected_source_keys": ["med_drug_interaction:0000"], "expected_terms": ["warfarin interaction warning added", "drug-drug interaction alert fired on 340 orders"]},
        {"query_text": "patient safety near-miss versus sentinel event classification with similar descriptions", "expected_source_keys": ["med_infection_control:0000"], "expected_terms": ["hand hygiene compliance rate improved to 92%", "antibiotic stewardship program reduced"]},
        {"query_text": "hospital accreditation full compliance versus conditional with corrective action", "expected_source_keys": ["med_hipaa_compliance:0000"], "expected_terms": ["HIPAA training completion rate reached 98%", "breach notification procedure updated"]},
        {"query_text": "vaccine efficacy confirmed versus questioned after breakthrough infections", "expected_source_keys": ["med_vaccination:0000"], "expected_terms": ["community vaccination event administered 1,200 doses", "immunization registry data exchange"]},
    ],
    "hard": [
        {"query_text": "specific override rate percentage for critical drug interaction alerts last quarter", "expected_source_keys": ["med_drug_interaction:0000"], "expected_terms": ["override rate for critical alerts", "decreased from 8% to 4.5%"]},
        {"query_text": "exact door-to-balloon time median for STEMI patients", "expected_source_keys": ["med_cardiology:0000"], "expected_terms": ["door-to-balloon time median maintained at 58 minutes"]},
        {"query_text": "specific hand hygiene compliance percentage after campaign relaunch", "expected_source_keys": ["med_infection_control:0000"], "expected_terms": ["hand hygiene compliance rate improved to 92%"]},
        {"query_text": "precise readmission rate percentage after telephonic outreach program", "expected_source_keys": ["med_patient_care:0000"], "expected_terms": ["30-day readmission rate decreased from 14.2% to 11.8%"]},
        {"query_text": "exact reduction percentage in broad-spectrum antibiotic prescribing", "expected_source_keys": ["med_infection_control:0000"], "expected_terms": ["reduced broad-spectrum prescribing by 18%"]},
    ],
    "harder": [
        {"query_text": "breakdown of drug interaction alert categories by severity with override rates", "expected_source_keys": ["med_drug_interaction:0000"], "expected_terms": ["340 critical contraindicated combinations", "2,100 major dose modification", "override rate for critical alerts decreased"]},
        {"query_text": "palliative care referral rate percentage for advanced cancer specifically", "expected_source_keys": ["med_oncology:0000", "med_palliative_care:0000"], "expected_terms": ["palliative care referral rate for advanced cancer patients reached 55%"]},
        {"query_text": "cardiac rehabilitation referral rate post-MI exact figure", "expected_source_keys": ["med_cardiology:0000", "med_rehabilitation:0000"], "expected_terms": ["cardiac rehabilitation referral rate improved to 62% post-MI"]},
    ],
    "very_hard": [
        {"query_text": "interaction alert volume by category with percentage change", "expected_source_keys": [], "expected_terms": ["12,400 up 15% from prior quarter", "2,100 major dose modification required"]},
        {"query_text": "community vaccination event dose count and duration", "expected_source_keys": [], "expected_terms": ["1,200 doses over two days", "community vaccination event administered"]},
        {"query_text": "EEG reporting turnaround in hours with remote reading", "expected_source_keys": [], "expected_terms": ["EEG reporting turnaround time reduced to 24 hours"]},
        {"query_text": "advanced cancer palliative referral versus heart failure readmission comparative outcomes", "expected_source_keys": [], "expected_terms": ["palliative care referral rate for advanced cancer patients reached 55%", "heart failure readmission rate decreased to 18%"]},
    ],
    "extreme": [
        {"query_text": "alert override rate decrease percentage point change", "expected_source_keys": [], "expected_terms": ["override rate for critical alerts decreased from 8% to 4.5%"]},
        {"query_text": "diagnostic turnaround AI minutes reduction", "expected_source_keys": [], "expected_terms": ["AI-assisted imaging reduced diagnostic turnaround by 35 minutes"]},
        {"query_text": "SSI reduction percentage after bundle", "expected_source_keys": [], "expected_terms": ["surgical site infection rate decreased 22% after bundle implementation"]},
        {"query_text": "specimen labeling error rate after barcode", "expected_source_keys": [], "expected_terms": ["specimen labeling error rate decreased to 0.02% after barcode implementation"]},
    ],
    "maximum": [
        {"query_text": "override rate", "expected_source_keys": [], "expected_terms": ["8% to 4.5%"]},
        {"query_text": "turnaround AI", "expected_source_keys": [], "expected_terms": ["reduced diagnostic turnaround by 35 minutes"]},
        {"query_text": "SSI bundle", "expected_source_keys": [], "expected_terms": ["decreased 22% after bundle implementation"]},
        {"query_text": "barcode error", "expected_source_keys": [], "expected_terms": ["decreased to 0.02% after barcode implementation"]},
        {"query_text": "boarding time", "expected_source_keys": [], "expected_terms": ["15% improvement"]},
    ],
}

PERTURBED_EXTRA_QUERIES: list[dict[str, Any]] = [
    {"query_text": "merger acquisition lawful conformity review", "expected_source_keys": ["merger_acquisition:0000"], "expected_terms": ["merger proposal pending regulatory", "acquisition proceeding"]},
    {"query_text": "financial quarterly results assessment", "expected_source_keys": [], "expected_terms": ["15 percent for Q4", "loan loss provision"]},
    {"query_text": "hazard assessment plan framework", "expected_source_keys": [], "expected_terms": ["hazardous waste disposal permit", "stricter conditions"]},
    {"query_text": "directors governance guideline assessment", "expected_source_keys": [], "expected_terms": ["board charter amendment", "committee oversight"]},
    {"query_text": "near duplicate records identification", "expected_source_keys": ["near_dup:0000"], "expected_terms": ["modified updated revised", "near_duplicate"]},
    {"query_text": "perturbed scope document retrieval", "expected_source_keys": [], "expected_terms": ["perturbed"]},
]

ADVERSARIAL_EXTRA_QUERIES: list[dict[str, Any]] = [
    {"query_text": "ACME merger approved by board", "expected_source_keys": ["adversarial:positive:0000"], "expected_terms": ["approved by board unanimously"]},
    {"query_text": "ACME merger blocked by board", "expected_source_keys": ["adversarial:negative:0000"], "expected_terms": ["blocked by board minority veto"]},
    {"query_text": "contract amendment signed and executed", "expected_source_keys": ["adversarial:positive:0001"], "expected_terms": ["signed and executed by both parties"]},
    {"query_text": "contract amendment rejected and returned", "expected_source_keys": ["adversarial:negative:0001"], "expected_terms": ["rejected and returned for revision"]},
    {"query_text": "compliance audit passed zero findings", "expected_source_keys": ["adversarial:positive:0002"], "expected_terms": ["passed with zero findings"]},
    {"query_text": "compliance audit failed critical findings", "expected_source_keys": ["adversarial:negative:0002"], "expected_terms": ["failed with three critical findings"]},
    {"query_text": "regulatory filing accepted first submission", "expected_source_keys": ["adversarial:positive:0003"], "expected_terms": ["accepted on first submission"]},
    {"query_text": "regulatory filing rejected requiring resubmission", "expected_source_keys": ["adversarial:negative:0003"], "expected_terms": ["rejected requiring resubmission"]},
    {"query_text": "patent application approved without objection", "expected_source_keys": ["adversarial:positive:0004"], "expected_terms": ["approved by examiner without objection"]},
    {"query_text": "patent application rejected prior art", "expected_source_keys": ["adversarial:negative:0004"], "expected_terms": ["rejected based on prior art"]},
    {"query_text": "settlement reached five million all parties", "expected_source_keys": ["adversarial:positive:0005"], "expected_terms": ["settlement reached for $5M"]},
    {"query_text": "settlement negotiations collapsed mediation failure", "expected_source_keys": ["adversarial:negative:0005"], "expected_terms": ["collapsed after mediation failure"]},
    {"query_text": "data breach contained 24 hours no exfiltration", "expected_source_keys": ["adversarial:positive:0006"], "expected_terms": ["contained within 24 hours no exfiltration"]},
    {"query_text": "data breach exfiltrated 200K records 30 days", "expected_source_keys": ["adversarial:negative:0006"], "expected_terms": ["exfiltrated 200K records over 30 days"]},
    {"query_text": "insurance claim approved full policy limit", "expected_source_keys": ["adversarial:positive:0007"], "expected_terms": ["approved at full policy limit"]},
    {"query_text": "insurance claim denied policy exclusion clause", "expected_source_keys": ["adversarial:negative:0007"], "expected_terms": ["denied based on policy exclusion clause"]},
    {"query_text": "vaccine efficacy 95 percent real-world data", "expected_source_keys": ["adversarial:positive:0011"], "expected_terms": ["efficacy confirmed at 95% in real-world data"]},
    {"query_text": "vaccine breakthrough cluster efficacy questioned", "expected_source_keys": ["adversarial:negative:0011"], "expected_terms": ["efficacy questioned after breakthrough cluster"]},
]


DIFFICULTY_CONFIG: list[dict[str, Any]] = [
    {"name": "trivial", "runs": (1, 5), "token_budget": 2000, "num_cases": (3, 5), "include_user": True, "include_global": True, "terms_scope": "assembled_context"},
    {"name": "easy", "runs": (6, 10), "token_budget": 1500, "num_cases": (5, 8), "include_user": True, "include_global": True, "terms_scope": "assembled_context"},
    {"name": "moderate_easy", "runs": (11, 15), "token_budget": 1200, "num_cases": (8, 12), "include_user": True, "include_global": True, "terms_scope": "assembled_context"},
    {"name": "moderate", "runs": (16, 20), "token_budget": 900, "num_cases": (12, 16), "include_user": True, "include_global": True, "terms_scope": "assembled_context"},
    {"name": "moderate_hard", "runs": (21, 25), "token_budget": 700, "num_cases": (16, 20), "include_user": True, "include_global": True, "terms_scope": "assembled_context"},
    {"name": "hard", "runs": (26, 30), "token_budget": 600, "num_cases": (20, 24), "include_user": True, "include_global": True, "terms_scope": "assembled_context"},
    {"name": "harder", "runs": (31, 35), "token_budget": 500, "num_cases": (24, 28), "include_user": False, "include_global": False, "terms_scope": "retrieved_records"},
    {"name": "very_hard", "runs": (36, 40), "token_budget": 400, "num_cases": (28, 32), "include_user": False, "include_global": False, "terms_scope": "retrieved_records"},
    {"name": "extreme", "runs": (41, 45), "token_budget": 300, "num_cases": (32, 36), "include_user": False, "include_global": False, "terms_scope": "retrieved_records"},
    {"name": "maximum", "runs": (46, 50), "token_budget": 200, "num_cases": (36, 40), "include_user": False, "include_global": False, "terms_scope": "retrieved_records"},
]


def _difficulty_for_run(run_num: int) -> dict[str, Any]:
    for cfg in DIFFICULTY_CONFIG:
        if cfg["runs"][0] <= run_num <= cfg["runs"][1]:
            return cfg
    return DIFFICULTY_CONFIG[-1]


def _build_cases_for_run(
    run_num: int,
    base_queries: dict[str, list[dict[str, Any]]],
    extra_queries: list[dict[str, Any]] | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    if rng is None:
        rng = random.Random(SEED + run_num)
    diff = _difficulty_for_run(run_num)
    diff_name = diff["name"]
    token_budget = diff["token_budget"]
    min_cases, max_cases = diff["num_cases"]
    num_cases = rng.randint(min_cases, max_cases)
    include_user = diff["include_user"]
    include_global = diff["include_global"]
    terms_scope = diff["terms_scope"]

    pool: list[dict[str, Any]] = list(base_queries.get(diff_name, []))
    for easier in range(DIFFICULTY_CONFIG.index(diff)):
        easier_name = DIFFICULTY_CONFIG[easier]["name"]
        pool.extend(base_queries.get(easier_name, []))
    if extra_queries and run_num > 25:
        pool.extend(extra_queries)

    if not pool:
        pool = list(base_queries.get("trivial", []))

    rng.shuffle(pool)
    selected = pool[:num_cases]

    cases: list[dict[str, Any]] = []
    for i, q in enumerate(selected):
        case: dict[str, Any] = {
            "name": f"run{run_num:02d}_case{i+1:02d}_{diff_name}",
            "query_text": q["query_text"],
            "expected_source_keys": list(q.get("expected_source_keys", [])),
            "expected_terms": list(q.get("expected_terms", [])),
            "expected_terms_scope": terms_scope,
            "token_budget": token_budget,
        }
        if include_user:
            case["include_user_scope"] = True
            case["user_id"] = q.get("user_id", f"u_{(i % 20) + 1:02d}")
        else:
            case["include_user_scope"] = False
        if include_global:
            case["include_global_scope"] = True
        else:
            case["include_global_scope"] = False
        if "session_id" in q:
            case["session_id"] = q["session_id"]
        else:
            case["session_id"] = f"s_run{run_num:02d}"
        cases.append(case)

    suite_name = f"run_{run_num:02d}_{diff_name}"
    return {"name": suite_name, "cases": cases}


def _validate_terms(cases_data: dict[str, Any], dataset_path: Path) -> dict[str, Any]:
    all_texts: list[str] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                all_texts.append(rec.get("text", ""))

    combined_text = "\n".join(all_texts)
    cases = cases_data.get("cases", [])
    total_warnings = 0
    total_removed = 0
    total_substituted = 0

    for case in cases:
        original_terms = list(case.get("expected_terms", []))
        validated_terms: list[str] = []
        for term in original_terms:
            if term in combined_text:
                validated_terms.append(term)
            else:
                total_warnings += 1
                logger.warning(
                    "Term '%s' not found in dataset %s for case '%s'",
                    term, dataset_path.name, case.get("name", "unknown"),
                )
                found_sub = False
                for text in all_texts:
                    if term.split()[0] in text:
                        start = text.find(term.split()[0])
                        end = min(start + len(term) + 20, len(text))
                        fragment = text[start:end].rstrip(".,;:!?")
                        if fragment and fragment != term:
                            validated_terms.append(fragment)
                            total_substituted += 1
                            found_sub = True
                            break
                if not found_sub:
                    total_removed += 1
                    logger.warning(
                        "Removed term '%s' from case '%s' - no substitute found",
                        term, case.get("name", "unknown"),
                    )
        case["expected_terms"] = validated_terms

    return {
        "total_warnings": total_warnings,
        "total_removed": total_removed,
        "total_substituted": total_substituted,
    }


def generate_all_cases() -> None:
    rng = random.Random(SEED)

    dataset_map = {
        "series_a": DATASETS_DIR / "perturbed_dataset.jsonl",
        "series_b": DATASETS_DIR / "medical_dataset.jsonl",
        "series_c": DATASETS_DIR / "adversarial_dataset.jsonl",
    }

    for series_name, (base_queries, extra_queries, series_dir) in [
        ("series_a", (BASE_QUERIES_BY_DIFFICULTY, PERTURBED_EXTRA_QUERIES, CASES_DIR / "series_a")),
        ("series_b", (MEDICAL_QUERIES_BY_DIFFICULTY, None, CASES_DIR / "series_b")),
        ("series_c", (BASE_QUERIES_BY_DIFFICULTY, ADVERSARIAL_EXTRA_QUERIES, CASES_DIR / "series_c")),
    ]:
        series_dir.mkdir(parents=True, exist_ok=True)
        for run_num in range(1, 51):
            run_rng = random.Random(SEED + hash(series_name) + run_num)
            cases_data = _build_cases_for_run(run_num, base_queries, extra_queries, run_rng)
            out_path = series_dir / f"run_{run_num:02d}_cases.json"
            out_path.write_text(json.dumps(cases_data, indent=2, ensure_ascii=False), encoding="utf-8")

    dataset_path = dataset_map.get("series_a")
    if dataset_path and dataset_path.exists():
        for series_dir in [CASES_DIR / "series_a"]:
            for cases_file in sorted(series_dir.glob("run_*_cases.json")):
                cases_data = json.loads(cases_file.read_text(encoding="utf-8"))
                result = _validate_terms(cases_data, dataset_path)
                if result["total_warnings"] > 0:
                    logger.info(
                        "Validation %s: %d warnings, %d substituted, %d removed",
                        cases_file.name, result["total_warnings"],
                        result["total_substituted"], result["total_removed"],
                    )
                cases_file.write_text(
                    json.dumps(cases_data, indent=2, ensure_ascii=False), encoding="utf-8"
                )

    dataset_path = dataset_map.get("series_b")
    if dataset_path and dataset_path.exists():
        for series_dir in [CASES_DIR / "series_b"]:
            for cases_file in sorted(series_dir.glob("run_*_cases.json")):
                cases_data = json.loads(cases_file.read_text(encoding="utf-8"))
                result = _validate_terms(cases_data, dataset_path)
                if result["total_warnings"] > 0:
                    logger.info(
                        "Validation %s: %d warnings, %d substituted, %d removed",
                        cases_file.name, result["total_warnings"],
                        result["total_substituted"], result["total_removed"],
                    )
                cases_file.write_text(
                    json.dumps(cases_data, indent=2, ensure_ascii=False), encoding="utf-8"
                )

    dataset_path = dataset_map.get("series_c")
    if dataset_path and dataset_path.exists():
        for series_dir in [CASES_DIR / "series_c"]:
            for cases_file in sorted(series_dir.glob("run_*_cases.json")):
                cases_data = json.loads(cases_file.read_text(encoding="utf-8"))
                result = _validate_terms(cases_data, dataset_path)
                if result["total_warnings"] > 0:
                    logger.info(
                        "Validation %s: %d warnings, %d substituted, %d removed",
                        cases_file.name, result["total_warnings"],
                        result["total_substituted"], result["total_removed"],
                    )
                cases_file.write_text(
                    json.dumps(cases_data, indent=2, ensure_ascii=False), encoding="utf-8"
                )

    print("All 150 case files generated.")


if __name__ == "__main__":
    generate_all_cases()
