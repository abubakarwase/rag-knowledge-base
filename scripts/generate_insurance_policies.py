#!/usr/bin/env python3
"""Generate synthetic insurance policy PDFs for the local RAG knowledge base.

Produces a mixed book of business with full-looking policies, varied structures
and limits, and intentional good / messy / bad quality tiers. Offline only:
no LLM API calls. Text is extractable via pypdf.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Corpus mix (exactly 400 when --count 400)
# ---------------------------------------------------------------------------

LOB_COUNTS: dict[str, int] = {
    "auto": 80,
    "home": 70,
    "commercial": 60,
    "life": 40,
    "health": 40,
    "travel": 30,
    "workers_comp": 25,
    "umbrella": 25,
    "cyber": 15,
    "specialty": 15,
}

LOB_PREFIX: dict[str, str] = {
    "auto": "auto",
    "home": "home",
    "commercial": "cgl",
    "life": "life",
    "health": "hlth",
    "travel": "trvl",
    "workers_comp": "wc",
    "umbrella": "umbr",
    "cyber": "cyber",
    "specialty": "spec",
}

LOB_TITLES: dict[str, list[str]] = {
    "auto": [
        "Personal Automobile Policy",
        "Private Passenger Auto Insurance Agreement",
        "Motor Vehicle Insurance Policy",
    ],
    "home": [
        "Homeowners Insurance Policy",
        "Dwelling Fire Policy Form",
        "Renters Personal Property Policy",
    ],
    "commercial": [
        "Commercial General Liability Policy",
        "Businessowners Package Policy",
        "Commercial Property and Liability Policy",
    ],
    "life": [
        "Term Life Insurance Policy",
        "Whole Life Insurance Contract",
        "Universal Life Policy Form",
    ],
    "health": [
        "Group Medical Expense Policy",
        "Individual Health Insurance Policy",
        "Preferred Provider Organization Plan Document",
    ],
    "travel": [
        "Travel Protection Policy",
        "Trip Cancellation and Medical Policy",
        "Worldwide Travel Assistance Policy",
    ],
    "workers_comp": [
        "Workers Compensation and Employers Liability Policy",
        "Statutory Workers Compensation Policy",
        "Employers Liability Insurance Policy",
    ],
    "umbrella": [
        "Personal Umbrella Liability Policy",
        "Commercial Excess Liability Policy",
        "Umbrella Follow-Form Excess Policy",
    ],
    "cyber": [
        "Cyber Liability and Privacy Policy",
        "Technology Errors and Omissions with Cyber",
        "Data Breach and Network Security Policy",
    ],
    "specialty": [
        "Pet Health Insurance Policy",
        "Pleasure Boat Insurance Policy",
        "Motorcycle Insurance Policy",
    ],
}

CARRIERS = [
    "Summit Mutual Insurance Company",
    "Northbridge Assurance Group",
    "Cedar Ridge Underwriters",
    "Harborpoint Insurance Co.",
    "Atlas Shield Mutual",
    "Prairie Star Casualty",
    "Ironwood Property & Casualty",
    "Lumen Life & Health",
    "Cascade Indemnity Partners",
    "Silverleaf Specialty Insurance",
    "Mercantile Mutual Fire",
    "Oakmont Commercial Lines",
]

FIRST_NAMES = [
    "Jordan", "Avery", "Morgan", "Casey", "Riley", "Quinn", "Taylor", "Alex",
    "Sam", "Jamie", "Cameron", "Drew", "Harper", "Reese", "Skyler", "Parker",
    "Dana", "Leslie", "Pat", "Robin", "Chris", "Lee", "Kim", "Tracy",
]

LAST_NAMES = [
    "Nguyen", "Patel", "Garcia", "Brooks", "Keller", "Singh", "Okafor", "Walsh",
    "Chen", "Ramirez", "Foster", "Hughes", "Diaz", "Coleman", "Bennett", "Price",
    "Murphy", "Reed", "Bailey", "Hayes", "Griffin", "Russell", "Ortiz", "Kim",
]

BUSINESS_NAMES = [
    "Brightside Bakery LLC", "Harbor Logistics Inc.", "Maple Street Dental",
    "Cobalt Software Solutions", "Riverbend Construction Co.",
    "Northwind Warehousing", "Lumen Retail Group", "Pinecrest Clinics PLLC",
    "Summit Auto Body", "Greenfield Agritech", "Coastal Catering Partners",
    "Vector Machine Shop", "Elm & Oak Property Mgmt", "Blue Heron Hotels",
]

CITIES = [
    ("Austin", "TX"), ("Denver", "CO"), ("Portland", "OR"), ("Chicago", "IL"),
    ("Atlanta", "GA"), ("Boston", "MA"), ("Seattle", "WA"), ("Phoenix", "AZ"),
    ("Nashville", "TN"), ("Minneapolis", "MN"), ("Raleigh", "NC"),
    ("Columbus", "OH"), ("Salt Lake City", "UT"), ("Kansas City", "MO"),
]

STATES = [s for _, s in CITIES]

SECTION_LABEL_SETS = [
    {
        "declarations": "Declarations",
        "definitions": "Definitions",
        "coverages": "Insuring Agreements",
        "exclusions": "Exclusions",
        "conditions": "Conditions",
        "endorsements": "Endorsements",
        "schedule": "Schedule of Limits",
    },
    {
        "declarations": "Policy Declarations Page",
        "definitions": "Defined Terms",
        "coverages": "Coverage Parts",
        "exclusions": "What Is Not Covered",
        "conditions": "General Conditions",
        "endorsements": "Policy Endorsements and Riders",
        "schedule": "Limits Schedule",
    },
    {
        "declarations": "Part A - Declarations",
        "definitions": "Part B - Definitions",
        "coverages": "Part C - Coverages",
        "exclusions": "Part D - Exclusions",
        "conditions": "Part E - Conditions",
        "endorsements": "Part F - Endorsements",
        "schedule": "Part G - Schedule of Forms and Limits",
    },
]

QUALITY_WEIGHTS = {"good": 0.60, "messy": 0.25, "bad": 0.15}


@dataclass
class PolicyBlueprint:
    index: int
    lob: str
    quality_tier: str
    carrier: str
    title: str
    policy_number: str
    named_insured: str
    mailing_address: str
    effective: str
    expiration: str
    territory: str
    premium: str
    limits: dict[str, str]
    deductible: str
    section_labels: dict[str, str]
    section_order: list[str]
    specialty_subtype: str = ""
    notes: list[str] = field(default_factory=list)
    filename: str = ""


def money(rng: random.Random, low: int, high: int, step: int = 1000) -> str:
    value = rng.randrange(low, high + 1, step)
    return f"${value:,}"


def pick_quality(rng: random.Random) -> str:
    roll = rng.random()
    cumulative = 0.0
    for tier, weight in QUALITY_WEIGHTS.items():
        cumulative += weight
        if roll <= cumulative:
            return tier
    return "good"


def person_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def address(rng: random.Random) -> str:
    city, state = rng.choice(CITIES)
    street_n = rng.randint(100, 9899)
    street = rng.choice(
        ["Oak", "Maple", "Cedar", "Pine", "Lake", "Hill", "River", "Market"]
    )
    kind = rng.choice(["St", "Ave", "Blvd", "Rd", "Ln", "Dr"])
    zip_code = rng.randint(10001, 99950)
    return f"{street_n} {street} {kind}, {city}, {state} {zip_code}"


def policy_dates(rng: random.Random) -> tuple[str, str]:
    start = date(2024, 1, 1) + timedelta(days=rng.randint(0, 700))
    term_months = rng.choice([6, 12, 12, 12, 24])
    end = start + timedelta(days=30 * term_months)
    return start.isoformat(), end.isoformat()


def slug_carrier(carrier: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", carrier.lower()).strip("_")[:24]


def build_limits(lob: str, rng: random.Random, subtype: str = "") -> dict[str, str]:
    if lob == "auto":
        bi = rng.choice(["25/50", "50/100", "100/300", "250/500"])
        return {
            "Bodily Injury Liability": f"{bi} (thousands)",
            "Property Damage Liability": money(rng, 25_000, 100_000, 5_000),
            "Uninsured Motorist": bi,
            "Collision": money(rng, 15_000, 60_000, 5_000),
            "Comprehensive": money(rng, 15_000, 60_000, 5_000),
            "Medical Payments": money(rng, 1_000, 10_000, 1_000),
            "Rental Reimbursement": f"{money(rng, 30, 50, 5)}/day, max {rng.choice([30, 45])} days",
        }
    if lob == "home":
        dwelling = money(rng, 180_000, 950_000, 10_000)
        return {
            "Coverage A Dwelling": dwelling,
            "Coverage B Other Structures": "10% of Coverage A",
            "Coverage C Personal Property": "50% of Coverage A",
            "Coverage D Loss of Use": "20% of Coverage A",
            "Coverage E Personal Liability": money(rng, 100_000, 500_000, 50_000),
            "Coverage F Medical Payments to Others": money(rng, 1_000, 5_000, 1_000),
            "Ordinance or Law": rng.choice(["10%", "25%", "Not purchased"]),
        }
    if lob == "commercial":
        return {
            "Each Occurrence": money(rng, 500_000, 2_000_000, 100_000),
            "General Aggregate": money(rng, 1_000_000, 4_000_000, 250_000),
            "Products-Completed Operations Aggregate": money(rng, 1_000_000, 4_000_000, 250_000),
            "Personal and Advertising Injury": money(rng, 500_000, 2_000_000, 100_000),
            "Damage to Premises Rented to You": money(rng, 50_000, 300_000, 25_000),
            "Medical Expense": money(rng, 5_000, 15_000, 1_000),
            "Building Limit": money(rng, 250_000, 5_000_000, 50_000),
            "Business Personal Property": money(rng, 50_000, 1_500_000, 25_000),
        }
    if lob == "life":
        face = money(rng, 50_000, 2_000_000, 25_000)
        return {
            "Face Amount": face,
            "Accidental Death Benefit": rng.choice(["None", face, "50% of Face Amount"]),
            "Waiver of Premium": rng.choice(["Included", "Not elected"]),
            "Accelerated Death Benefit": "Up to 50% of Face Amount if terminal illness",
            "Policy Loan Interest Rate": f"{rng.choice([5, 6, 7, 8])}%",
        }
    if lob == "health":
        return {
            "Individual Deductible": money(rng, 500, 7_500, 250),
            "Family Deductible": money(rng, 1_000, 15_000, 500),
            "Out-of-Pocket Maximum Individual": money(rng, 3_000, 9_000, 500),
            "Out-of-Pocket Maximum Family": money(rng, 6_000, 18_000, 500),
            "Primary Care Copay": f"${rng.choice([15, 25, 35, 40])}",
            "Specialist Copay": f"${rng.choice([40, 50, 75, 90])}",
            "Emergency Room Copay": f"${rng.choice([150, 250, 350, 500])}",
            "Coinsurance": f"{rng.choice([10, 20, 30])}% after deductible",
        }
    if lob == "travel":
        return {
            "Trip Cancellation": money(rng, 1_000, 15_000, 500),
            "Trip Interruption": money(rng, 1_000, 15_000, 500),
            "Emergency Medical": money(rng, 25_000, 250_000, 25_000),
            "Emergency Evacuation": money(rng, 50_000, 500_000, 50_000),
            "Baggage Delay": money(rng, 100, 500, 50),
            "Baggage Loss": money(rng, 500, 3_000, 100),
            "Travel Delay": f"{money(rng, 100, 500, 50)} per day, max {rng.choice([3, 5, 7])} days",
        }
    if lob == "workers_comp":
        return {
            "Workers Compensation": "Statutory limits of the state of hire",
            "Employers Liability Bodily Injury by Accident": money(rng, 100_000, 1_000_000, 100_000),
            "Employers Liability Bodily Injury by Disease Each Employee": money(
                rng, 100_000, 1_000_000, 100_000
            ),
            "Employers Liability Bodily Injury by Disease Policy Limit": money(
                rng, 500_000, 2_000_000, 100_000
            ),
            "Experience Mod Factor": f"{rng.uniform(0.75, 1.35):.2f}",
        }
    if lob == "umbrella":
        return {
            "Each Occurrence": money(rng, 1_000_000, 10_000_000, 1_000_000),
            "Aggregate": money(rng, 1_000_000, 10_000_000, 1_000_000),
            "Retained Limit / SIR": money(rng, 0, 25_000, 5_000),
            "Underlying Auto Minimum": "100/300/100 or equivalent",
            "Underlying Home / CGL Minimum": money(rng, 300_000, 1_000_000, 100_000),
        }
    if lob == "cyber":
        return {
            "Network Security Liability Each Claim": money(rng, 250_000, 5_000_000, 250_000),
            "Privacy Liability Aggregate": money(rng, 250_000, 5_000_000, 250_000),
            "Breach Response Costs": money(rng, 50_000, 1_000_000, 50_000),
            "Business Interruption Waiting Period": f"{rng.choice([8, 12, 24])} hours",
            "Cyber Extortion": money(rng, 50_000, 1_000_000, 50_000),
            "Media Liability": money(rng, 100_000, 2_000_000, 100_000),
        }
    # specialty
    if subtype == "pet":
        return {
            "Annual Maximum": money(rng, 5_000, 30_000, 1_000),
            "Per-Incident Limit": money(rng, 2_000, 15_000, 500),
            "Reimbursement Percentage": f"{rng.choice([70, 80, 90])}%",
            "Annual Deductible": money(rng, 100, 1_000, 50),
            "Waiting Period Illness": f"{rng.choice([14, 30])} days",
            "Waiting Period Accident": f"{rng.choice([0, 3, 14])} days",
        }
    if subtype == "boat":
        return {
            "Hull Agreed Value": money(rng, 15_000, 350_000, 5_000),
            "Protection and Indemnity": money(rng, 100_000, 1_000_000, 50_000),
            "Medical Payments": money(rng, 1_000, 10_000, 1_000),
            "Uninsured Boater": money(rng, 50_000, 300_000, 25_000),
            "Trailer": money(rng, 1_000, 15_000, 500),
        }
    return {
        "Liability Each Occurrence": money(rng, 50_000, 500_000, 25_000),
        "Physical Damage": money(rng, 5_000, 40_000, 1_000),
        "Medical Payments": money(rng, 1_000, 5_000, 500),
        "Uninsured Motorist": money(rng, 25_000, 100_000, 25_000),
    }


def deductible_for(lob: str, rng: random.Random) -> str:
    if lob in {"life"}:
        return "Not applicable"
    if lob == "health":
        return "See schedule (medical deductible)"
    if lob == "workers_comp":
        return rng.choice(["None", "$500 medical deductible where permitted"])
    if lob == "cyber":
        return money(rng, 2_500, 50_000, 2_500)
    if lob == "umbrella":
        return "Self-insured retention as scheduled"
    return money(rng, 250, 5_000, 250)


def build_blueprint(index: int, lob: str, rng: random.Random) -> PolicyBlueprint:
    quality = pick_quality(rng)
    carrier = rng.choice(CARRIERS)
    title = rng.choice(LOB_TITLES[lob])
    subtype = ""
    if lob == "specialty":
        subtype = rng.choice(["pet", "boat", "motorcycle"])
        title = {
            "pet": "Pet Health Insurance Policy",
            "boat": "Pleasure Boat Insurance Policy",
            "motorcycle": "Motorcycle Insurance Policy",
        }[subtype]
    if lob in {"commercial", "workers_comp", "cyber"} or (
        lob == "umbrella" and rng.random() < 0.4
    ):
        insured = rng.choice(BUSINESS_NAMES)
    else:
        insured = person_name(rng)
    effective, expiration = policy_dates(rng)
    city, state = rng.choice(CITIES)
    labels = dict(rng.choice(SECTION_LABEL_SETS))
    base_order = [
        "declarations",
        "schedule",
        "definitions",
        "coverages",
        "exclusions",
        "conditions",
        "endorsements",
    ]
    # Rotate and occasionally swap adjacent sections for structural variety
    rotate = rng.randint(0, 3)
    order = base_order[rotate:] + base_order[:rotate]
    if rng.random() < 0.45:
        i = rng.randint(1, len(order) - 2)
        order[i], order[i + 1] = order[i + 1], order[i]

    prefix = LOB_PREFIX[lob]
    pol_num = f"{prefix.upper()}-{rng.randint(100000, 999999)}-{rng.randint(10, 99)}"
    fname = f"{prefix}_pol_{index:04d}_{slug_carrier(carrier)}.pdf"

    bp = PolicyBlueprint(
        index=index,
        lob=lob,
        quality_tier=quality,
        carrier=carrier,
        title=title,
        policy_number=pol_num,
        named_insured=insured,
        mailing_address=address(rng),
        effective=effective,
        expiration=expiration,
        territory=f"{state} and contiguous United States" if rng.random() < 0.7 else "United States and its territories",
        premium=money(rng, 180, 48_000, 10),
        limits=build_limits(lob, rng, subtype),
        deductible=deductible_for(lob, rng),
        section_labels=labels,
        section_order=order,
        specialty_subtype=subtype,
        filename=fname,
    )
    if quality == "bad":
        bp.notes.append("intentional defects applied")
    return bp


def lob_coverages(bp: PolicyBlueprint, rng: random.Random) -> list[str]:
    lob = bp.lob
    if lob == "auto":
        return [
            f"We will pay damages for bodily injury or property damage for which any insured becomes legally responsible because of an auto accident. The most we will pay is shown on the {bp.section_labels['schedule']}.",
            "Collision coverage pays for direct and accidental loss to your covered auto caused by collision with another object or upset of your covered auto, less the deductible.",
            "Other Than Collision (Comprehensive) covers loss caused by missiles, falling objects, fire, theft, explosion, earthquake, windstorm, hail, water, flood, malicious mischief, riot, or contact with a bird or animal.",
            "Uninsured Motorist coverage pays compensatory damages an insured is legally entitled to recover from the owner or operator of an uninsured motor vehicle because of bodily injury.",
            "Medical Payments coverage pays reasonable expenses incurred for necessary medical and funeral services because of bodily injury caused by an accident.",
        ]
    if lob == "home":
        return [
            "We insure against direct physical loss to property described in Coverages A and B caused by a Peril Insured Against, unless excluded.",
            "Coverage C Personal Property covers personal property owned or used by an insured while it is anywhere in the world, subject to special limits for money, securities, jewelry, and firearms.",
            "Coverage D Loss of Use pays the necessary increase in living expenses when a covered loss makes the residence premises uninhabitable.",
            "Coverage E Personal Liability pays sums an insured becomes legally obligated to pay as damages because of bodily injury or property damage caused by an occurrence.",
            "Coverage F Medical Payments to Others pays medical expenses for bodily injury caused by an animal owned by or in the care of an insured, or arising out of a condition on the insured location.",
        ]
    if lob == "commercial":
        return [
            "We will pay those sums that the insured becomes legally obligated to pay as damages because of bodily injury or property damage to which this insurance applies. This insurance applies only to bodily injury or property damage that occurs during the policy period.",
            "Personal and Advertising Injury coverage applies to offenses arising out of your business, including false arrest, malicious prosecution, wrongful eviction, slander, libel, and infringement of copyright in your advertisement.",
            "Products-Completed Operations coverage applies to bodily injury or property damage occurring away from premises you own or rent and arising out of your product or completed work.",
            "We cover direct physical loss of or damage to Covered Property at the premises described in the Declarations caused by or resulting from a Covered Cause of Loss.",
            "Business Income and Extra Expense may apply when a Covered Cause of Loss causes a necessary suspension of operations, subject to the waiting period and period of restoration.",
        ]
    if lob == "life":
        return [
            f"Upon due proof of death of the Insured while this policy is in force, we will pay the Face Amount of {bp.limits.get('Face Amount', 'the scheduled amount')} to the beneficiary, less any policy loan and unpaid premium.",
            "If the Accidental Death Benefit is in force and death results from accidental bodily injury independent of all other causes within 90 days of the injury, we will pay the Accidental Death Benefit in addition to the Face Amount.",
            "The Accelerated Death Benefit allows a one-time advance of a portion of the Face Amount if the Insured is diagnosed with a terminal illness with life expectancy of 12 months or less.",
            "Premiums are payable in advance. Grace period is 31 days. If premium is not paid within the grace period, the policy will lapse subject to any nonforfeiture options.",
            "Suicide within two years of the issue date limits recovery to return of premiums paid, without interest.",
        ]
    if lob == "health":
        return [
            "We provide coverage for Medically Necessary services and supplies when furnished by a Participating Provider, subject to the deductible, copayments, coinsurance, and out-of-pocket maximums in the schedule.",
            "Preventive services rated A or B by the U.S. Preventive Services Task Force are covered at 100% when received in-network, without application of the deductible.",
            "Emergency services are covered whether received in-network or out-of-network. The emergency room copay is waived if admitted.",
            "Prescription drugs are covered according to the formulary tiers. Prior authorization may be required for specialty medications.",
            "Mental health and substance use disorder benefits are provided in parity with medical and surgical benefits under applicable federal law.",
        ]
    if lob == "travel":
        return [
            "Trip Cancellation reimburses prepaid, non-refundable trip costs if you cancel for a covered reason, including illness of a traveling companion, jury duty, or home uninhabitability due to natural disaster.",
            "Emergency Medical coverage pays usual and customary charges for medically necessary treatment of a covered injury or sickness that occurs during the covered trip.",
            "Emergency Evacuation pays for transportation to the nearest adequate medical facility when ordered by a physician and arranged by the assistance provider.",
            "Baggage Delay pays for essential personal effects if checked baggage is delayed by a common carrier for more than the waiting period shown in the schedule.",
            "Travel Delay pays reasonable accommodations and meal expenses when a common carrier delay exceeds the scheduled waiting period for a covered reason.",
        ]
    if lob == "workers_comp":
        return [
            "Part One - Workers Compensation Insurance: we will pay promptly when due the benefits required of you by the workers compensation law of any state listed in Item 3.A. of the Information Page.",
            "Part Two - Employers Liability Insurance: we will pay damages you become legally obligated to pay because of bodily injury by accident or disease to your employee arising out of and in the course of employment.",
            "We have the right and duty to defend any claim, proceeding, or suit against you for benefits payable by this insurance.",
            "Recovery from others: we have your rights and the rights of persons entitled to benefits to recover payments from anyone liable for an injury covered by this policy.",
            "Premium is determined by applying rate classifications to remuneration. Final premium is based on actual payroll audit after policy expiration.",
        ]
    if lob == "umbrella":
        return [
            "We will pay damages in excess of the retained limit for bodily injury, property damage, or personal injury to which this insurance applies, caused by an occurrence.",
            "This policy follows form to the underlying insurance scheduled in the Declarations, except where this policy provides broader coverage or different terms.",
            "Defense costs are outside the limits unless the underlying policy treats defense within limits and this policy so states by endorsement.",
            "Coverage territory is worldwide, provided suit is brought in the United States, its territories, or Canada, unless otherwise endorsed.",
            "You must maintain the underlying insurance at the required minimum limits. Failure to maintain underlying insurance does not invalidate this policy, but we will only be liable to the same extent as if underlying were maintained.",
        ]
    if lob == "cyber":
        return [
            "Network Security Liability covers damages and claim expenses arising from a security failure that results in unauthorized access to or use of a computer system.",
            "Privacy Liability covers damages arising from a privacy event, including failure to protect confidential information or violation of a privacy regulation.",
            "Breach Response Costs cover computer forensic, legal, notification, call center, and credit monitoring expenses following a privacy event, subject to the sublimit.",
            "Business Interruption covers loss of income and extra expense resulting from a security failure that causes a total or partial interruption of the insured's computer system after the waiting period.",
            "Cyber Extortion covers ransom amounts and related expenses paid with our prior consent in response to a credible threat to the computer system or confidential information.",
        ]
    if bp.specialty_subtype == "pet":
        return [
            "We reimburse covered veterinary expenses for accidents and illnesses that first occur or show clinical signs after the applicable waiting period, subject to the annual maximum and deductible.",
            "Covered expenses include examination fees, diagnostics, surgery, hospitalization, prescription medications, and physical therapy prescribed by a licensed veterinarian.",
            "Hereditary and congenital conditions may be covered after a continuous enrollment period if shown as included on the declarations.",
            "Pre-existing conditions are excluded. A condition is pre-existing if signs or symptoms were noted prior to the effective date or during a waiting period.",
            "Claims must be submitted within 90 days of treatment with itemized invoices and medical records upon request.",
        ]
    if bp.specialty_subtype == "boat":
        return [
            "Hull coverage pays for accidental physical loss of or damage to the scheduled vessel, including equipment usually aboard, subject to the deductible and agreed value.",
            "Protection and Indemnity covers your legal liability for bodily injury or property damage arising out of ownership, maintenance, or use of the insured vessel.",
            "Medical Payments covers reasonable medical expenses for persons injured while in, upon, boarding, or leaving the insured vessel.",
            "Lay-up warranty: if a lay-up period is scheduled, the vessel must be laid up and out of commission ashore or in a safe berth for that period.",
            "Navigation warranty: coverage applies only within the navigation limits shown on the declarations.",
        ]
    return [
        "Liability coverage pays damages for bodily injury or property damage for which an insured becomes legally responsible because of an occurrence arising from ownership or use of the scheduled motorcycle.",
        "Physical Damage covers collision and comprehensive losses to the scheduled motorcycle, less the deductible.",
        "Medical Payments covers reasonable medical expenses for an insured who sustains bodily injury while occupying the motorcycle.",
        "Uninsured Motorist coverage applies as shown in the schedule when the at-fault party has no liability insurance or less than the required minimum.",
        "Accessories and custom parts are covered only up to the sublimit shown, unless scheduled by endorsement.",
    ]


def lob_exclusions(bp: PolicyBlueprint, rng: random.Random) -> list[str]:
    common = [
        "Intentional acts by or at the direction of an insured, except for acts of self-defense.",
        "War, including undeclared war, civil war, insurrection, rebellion, or revolution.",
        "Nuclear hazard, radiation, or radioactive contamination, whether controlled or uncontrolled.",
        "Wear and tear, marring, deterioration, mechanical breakdown, or latent defect.",
    ]
    specific: dict[str, list[str]] = {
        "auto": [
            "Using a vehicle without a reasonable belief that you are entitled to do so.",
            "Bodily injury to an employee of an insured arising out of employment, except domestic employees not entitled to workers compensation.",
            "Damage to property owned by, rented to, or transported by an insured.",
            "Racing, speed contests, or driver training on a track designed for racing.",
        ],
        "home": [
            "Flood, surface water, waves, tidal water, overflow of a body of water, or spray from any of these.",
            "Earth movement, including earthquake, landslide, mine subsidence, and earth sinking.",
            "Mold, fungus, or wet rot, except as limited by any mold endorsement if attached.",
            "Business pursuits of an insured, except incidental activities and occupancy of the residence premises as a dwelling.",
        ],
        "commercial": [
            "Expected or intended injury from the standpoint of the insured.",
            "Pollution, including any request or order to test for, monitor, clean up, or neutralize pollutants, except as limited by the hostile fire exception.",
            "Employment-related practices, including wrongful termination, discrimination, and harassment.",
            "Aircraft, auto, or watercraft liability except as specifically provided.",
        ],
        "life": [
            "Death by suicide within the contestable period as stated in the policy.",
            "Misrepresentation of material facts in the application, subject to the incontestability clause.",
            "Active duty military service in a declared war, if so endorsed as excluded.",
        ],
        "health": [
            "Services that are not Medically Necessary or are experimental or investigational.",
            "Cosmetic surgery except as required to restore function after an accidental injury.",
            "Care for injuries covered by workers compensation or similar law.",
            "Services received outside the service area except for emergencies and urgently needed care.",
        ],
        "travel": [
            "Pre-existing medical conditions unless the waiver is purchased and eligibility criteria are met.",
            "Trips undertaken against medical advice.",
            "Losses arising from participation in professional athletics or extreme sports listed in the policy.",
            "Failure of a travel supplier due to financial insolvency if not covered by a supplier default endorsement.",
        ],
        "workers_comp": [
            "Liability assumed under a contract, except a warranty of fitness of your products.",
            "Punitive or exemplary damages, where not required by law.",
            "Bodily injury intentionally caused or aggravated by you.",
            "Operations subject to federal employers liability or maritime law unless endorsed.",
        ],
        "umbrella": [
            "Obligations under workers compensation, disability benefits, or unemployment compensation law.",
            "Property damage to property you own, rent, or occupy.",
            "Personal injury arising out of oral or written publication of material first published before the policy period.",
            "Aircraft liability unless scheduled underlying aircraft liability is maintained and endorsed.",
        ],
        "cyber": [
            "Bodily injury or property damage, except data and software as defined.",
            "Infrastructure failure of internet, telecommunications, or utilities not under your control.",
            "War, including cyber operations that constitute war under the applicable endorsement.",
            "Theft of money or securities via social engineering unless a social engineering endorsement is attached.",
        ],
        "specialty": [
            "Pre-existing conditions and elective procedures (pet policies).",
            "Commercial use, charter, or hire of the vessel or motorcycle unless endorsed.",
            "Operating under the influence of alcohol or drugs.",
            "Wear and tear, gradual deterioration, and vermin damage.",
        ],
    }
    items = specific.get(bp.lob, specific["specialty"]) + common
    rng.shuffle(items)
    return items[: rng.randint(5, 8)]


def lob_definitions(bp: PolicyBlueprint) -> list[tuple[str, str]]:
    shared = [
        (
            "You / Your",
            "The Named Insured shown in the Declarations and, if an individual, that person's resident spouse.",
        ),
        (
            "We / Us / Our",
            f"{bp.carrier}, the company providing this insurance.",
        ),
        (
            "Occurrence",
            "An accident, including continuous or repeated exposure to substantially the same general harmful conditions.",
        ),
        (
            "Policy Period",
            f"The period from {bp.effective} to {bp.expiration} at 12:01 a.m. standard time at the address of the Named Insured.",
        ),
    ]
    extra: dict[str, list[tuple[str, str]]] = {
        "auto": [
            ("Covered Auto", "Any vehicle shown in the Declarations, a newly acquired auto, and any temporary substitute auto."),
            ("Insured", "You, any family member, and any person using your covered auto with your permission."),
        ],
        "home": [
            ("Residence Premises", "The one-family dwelling where you reside, shown as the residence premises in the Declarations."),
            ("Insured Location", "The residence premises, vacant land you own, and premises used as a residence acquired during the policy period."),
        ],
        "commercial": [
            ("Employee", "A person employed by you, including leased workers, but not temporary workers unless endorsed."),
            ("Your Product", "Goods or products manufactured, sold, handled, distributed, or disposed of by you."),
        ],
        "life": [
            ("Insured", "The person whose life is covered under this policy, as named in the Declarations."),
            ("Beneficiary", "The person or entity designated to receive the death benefit."),
        ],
        "health": [
            ("Medically Necessary", "Services appropriate for the diagnosis or treatment of a condition in accordance with generally accepted standards of medical practice."),
            ("Participating Provider", "A provider that has a contract with us to furnish covered services at negotiated rates."),
        ],
        "travel": [
            ("Covered Trip", "A trip for which coverage has been purchased and premiums paid, departing from and returning to your home."),
            ("Traveling Companion", "A person sharing travel arrangements with you and traveling on the same Covered Trip."),
        ],
        "workers_comp": [
            ("State", "Any state of the United States of America and the District of Columbia."),
            ("Workers Compensation Law", "The workers or workmen's compensation law and occupational disease law of each state listed."),
        ],
        "umbrella": [
            ("Retained Limit", "The scheduled self-insured retention or the applicable limits of underlying insurance, whichever applies."),
            ("Underlying Insurance", "The policies listed in the Schedule of Underlying Insurance."),
        ],
        "cyber": [
            ("Computer System", "Computer hardware, software, firmware, and networks under your ownership or operational control."),
            ("Confidential Information", "Non-public information in your care that is subject to a confidentiality or privacy obligation."),
        ],
        "specialty": [
            ("Scheduled Property", "The pet, vessel, or motorcycle described in the Declarations."),
            ("Insured", "You and any person using the scheduled property within the scope of permission granted."),
        ],
    }
    return shared + extra.get(bp.lob, extra["specialty"])


def lob_conditions(bp: PolicyBlueprint) -> list[str]:
    return [
        "Duties After Loss: promptly notify us, protect property from further damage, submit a signed proof of loss within 60 days if requested, and cooperate in investigation and settlement.",
        "Other Insurance: if other valid insurance applies, this policy pays on an excess basis unless the other insurance is written specifically as excess over this policy.",
        "Concealment or Fraud: we do not provide coverage if an insured has intentionally concealed or misrepresented a material fact relating to this insurance.",
        "Cancellation: we may cancel for nonpayment of premium with at least 10 days notice, or for other reasons permitted by law with the notice period required in your state.",
        "Changes: this policy contains all agreements between you and us. Its terms may be changed only by endorsement issued by us.",
        f"Premium: the total premium for this policy period is {bp.premium}, payable according to the billing plan selected. Taxes and surcharges may apply.",
        "Appraisal: if you and we fail to agree on the amount of loss, either may demand appraisal. Each selects an appraiser; the appraisers select an umpire.",
        "Subrogation: if we pay a loss, we are entitled to your rights of recovery against third parties. You must do nothing after loss to impair those rights.",
        "Assignment: your interest under this policy may not be assigned without our written consent.",
        "Conformity to Statute: any provision that conflicts with the law of the state where the policy is issued is amended to conform to the minimum requirements of that law.",
    ]


def lob_endorsements(bp: PolicyBlueprint, rng: random.Random) -> list[str]:
    pool = [
        f"Endorsement {rng.randint(1000, 9999)}: Additional Insured - {person_name(rng)} is added as an additional insured, but only with respect to liability arising out of your ownership or use of the covered property.",
        f"Endorsement {rng.randint(1000, 9999)}: Increased Limits - the limit for the coverage part identified in the schedule is increased as shown; all other terms remain unchanged.",
        f"Endorsement {rng.randint(1000, 9999)}: Deductible Amendment - the deductible applicable to physical damage or property coverage is amended to {bp.deductible}.",
        f"Endorsement {rng.randint(1000, 9999)}: Exclusion - Absolute Pollution - pollution exclusion is restated without the hostile fire exception for the operations described.",
        f"Endorsement {rng.randint(1000, 9999)}: Notice of Cancellation to Certificate Holders - we will mail notice to certificate holders listed on file at least 30 days before cancellation, except 10 days for nonpayment.",
        f"Endorsement {rng.randint(1000, 9999)}: Territory Extension - coverage territory is extended to include incidental travel to Mexico within {rng.choice([25, 50, 100])} miles of the U.S. border for a period not exceeding {rng.choice([7, 14, 30])} days.",
        f"Endorsement {rng.randint(1000, 9999)}: Named Driver / Operator Restriction - coverage for physical damage while the scheduled vehicle is operated by an excluded driver is void.",
        f"Endorsement {rng.randint(1000, 9999)}: Waiver of Subrogation - we waive rights of recovery against the person or organization shown in the schedule, but only to the extent you are required to waive such rights by written contract.",
    ]
    rng.shuffle(pool)
    count = rng.randint(2, 5)
    endorsements = pool[:count]
    if bp.quality_tier == "bad" and rng.random() < 0.7:
        endorsements.append(
            f"Endorsement {rng.randint(1000, 9999)}: [PLACEHOLDER - INSERT CLIENT SCHEDULE] coverage applies as TBD pending underwriting review."
        )
        if rng.random() < 0.5:
            endorsements[-1] = endorsements[-1][:80] + " ..."
    return endorsements


def intro_paragraphs(bp: PolicyBlueprint) -> list[str]:
    return [
        f"This policy is a legal contract between you and {bp.carrier}. In return for payment of the premium and subject to all terms of this policy, we agree to provide the insurance described.",
        f"Please read your policy carefully. The Declarations show who is insured, the policy period, the covered locations or vehicles, and the limits of insurance. Keep this document with your records.",
        f"Policy Number {bp.policy_number} is issued to {bp.named_insured}. Coverage applies only during the Policy Period and within the Coverage Territory unless an endorsement states otherwise.",
    ]


def apply_messy_text(text: str, rng: random.Random) -> str:
    # Light typos and awkward phrasing; still extractable English.
    replacements = [
        ("the insured", "the  insured"),
        ("coverage", "coverge"),
        ("liability", "liabilty"),
        ("because of", "becuase of"),
        ("occurrence", "occurence"),
        ("necessary", "neccessary"),
        ("policy", "polcy"),
        ("damage", "damge"),
    ]
    out = text
    for old, new in replacements:
        if old in out.lower() and rng.random() < 0.35:
            # case-sensitive simple replace of first occurrence variant
            pattern = re.compile(re.escape(old), re.IGNORECASE)
            out = pattern.sub(new, out, count=1)
    if rng.random() < 0.3:
        out = out.replace(". ", ".  ")
    if rng.random() < 0.2:
        out = out + " See also section above regarding same."
    return out


def apply_quality(
    bp: PolicyBlueprint,
    sections: dict[str, list[str]],
    rng: random.Random,
) -> dict[str, list[str]]:
    result = {k: list(v) for k, v in sections.items()}
    if bp.quality_tier == "good":
        return result

    if bp.quality_tier == "messy":
        for key in list(result.keys()):
            result[key] = [apply_messy_text(p, rng) for p in result[key]]
            if key == "coverages" and rng.random() < 0.5 and result[key]:
                # Redundant clause
                result[key].append(result[key][0])
        return result

    # bad tier
    droppable = [k for k in result if k not in {"declarations", "schedule"}]
    if droppable and rng.random() < 0.85:
        dropped = rng.choice(droppable)
        result.pop(dropped, None)
        bp.notes.append(f"missing_section:{dropped}")

    # Contradictory limit vs declarations schedule text
    if bp.limits:
        first_key = next(iter(bp.limits))
        wrong = money(rng, 1_000, 9_000, 500)
        result.setdefault("coverages", []).append(
            f"NOTE: For {first_key}, the limit applicable under this coverage part is {wrong}, "
            f"notwithstanding any higher amount shown on the declarations schedule."
        )
        bp.notes.append("contradictory_limit")

    # Wrong cross-reference
    result.setdefault("conditions", []).append(
        f"Refer to Section Z-{rng.randint(10, 99)} for claim reporting procedures "
        "(section numbering may not match this booklet)."
    )
    bp.notes.append("bad_cross_ref")

    # Garble one schedule-related sentence
    if "schedule" in result and result["schedule"]:
        garbled = result["schedule"][0]
        garbled = re.sub(r"\d", lambda m: str((int(m.group(0)) + 3) % 10), garbled, count=4)
        result["schedule"][0] = garbled + " |||| #REF!"
        bp.notes.append("garbled_schedule")

    for key in list(result.keys()):
        if rng.random() < 0.4:
            result[key] = [apply_messy_text(p, rng) for p in result[key]]

    return result


def extra_depth_paragraphs(bp: PolicyBlueprint, rng: random.Random) -> list[str]:
    """Add LOB-specific depth so PDFs land in the planned page ranges."""
    target_pages = {
        "auto": rng.randint(4, 8),
        "home": rng.randint(5, 10),
        "commercial": rng.randint(6, 12),
        "life": rng.randint(4, 8),
        "health": rng.randint(5, 10),
        "travel": rng.randint(3, 6),
        "workers_comp": rng.randint(5, 9),
        "umbrella": rng.randint(4, 7),
        "cyber": rng.randint(5, 8),
        "specialty": rng.randint(3, 6),
    }.get(bp.lob, 5)
    # Roughly ~2-3 paragraphs per extra page beyond a thin 2-page base
    n = max(4, (target_pages - 2) * 3)
    templates = [
        (
            "Claims Cooperation: You must authorize us to obtain records and information "
            "as we reasonably require. Failure to cooperate may result in denial of the claim "
            f"under policy {bp.policy_number}."
        ),
        (
            "Inspection and Audit: We have the right but not the duty to inspect your property "
            "and operations. Our inspections are for underwriting purposes and are not safety "
            "inspections or warranties of conditions."
        ),
        (
            "Liberalization: If we adopt a revision that would broaden coverage under this "
            "edition without additional premium, that broadened coverage will apply to this "
            "policy as of the date the revision is effective in your state."
        ),
        (
            "Transfer of Rights: If we make a payment under this policy and the person or "
            "organization to or for whom payment was made has rights to recover damages from "
            "another, those rights are transferred to us to the extent of our payment."
        ),
        (
            "Abandonment: There can be no abandonment of any property to us. You retain title "
            "and control of damaged property unless we elect to take ownership after payment of "
            "a total loss."
        ),
        (
            "Pair or Set: In case of loss to a pair or set we may repair or replace any part "
            "to restore the pair or set to its value before the loss, or pay the difference "
            "between the actual cash value of the property before and after the loss."
        ),
        (
            "Recovered Property: If you or we recover any property for which we have made "
            "payment, you or we will notify the other of the recovery. At your option, the "
            "property will be returned to you or become our property."
        ),
        (
            "No Benefit to Bailee: This insurance shall not benefit any carrier or other "
            "bailee for hire. You may not assign rights under this policy to a bailee."
        ),
        (
            f"Premium Audit Period: The premium shown as {bp.premium} is an estimated premium. "
            "We may examine and audit your books and records as they relate to this policy "
            "during the policy period and within three years after its expiration."
        ),
        (
            "Separation of Insureds: Except with respect to the Limits of Insurance and any "
            "rights or duties specifically assigned to the first Named Insured, this insurance "
            "applies as if each Named Insured were the only Named Insured."
        ),
        (
            "Knowledge of Occurrence: Knowledge of an occurrence by your agent, servant, or "
            "employee shall not constitute knowledge by you unless received by a person "
            "designated to give notice of claims to us."
        ),
        (
            "Bankruptcy: Bankruptcy or insolvency of an insured or of the insured's estate "
            "will not relieve us of our obligations under this policy."
        ),
        (
            "Legal Action Against Us: No one may bring a legal action against us under this "
            "policy unless there has been full compliance with all terms, and the action is "
            "brought within the time allowed by the applicable statute of limitations."
        ),
        (
            f"Coverage Territory Clarification: For purposes of this policy, the Coverage "
            f"Territory is {bp.territory}. Suits must be brought in a forum permitted by the "
            "policy conditions unless an endorsement expands venue."
        ),
        (
            "Electronic Data: Loss to electronic data is covered only to the extent expressly "
            "provided. The cost to research, replace, or restore electronic data is subject "
            "to any applicable sublimit and deductible."
        ),
        (
            "Voluntary Payments: No insured will, except at that insured's own cost, "
            "voluntarily make a payment, assume any obligation, or incur any expense other "
            "than for first aid, without our consent."
        ),
        (
            "Primary and Noncontributory: When required by written contract executed prior to "
            "loss, this insurance is primary and we will not seek contribution from other "
            "insurance available to an additional insured, except as excess over this policy."
        ),
        (
            "Notice of Claim: Written notice should include the policy number, named insured, "
            "time, place, and circumstances of the event, names of injured persons and "
            f"witnesses, and the nature of any claim. Send notice to {bp.carrier} Claims."
        ),
        (
            "Mortgagee Clause: Loss shall be payable to any mortgagee named in the Declarations "
            "as interest may appear, subject to the standard mortgagee provisions of this form."
        ),
        (
            "Vacancy: If the building where loss occurs has been vacant for more than 60 "
            "consecutive days before loss, we do not cover loss from vandalism, sprinkler "
            "leakage, building glass breakage, water damage, theft, or attempted theft, "
            "unless otherwise endorsed."
        ),
    ]
    rng.shuffle(templates)
    chosen = templates[:n]
    if bp.quality_tier == "messy":
        chosen = [apply_messy_text(p, rng) for p in chosen]
    if bp.quality_tier == "bad" and chosen:
        chosen[-1] = (
            "TODO: underwriter to confirm final wording prior to bind. "
            "XXXX insert state amendatory here. "
            + chosen[-1][:120]
        )
    return chosen


def build_section_content(bp: PolicyBlueprint, rng: random.Random) -> dict[str, list[str]]:
    decl = [
        f"Named Insured: {bp.named_insured}",
        f"Mailing Address: {bp.mailing_address}",
        f"Policy Number: {bp.policy_number}",
        f"Insurance Company: {bp.carrier}",
        f"Policy Period: {bp.effective} to {bp.expiration}",
        f"Coverage Territory: {bp.territory}",
        f"Total Premium: {bp.premium}",
        f"Deductible / Retention: {bp.deductible}",
        f"Form Edition: {bp.lob.upper()}-FORM-{rng.randint(2018, 2025)}-{rng.choice(['01', '07', '12'])}",
        *intro_paragraphs(bp),
    ]
    schedule = [
        "The following limits of insurance apply, subject to all policy terms:",
    ] + [f"{k}: {v}" for k, v in bp.limits.items()]
    schedule.append(
        "Unless otherwise stated, the limits shown above are the most we will pay "
        "regardless of the number of insureds, claims, or vehicles/locations involved."
    )
    definitions = [f"{term}: {defn}" for term, defn in lob_definitions(bp)]
    coverages = lob_coverages(bp, rng)
    coverages.extend(
        [
            f"All payments under this policy reduce the applicable limit unless the form expressly states that defense costs are outside the limits. Applicable form: {bp.lob.upper()}-COV-{rng.randint(100, 999)}.",
            "Where a sublimit applies, that sublimit is part of, and not in addition to, the limit of insurance shown for the coverage part.",
            "Coverage is primary unless otherwise stated. When this policy is excess, we will have no duty to defend if any other insurer has a duty to defend.",
        ]
    )
    exclusions = lob_exclusions(bp, rng)
    conditions = lob_conditions(bp)
    rng.shuffle(conditions)
    conditions = conditions[: rng.randint(6, 10)]
    conditions.extend(extra_depth_paragraphs(bp, rng))
    endorsements = lob_endorsements(bp, rng)

    sections = {
        "declarations": decl,
        "schedule": schedule,
        "definitions": definitions,
        "coverages": coverages,
        "exclusions": exclusions,
        "conditions": conditions,
        "endorsements": endorsements,
    }
    return apply_quality(bp, sections, rng)


def make_styles(quality: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    title_size = 14 if quality != "messy" else 12
    body_leading = 14 if quality == "good" else 12
    styles = {
        "title": ParagraphStyle(
            "PolTitle",
            parent=base["Heading1"],
            fontSize=title_size,
            alignment=TA_CENTER,
            spaceAfter=8,
            textColor=colors.HexColor("#1a1a1a"),
        ),
        "subtitle": ParagraphStyle(
            "PolSub",
            parent=base["Normal"],
            fontSize=10,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "PolH2",
            parent=base["Heading2"],
            fontSize=11 if quality == "good" else 10,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#222222"),
        ),
        "body": ParagraphStyle(
            "PolBody",
            parent=base["Normal"],
            fontSize=9 if quality != "bad" else 8,
            leading=body_leading,
            alignment=TA_JUSTIFY if quality == "good" else TA_LEFT,
            spaceAfter=6,
        ),
        "meta": ParagraphStyle(
            "PolMeta",
            parent=base["Normal"],
            fontSize=8,
            leading=11,
            spaceAfter=3,
        ),
        "footer": ParagraphStyle(
            "PolFooter",
            parent=base["Normal"],
            fontSize=7,
            alignment=TA_CENTER,
            textColor=colors.grey,
        ),
    }
    return styles


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_pdf(bp: PolicyBlueprint, out_path: Path, rng: random.Random) -> int:
    sections = build_section_content(bp, rng)
    styles = make_styles(bp.quality_tier)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=f"{bp.title} - {bp.policy_number}",
        author=bp.carrier,
    )
    story: list[Any] = []
    story.append(Paragraph(escape_xml(bp.carrier.upper()), styles["subtitle"]))
    story.append(Paragraph(escape_xml(bp.title), styles["title"]))
    story.append(
        Paragraph(
            escape_xml(
                f"Policy No. {bp.policy_number} | Effective {bp.effective} | "
                f"Expires {bp.expiration}"
            ),
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 6))

    # Limits table near front for scannability (real-world declarations feel)
    table_data = [["Coverage / Limit Item", "Amount"]]
    for k, v in bp.limits.items():
        display_v = v
        if bp.quality_tier == "bad" and rng.random() < 0.25:
            display_v = f"{v} / #ERR"
        table_data.append([k, display_v])
    table = Table(table_data, colWidths=[4.2 * inch, 2.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 10))

    for key in bp.section_order:
        if key not in sections:
            continue
        label = bp.section_labels.get(key, key.title())
        story.append(Paragraph(escape_xml(label), styles["h2"]))
        for i, para in enumerate(sections[key], start=1):
            prefix = f"{i}. " if key in {"coverages", "exclusions", "conditions"} else ""
            story.append(Paragraph(escape_xml(prefix + para), styles["body"]))
            # Extra filler for page depth on longer LOBs
            if key == "coverages" and i == 1 and bp.lob in {
                "commercial",
                "home",
                "workers_comp",
                "cyber",
                "health",
            }:
                filler = (
                    "Additional Clarification: Nothing in this coverage part shall be construed "
                    "to broaden coverage beyond the terms expressly stated. Any reference to "
                    "industry circulars, bureau forms, or prior editions is for identification only. "
                    f"This clarification is issued as of {bp.effective}."
                )
                if bp.quality_tier != "good":
                    filler = apply_messy_text(filler, rng)
                story.append(Paragraph(escape_xml(filler), styles["body"]))

    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            escape_xml(
                f"End of policy booklet for {bp.policy_number}. "
                f"Copyright {bp.effective[:4]} {bp.carrier}. All rights reserved. "
                "This is a specimen synthetic document for systems testing."
            ),
            styles["footer"],
        )
    )

    def _on_page(canvas: Any, doc_obj: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(
            0.75 * inch,
            0.4 * inch,
            f"{bp.policy_number} | {bp.carrier}",
        )
        canvas.drawRightString(
            LETTER[0] - 0.75 * inch,
            0.4 * inch,
            f"Page {doc_obj.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    # Page count from the built document
    return int(doc.page)


def expand_lob_list(total: int) -> list[str]:
    """Build LOB list matching planned ratios; truncate/pad to exact total."""
    planned = sum(LOB_COUNTS.values())
    items: list[str] = []
    for lob, count in LOB_COUNTS.items():
        scaled = max(1, round(count * total / planned)) if total != planned else count
        items.extend([lob] * scaled)
    # Adjust to exact total
    rng_adj = random.Random(0)
    while len(items) > total:
        items.pop(rng_adj.randrange(len(items)))
    while len(items) < total:
        items.append(rng_adj.choice(list(LOB_COUNTS.keys())))
    return items


def generate_corpus(
    out_dir: Path,
    count: int,
    seed: int,
    force: bool,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    lobs = expand_lob_list(count)
    rng.shuffle(lobs)

    existing = list(out_dir.glob("*.pdf"))
    if existing and not force:
        raise SystemExit(
            f"Found {len(existing)} existing PDFs in {out_dir}. "
            "Re-run with --force to overwrite, or choose another --out."
        )
    if force:
        for pdf in existing:
            pdf.unlink()
        for leftover in out_dir.glob("corpus_manifest.json"):
            leftover.unlink()

    manifest: list[dict[str, Any]] = []
    for i, lob in enumerate(lobs, start=1):
        # Per-doc RNG derived from master seed for stability if interrupted mid-run
        doc_rng = random.Random(seed + i * 9973)
        bp = build_blueprint(i, lob, doc_rng)
        path = out_dir / bp.filename
        pages = render_pdf(bp, path, doc_rng)
        entry = {
            **asdict(bp),
            "pages": pages,
            "path": str(path.name),
        }
        manifest.append(entry)
        if i % 50 == 0 or i == count:
            print(f"Generated {i}/{count} PDFs...", flush=True)

    manifest_path = out_dir / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "count": count,
                "generated": len(manifest),
                "quality_counts": _count_by(manifest, "quality_tier"),
                "lob_counts": _count_by(manifest, "lob"),
                "policies": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote manifest: {manifest_path}")
    return manifest


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        k = str(row.get(key, ""))
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))


def smoke_check(out_dir: Path, sample: int = 12, seed: int = 42) -> None:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit("pypdf is required for --smoke-check") from exc

    pdfs = sorted(out_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {out_dir}")
    rng = random.Random(seed)
    picks = pdfs if len(pdfs) <= sample else rng.sample(pdfs, sample)
    failures: list[str] = []
    for path in picks:
        reader = PdfReader(str(path))
        n_pages = len(reader.pages)
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        if n_pages < 2:
            failures.append(f"{path.name}: pages={n_pages} (expected >= 2)")
        if len(text.strip()) < 200:
            failures.append(f"{path.name}: extracted text too short ({len(text)} chars)")
    if failures:
        raise SystemExit("Smoke check failed:\n  - " + "\n  - ".join(failures))
    print(f"Smoke check OK: {len(picks)} PDFs, each >= 2 pages with extractable text.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate synthetic insurance policy PDFs for rag-knowledge-base."
    )
    parser.add_argument("--count", type=int, default=400, help="Number of PDFs (default 400)")
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "knowledge-base",
        help="Output directory (default: knowledge-base/)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing PDFs in the output directory",
    )
    parser.add_argument(
        "--smoke-check",
        action="store_true",
        help="After generation, verify a sample extracts via pypdf",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Only run smoke check on existing PDFs (no generation)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if args.smoke_only:
        smoke_check(args.out)
        return
    generate_corpus(args.out, args.count, args.seed, args.force)
    if args.smoke_check:
        smoke_check(args.out)


if __name__ == "__main__":
    main()
