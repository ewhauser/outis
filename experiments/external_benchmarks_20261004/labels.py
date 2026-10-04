"""Fixed label mappings, declared before any external system was scored.

Every benchmark and system has its own label scheme. Gold labels are mapped to
project entity types once, at conversion time. Predicted native labels are
mapped at scoring time. A native label mapped to ``None`` is still counted as
redacted text in the label-agnostic metrics, because the system would remove it,
but it is excluded from typed scoring.

Cross-system typed scoring then uses the coarse groups in ``COARSE``. They
merge distinctions that the benchmarks or the external label sets cannot make,
for example ASQ-PHI tags hospitals as GEOGRAPHIC_LOCATION and MedDeID has a
single ID family.
"""
from __future__ import annotations

import re


ASQ_PHI_GOLD = {
    "NAME": "NAME",
    "DATE": "DATE",
    "GEOGRAPHIC_LOCATION": "LOCATION",
    "MEDICAL_RECORD_NUMBER": "MRN",
    "HEALTH_PLAN_BENEFICIARY_NUMBER": "ID",
    "PHONE_NUMBER": "PHONE",
    "FAX_NUMBER": "PHONE",
    "SOCIAL_SECURITY_NUMBER": "SSN",
    "EMAIL_ADDRESS": "EMAIL",
    "UNIQUE_IDENTIFIER": "ID",
    "ACCOUNT_NUMBER": "ACCOUNT",
    "CERTIFICATE_LICENSE_NUMBER": "ID",
    "IP_ADDRESS": "IP",
}

# Project entity type -> coarse comparison group (also a project entity type).
COARSE = {
    "NAME": "NAME",
    "DATE": "DATE",
    "ADDRESS": "LOCATION",
    "LOCATION": "LOCATION",
    "FACILITY": "LOCATION",
    "MRN": "ID",
    "SSN": "ID",
    "ACCOUNT": "ID",
    "ID": "ID",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
    "URL": "URL",
    "IP": "IP",
    "AGE": "AGE",
}

# nvidia/gliner-PII prompt labels. TRAINED is the HIPAA-relevant subset of the
# Nemotron-PII training labels; it has no hospital or facility label. The
# facility arm adds one declared zero-shot label.
GLINER_TRAINED = {
    "first_name": "NAME",
    "last_name": "NAME",
    "date": "DATE",
    "date_of_birth": "DATE",
    "date_time": "DATE",
    "age": "AGE",
    "street_address": "ADDRESS",
    "city": "LOCATION",
    "county": "LOCATION",
    "state": "LOCATION",
    "postcode": "LOCATION",
    "coordinate": "LOCATION",
    "phone_number": "PHONE",
    "fax_number": "PHONE",
    "email": "EMAIL",
    "url": "URL",
    "ipv4": "IP",
    "ipv6": "IP",
    "ssn": "SSN",
    "medical_record_number": "MRN",
    "health_plan_beneficiary_number": "ID",
    "account_number": "ACCOUNT",
    "certificate_license_number": "ID",
    "vehicle_identifier": "ID",
    "license_plate": "ID",
    "unique_id": "ID",
    "customer_id": "ID",
    "employee_id": "ID",
    "biometric_identifier": "ID",
}
GLINER_FACILITY = {**GLINER_TRAINED, "hospital or medical facility": "FACILITY"}
GLINER_ARMS = {"trained": GLINER_TRAINED, "facility": GLINER_FACILITY}

MEDDEID = {
    "Name:Patient": "NAME",
    "Name:Caregiver": "NAME",
    "Name:Other": "NAME",
    "Date": "DATE",
    "Age_Birthdate": "AGE",
    "Address_Location:Patient": "ADDRESS",
    "Address_Location:Caregiver": "ADDRESS",
    "Address_Location:Other": "ADDRESS",
    "ID:Patient": "ID",
    "ID:Caregiver": "ID",
    "Organization:Healthcare": "FACILITY",
    "Organization:Other": "FACILITY",
    "Contactdetails": "CONTACT",
    "Profession": None,
}

_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")
_URL = re.compile(r"(?i)^(?:https?://|www\.)")
_IP = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def contact_type(text: str) -> str:
    """Split a generic contact-details span into a project entity type."""
    value = text.strip()
    if _EMAIL.search(value):
        return "EMAIL"
    if _URL.search(value):
        return "URL"
    if _IP.match(value):
        return "IP"
    return "PHONE"


def project_label(system: str, native: str, text: str) -> str | None:
    """Map a system's native label to a project entity type, or None."""
    if system in ("project", "project-model"):
        return native
    if system.startswith("gliner-"):
        mapping = GLINER_ARMS[system.removeprefix("gliner-")]
    elif system == "meddeid":
        mapping = MEDDEID
    else:
        raise KeyError(f"no label mapping declared for system {system!r}")
    if native not in mapping:
        raise KeyError(f"{system} produced undeclared label {native!r}")
    label = mapping[native]
    return contact_type(text) if label == "CONTACT" else label
