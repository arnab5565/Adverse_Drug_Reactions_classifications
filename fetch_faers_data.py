"""
=============================================================================
FAERS Data Fetching & Processing Module
Retrieves real FDA adverse drug reaction data from OpenFDA API
=============================================================================
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import time
import json

# OpenFDA API Configuration
OPENFDA_BASE_URL = "https://api.fda.gov/drug/event.json"
BATCH_SIZE = 100  # Records per request
MAX_REQUESTS = 5  # Limit to control data size

class FAERSDataFetcher:
    """Fetch and process real FAERS data from FDA OpenAPI"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the FAERS data fetcher
        
        Args:
            api_key: Optional FDA API key (not required for basic access)
        """
        self.api_key = api_key
        self.base_url = OPENFDA_BASE_URL
        self.params = {"limit": BATCH_SIZE}
        if api_key:
            self.params["api_key"] = api_key
    
    def fetch_adverse_events(self, limit: int = 500) -> List[Dict]:
        """
        Fetch adverse event records from OpenFDA API
        
        Args:
            limit: Total number of records to fetch
            
        Returns:
            List of adverse event records
        """
        records = []
        skip = 0
        max_requests = (limit // BATCH_SIZE) + 1
        
        print(f"\n[*] Fetching real FAERS data from OpenFDA API...")
        print(f"    Target: {limit} records\n")
        
        for request_num in range(min(max_requests, MAX_REQUESTS)):
            try:
                params = self.params.copy()
                params["skip"] = skip
                
                response = requests.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                batch = data.get("results", [])
                
                if not batch:
                    print(f"    No more records available. Stopping.")
                    break
                
                records.extend(batch)
                skip += BATCH_SIZE
                
                print(f"    ✓ Batch {request_num + 1}: {len(batch)} records (total: {len(records)})")
                
                if len(records) >= limit:
                    records = records[:limit]
                    break
                
                # Rate limiting to avoid overwhelming the API
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                print(f"    ✗ Error fetching data: {e}")
                if request_num > 0:
                    print(f"    Proceeding with {len(records)} records collected so far.")
                break
        
        return records
    
    @staticmethod
    def extract_severity(record: Dict) -> int:
        """
        Map FAERS seriousness flags to severity classes
        
        Args:
            record: FAERS adverse event record
            
        Returns:
            Severity class (0, 1, 2, or 3)
        """
        serious_data = record.get("serious", None)
        
        # Check seriousness flags in priority order
        if serious_data == "1":  # Mark as serious
            # Try multiple field structures as FAERS data can vary
            seriousness = record.get("seriousness", {})
            if isinstance(seriousness, list):
                seriousness = seriousness[0] if seriousness else {}
            
            # Check for death flags (multiple field variations)
            if (seriousness.get("seriousnessdeathstring") == "Yes" or 
                seriousness.get("seriousnessdeathstring") == "yes" or
                seriousness.get("seriousness_death") == "1" or
                record.get("seriousnessdeathstring") == "Yes"):
                return 3  # Death
            
            # Check for life-threatening flags
            if (seriousness.get("seriousnesslifethreateningstring") == "Yes" or 
                seriousness.get("seriousnesslifethreateningstring") == "yes" or
                seriousness.get("seriousness_life_threat") == "1" or
                record.get("seriousnesslifethreateningstring") == "Yes"):
                return 2  # Life-threatening
            
            # Check for disabling flags
            if (seriousness.get("seriousnessdisablingstring") == "Yes" or 
                seriousness.get("seriousnessdisablingstring") == "yes" or
                seriousness.get("seriousness_disabling") == "1" or
                record.get("seriousnessdisablingstring") == "Yes"):
                return 2  # Disabling (group with life-threatening)
            
            # Check for hospitalization flags
            if (seriousness.get("seriousnesshospitalizationstring") == "Yes" or 
                seriousness.get("seriousnesshospitalizationstring") == "yes" or
                seriousness.get("seriousness_hospitalization") == "1" or
                record.get("seriousnesshospitalizationstring") == "Yes"):
                return 1  # Hospitalization
            
            # If serious flag is set but no specific seriousness, assume hospitalization
            return 1
        
        # If not explicitly serious, assign based on reaction severity
        # Use number of reactions as proxy for severity
        reactions = record.get("patient", {}).get("reaction", [])
        if len(reactions) >= 3:
            return 1  # Multiple serious reactions
        
        return 0  # No serious reaction
    
    @staticmethod
    def extract_age_group(patient_data: Dict) -> str:
        """
        Extract and normalize patient age group
        
        Args:
            patient_data: Patient information from FAERS record
            
        Returns:
            Age group string (0-20, 21-40, 41-60, 61-80, 80+, or Unknown)
        """
        try:
            age_in_years = patient_data.get("patientonsetage")
            age_unit = patient_data.get("patientonsetageunit", "800")  # 800 = years
            
            if age_in_years is None:
                return "Unknown"
            
            # Convert to years if needed
            if age_unit == "801":  # months
                age_in_years = float(age_in_years) / 12
            elif age_unit == "802":  # weeks
                age_in_years = float(age_in_years) / 52
            elif age_unit == "803":  # days
                age_in_years = float(age_in_years) / 365
            else:
                age_in_years = float(age_in_years)
            
            # Bin into age groups
            if age_in_years < 21:
                return "0-20"
            elif age_in_years < 41:
                return "21-40"
            elif age_in_years < 61:
                return "41-60"
            elif age_in_years < 81:
                return "61-80"
            else:
                return "80+"
        except (ValueError, TypeError):
            return "Unknown"
    
    @staticmethod
    def extract_gender(patient_data: Dict) -> str:
        """
        Extract patient gender
        
        Args:
            patient_data: Patient information from FAERS record
            
        Returns:
            Gender string (Male, Female, or Unknown)
        """
        gender_code = patient_data.get("patientsex", None)
        
        # FAERS gender codes: 1 = Male, 2 = Female, null/other = Unknown
        if gender_code == "1":
            return "Male"
        elif gender_code == "2":
            return "Female"
        else:
            return "Unknown"
    
    @staticmethod
    def extract_drug_info(product_list: List[Dict]) -> Tuple[str, str]:
        """
        Extract drug class and dosage form from product information
        
        Args:
            product_list: List of drug products in the report
            
        Returns:
            Tuple of (drug_class, dosage_form)
        """
        if not product_list:
            return "Unknown", "Unknown"
        
        # Use first drug in the list
        product = product_list[0] if isinstance(product_list, list) else product_list
        
        # Map openfda drug class
        openfda = product.get("openfda", {})
        pharm_class = openfda.get("pharm_class", [])
        
        drug_class = "Other"
        if pharm_class:
            pharm = pharm_class[0].lower()
            if "antibiotic" in pharm or "antimicrob" in pharm:
                drug_class = "Antibiotic"
            elif "anticoagulant" in pharm or "antiplatelet" in pharm:
                drug_class = "Anticoagulant"
            elif "nsaid" in pharm or "nonsteroidal" in pharm or "inflammatory" in pharm:
                drug_class = "NSAID"
            elif "antihypertensive" in pharm or "beta blocker" in pharm or "ace inhibitor" in pharm:
                drug_class = "Antihypertensive"
            elif "statin" in pharm or "cholesterol" in pharm:
                drug_class = "Statin"
            elif "antidepressant" in pharm or "psychotropic" in pharm:
                drug_class = "Antidepressant"
        
        # Extract dosage form
        dosage_form = "Unknown"
        route = product.get("drugseparatedroute", "")
        if route:
            route_lower = route.lower()
            if "oral" in route_lower or "tablet" in route_lower:
                dosage_form = "Tablet"
            elif "intravenous" in route_lower or "iv" in route_lower:
                dosage_form = "Injection"
            elif "intramuscular" in route_lower or "im" in route_lower:
                dosage_form = "Injection"
            elif "capsule" in route_lower:
                dosage_form = "Capsule"
            elif "solution" in route_lower:
                dosage_form = "Solution"
            elif "patch" in route_lower or "transdermal" in route_lower:
                dosage_form = "Patch"
            elif "inhaler" in route_lower or "inhalation" in route_lower:
                dosage_form = "Inhaler"
        
        return drug_class, dosage_form
    
    def process_records(self, records: List[Dict], include_unknowns: bool = False) -> pd.DataFrame:
        """
        Process raw FAERS records into a structured DataFrame
        
        Args:
            records: List of FAERS adverse event records
            include_unknowns: Whether to include records with Unknown values
            
        Returns:
            Processed DataFrame with features ready for ML
        """
        processed = []
        skipped = 0
        
        for record in records:
            try:
                # Extract patient info
                patient = record.get("patient", {})
                if not patient:
                    skipped += 1
                    continue
                
                age_group = self.extract_age_group(patient)
                gender = self.extract_gender(patient)
                
                if not include_unknowns:
                    if age_group == "Unknown" or gender == "Unknown":
                        skipped += 1
                        continue
                
                # Extract drug and dosage info
                products = record.get("patient", {}).get("drug", [])
                if not products:
                    skipped += 1
                    continue
                
                drug_class, dosage_form = self.extract_drug_info(products)
                
                # Count concurrent drugs (polypharmacy)
                num_drugs = len(products)
                
                # Extract severity
                severity = self.extract_severity(record)
                
                # Count countries (as proxy for reporting source)
                country = record.get("occurcountry", "US")[:2].upper()  # Get country code
                
                # Check for prior ADR
                patient_history = patient.get("patientmedicalhistory", [])
                prior_adr = 1 if patient_history else 0
                
                # Renal and hepatic impairment (simplified - would need reaction parsing)
                reactions = patient.get("reaction", [])
                renal_impairment = 0
                hepatic_impairment = 0
                
                for reaction in reactions:
                    reaction_text = reaction.get("reactionmeddrapt", "").lower()
                    if "renal" in reaction_text or "kidney" in reaction_text:
                        renal_impairment = 1
                    if "hepatic" in reaction_text or "liver" in reaction_text:
                        hepatic_impairment = 1
                
                processed.append({
                    "age_group": age_group,
                    "gender": gender,
                    "dosage_form": dosage_form,
                    "drug_class": drug_class,
                    "country": country,
                    "num_drugs": min(num_drugs, 5),  # Cap at 5 for consistency
                    "prior_adr": prior_adr,
                    "renal_impairment": renal_impairment,
                    "hepatic_impairment": hepatic_impairment,
                    "severity": severity
                })
                
            except Exception as e:
                skipped += 1
                continue
        
        df = pd.DataFrame(processed)
        print(f"    Processed: {len(df)} records | Skipped: {skipped}")
        
        return df


def fetch_real_faers_data(num_records: int = 500) -> pd.DataFrame:
    """
    Main entry point to fetch and process real FAERS data
    
    Args:
        num_records: Number of records to fetch
        
    Returns:
        Processed DataFrame ready for ML
    """
    fetcher = FAERSDataFetcher()
    
    # Fetch raw data
    records = fetcher.fetch_adverse_events(limit=num_records)
    
    if not records:
        print("    ⚠ No data fetched. Using synthetic data instead.")
        return None
    
    # Process into structured format
    print("\n[*] Processing records into features...")
    df = fetcher.process_records(records, include_unknowns=False)
    
    if df.empty:
        print("    ⚠ No valid records after processing. Using synthetic data instead.")
        return None
    
    print(f"    ✓ Successfully processed {len(df)} records\n")
    return df


if __name__ == "__main__":
    # Test the module
    df = fetch_real_faers_data(num_records=100)
    if df is not None:
        print("\nSample of real FAERS data:")
        print(df.head(10))
        print(f"\nDataset shape: {df.shape}")
        print(f"\nSeverity distribution:")
        print(df["severity"].value_counts().sort_index())
