import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

class DataProcessor:
    """Class to handle data cleaning and preprocessing for transaction data"""
    
    def __init__(self):
        self.required_columns = [
            'Date', 'Mode', 'Category', 'Subcategory', 
            'Note', 'Amount', 'Income/Expense', 'Currency'
        ]
    
    def validate_columns(self, df):
        """Validate that the dataframe has all required columns"""
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return True
    
    def clean_data(self, df):
        """Clean and preprocess the transaction data"""
        try:
            # Validate columns
            self.validate_columns(df)
            
            # Create a copy to avoid modifying original data
            df_clean = df.copy()
            
            # Clean Date column
            df_clean['Date'] = self.clean_dates(df_clean['Date'])
            
            # Clean Amount column
            df_clean['Amount'] = self.clean_amounts(df_clean['Amount'])
            
            # Clean categorical columns
            df_clean = self.clean_categorical_columns(df_clean)
            
            # Remove rows with invalid data
            df_clean = self.remove_invalid_rows(df_clean)
            
            # Add derived columns
            df_clean = self.add_derived_columns(df_clean)
            
            return df_clean
            
        except Exception as e:
            st.error(f"Error in data cleaning: {str(e)}")
            raise e
    
    def clean_dates(self, date_series):
        """Clean and standardize date formats"""
        cleaned_dates = []
        
        for date_val in date_series:
            if pd.isna(date_val):
                cleaned_dates.append(pd.NaT)
                continue
            
            try:
                # Try different date formats
                date_str = str(date_val).strip()
                
                # Common formats to try
                formats = [
                    '%d/%m/%Y %H:%M:%S',  # 20/09/2018 12:04:08
                    '%d/%m/%Y',           # 20/09/2018
                    '%d-%m-%Y %H:%M:%S',  # 20-09-2018 12:04:08
                    '%d-%m-%Y',           # 20-09-2018
                    '%Y-%m-%d %H:%M:%S',  # 2018-09-20 12:04:08
                    '%Y-%m-%d',           # 2018-09-20
                    '%m/%d/%Y %H:%M:%S',  # 09/20/2018 12:04:08
                    '%m/%d/%Y',           # 09/20/2018
                ]
                
                parsed_date = None
                for fmt in formats:
                    try:
                        parsed_date = pd.to_datetime(date_str, format=fmt)
                        break
                    except ValueError:
                        continue
                
                if parsed_date is None:
                    # Try pandas auto-parsing as last resort
                    parsed_date = pd.to_datetime(date_str, errors='coerce')
                
                cleaned_dates.append(parsed_date)
                
            except Exception:
                cleaned_dates.append(pd.NaT)
        
        return pd.Series(cleaned_dates)
    
    def clean_amounts(self, amount_series):
        """Clean and standardize amount values"""
        cleaned_amounts = []
        
        for amount in amount_series:
            if pd.isna(amount):
                cleaned_amounts.append(0.0)
                continue
            
            try:
                # Handle string amounts with currency symbols or commas
                if isinstance(amount, str):
                    # Remove currency symbols and commas
                    amount_str = amount.replace('₹', '').replace(',', '').replace('INR', '').strip()
                    amount_val = float(amount_str)
                else:
                    amount_val = float(amount)
                
                # Ensure positive amounts
                amount_val = abs(amount_val)
                cleaned_amounts.append(amount_val)
                
            except (ValueError, TypeError):
                cleaned_amounts.append(0.0)
        
        return pd.Series(cleaned_amounts)
    
    def clean_categorical_columns(self, df):
        """Clean categorical columns"""
        categorical_columns = ['Mode', 'Category', 'Subcategory', 'Note', 'Income/Expense', 'Currency']
        
        for col in categorical_columns:
            if col in df.columns:
                # Fill missing values with 'Unknown' or appropriate default
                if col == 'Income/Expense':
                    df[col] = df[col].fillna('Expense')
                elif col == 'Currency':
                    df[col] = df[col].fillna('INR')
                else:
                    df[col] = df[col].fillna('Unknown')
                
                # Strip whitespace and standardize
                df[col] = df[col].astype(str).str.strip()
                
                # Standardize Income/Expense values
                if col == 'Income/Expense':
                    df[col] = df[col].str.title()
                    # Handle variations
                    df[col] = df[col].replace({
                        'Transfer-Out': 'Expense',
                        'Transfer-In': 'Income',
                        'Expense': 'Expense',
                        'Income': 'Income'
                    })
        
        return df
    
    def remove_invalid_rows(self, df):
        """Remove rows with invalid or missing critical data"""
        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        # Remove rows with zero or negative amounts (after cleaning)
        df = df[df['Amount'] > 0]
        
        # Remove rows with missing critical information
        df = df.dropna(subset=['Category', 'Income/Expense'])
        
        return df.reset_index(drop=True)
    
    def add_derived_columns(self, df):
        """Add derived columns for analysis"""
        # Extract date components
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.day_name()
        df['Month_Year'] = df['Date'].dt.to_period('M')
        
        # Add amount by type (positive for income, negative for expense for net calculations)
        df['Signed_Amount'] = df.apply(
            lambda row: row['Amount'] if row['Income/Expense'] == 'Income' else -row['Amount'], 
            axis=1
        )
        
        return df
    
    def get_data_summary(self, df):
        """Generate a summary of the cleaned data"""
        summary = {
            'total_transactions': len(df),
            'date_range': {
                'start': df['Date'].min(),
                'end': df['Date'].max()
            },
            'total_income': df[df['Income/Expense'] == 'Income']['Amount'].sum(),
            'total_expenses': df[df['Income/Expense'] == 'Expense']['Amount'].sum(),
            'unique_categories': df['Category'].nunique(),
            'unique_modes': df['Mode'].nunique(),
            'currencies': df['Currency'].unique().tolist()
        }
        
        return summary
