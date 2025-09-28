# src/backend/utils/pdf_processor.py

import json
import logging
import os
import re
import uuid
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

import pdfplumber
import pandas as pd
import google.generativeai as genai
from pydantic import BaseModel, Field, create_model
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Integer, insert, Text
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException

logger = logging.getLogger(__name__)

@dataclass
class TableInfo:
    """Data class to hold table information."""
    name: str
    schema: Dict[str, str]
    description: str
    data: List[List[str]]
    column_count: int
    context: Optional[Dict[str, str]] = None

class TableSchema(BaseModel):
    """Pydantic model for table schema from Gemini."""
    table_name: str = Field(..., description="Name of the table")
    table_schema: Dict[str, str] = Field(..., description="Column name to type mapping")
    description: str = Field(..., description="Description of the table content")

class PDFProcessor:
    """Enhanced utility class for processing PDF files with Gemini-powered schema inference."""
    
    def __init__(self, database_url: str = None, gemini_api_key: str = None):
        try:
            # Add debug logging at the start
            print(f"DEBUG PDFProcessor: Initializing with database_url: {bool(database_url)}")
            print(f"DEBUG PDFProcessor: Initializing with gemini_api_key: {bool(gemini_api_key)}")
            
            # Database setup
            if database_url is None:
                from ..config import config
                database_url = config.database_url
                print(f"DEBUG PDFProcessor: Using config database_url")
                
            print(f"DEBUG PDFProcessor: Final database_url: {database_url[:50]}...")
            self.engine = create_engine(database_url)
            self.metadata = MetaData()
            
            # Gemini setup with detailed debugging
            if gemini_api_key is None:
                from ..config import config
                gemini_api_key = config.GEMINI_API_KEY
                print(f"DEBUG PDFProcessor: Using config GEMINI_API_KEY")
                
            print(f"DEBUG PDFProcessor: gemini_api_key type: {type(gemini_api_key)}")
            print(f"DEBUG PDFProcessor: gemini_api_key length: {len(gemini_api_key) if gemini_api_key else 0}")
            print(f"DEBUG PDFProcessor: gemini_api_key value: {gemini_api_key[:10]}..." if gemini_api_key else "None")
            
            # Check if the API key is valid before configuring
            if not gemini_api_key or not gemini_api_key.strip():
                raise ValueError("GEMINI_API_KEY is empty or None")
                
            print(f"DEBUG PDFProcessor: About to configure genai with API key...")
            genai.configure(api_key=gemini_api_key)
            print(f"DEBUG PDFProcessor: genai.configure() successful")
            
            print(f"DEBUG PDFProcessor: About to create GenerativeModel...")
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print(f"DEBUG PDFProcessor: GenerativeModel created successfully")
            
            # Schema storage
            self.schema_file = Path("src/backend/utils/table_schema.json")
            self.schemas = self._load_schemas()
            
            with self.engine.connect() as conn:
                logger.info("Successfully connected to PostgreSQL and Gemini")
                print("Successfully connected to PostgreSQL and Gemini")
                
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {str(e)}")
            print(f"Error: Database connection failed: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Database connection error: {str(e)}")
        except Exception as e:
            logger.error(f"Gemini configuration failed: {str(e)}")
            print(f"Error: Gemini configuration failed: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500, detail=f"Gemini configuration error: {str(e)}")

    def _sanitize_column_name(self, column_name: str) -> str:
        """
        Sanitize column names to follow MySQL identifier naming conventions.
        
        Rules:
        - Must start with letter or underscore
        - Can contain letters, digits, underscores
        - Max 64 characters
        - Reserved words are handled by quoting
        """
        import re
        
        if not column_name or not column_name.strip():
            return "unnamed_column"
        
        # Clean the name
        cleaned = column_name.strip()
        
        # Replace spaces and special characters with underscores
        cleaned = re.sub(r'[^\w]', '_', cleaned)
        
        # Remove consecutive underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        
        # Ensure it starts with letter or underscore
        if cleaned and not (cleaned[0].isalpha() or cleaned[0] == '_'):
            cleaned = 'col_' + cleaned
        
        # Remove trailing underscores
        cleaned = cleaned.rstrip('_')
        
        # Ensure it's not empty after cleaning
        if not cleaned:
            cleaned = "unnamed_column"
        
        # Truncate to 64 characters (MySQL limit)
        if len(cleaned) > 64:
            cleaned = cleaned[:60] + '_trunc'
        
        # Handle MySQL reserved words by adding suffix
        mysql_reserved = {
            'order', 'group', 'select', 'from', 'where', 'insert', 'update', 
            'delete', 'create', 'drop', 'alter', 'index', 'table', 'database',
            'key', 'primary', 'foreign', 'unique', 'null', 'not', 'and', 'or',
            'in', 'exists', 'between', 'like', 'is', 'as', 'on', 'join', 'inner',
            'outer', 'left', 'right', 'union', 'distinct', 'count', 'sum', 'avg',
            'min', 'max', 'having', 'case', 'when', 'then', 'else', 'end'
        }
        
        if cleaned.lower() in mysql_reserved:
            cleaned += '_col'
        
        return cleaned

    def _load_schemas(self) -> Dict:
        """Load existing table schemas from JSON file."""
        if self.schema_file.exists():
            try:
                with open(self.schema_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load schemas: {e}")
                return {}
        return {}

    def _save_schemas(self):
        """Save table schemas to JSON file."""
        try:
            with open(self.schema_file, 'w') as f:
                json.dump(self.schemas, f, indent=2)
            logger.info(f"Saved schemas to {self.schema_file}")
        except Exception as e:
            logger.error(f"Failed to save schemas: {e}")

    def _get_context_text(self, pdf_path: str, page_num: int, table_position: int) -> dict:
        """Extract 400 characters of text before and after the table for context."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_num - 1]
                text = page.extract_text() or ""
                
                # Simple heuristic: split text around table position
                # This is approximate since we don't have exact table positions
                mid_point = len(text) // 2
                
                # Extract 400 chars before and after the estimated table position
                before_start = max(0, mid_point - 400)
                before_text = text[before_start:mid_point].strip()
                
                after_end = min(len(text), mid_point + 400)
                after_text = text[mid_point:after_end].strip()
                
                return {
                    "before": before_text,
                    "after": after_text
                }
        except Exception as e:
            logger.warning(f"Failed to extract context text: {e}")
            return {"before": "", "after": ""}



    def _generate_detailed_description(self, table_info: TableInfo, stored_row_count: int) -> str:
        """Generate detailed table description after data is stored."""
        # Get context text if available
        context_before = table_info.context.get("before", "") if table_info.context else ""
        context_after = table_info.context.get("after", "") if table_info.context else ""
        
        # Prepare full table preview (first 3 rows + last 2 rows if more than 5 total)
        headers = list(table_info.schema.keys())
        data_rows = table_info.data[1:]  # Skip header
        
        preview_data = []
        if len(data_rows) <= 5:
            preview_data = data_rows
        else:
            preview_data = data_rows[:3] + ["..."] + data_rows[-2:]
        
        # Format table for display
        table_display = []
        table_display.append(headers)
        for row in preview_data:
            if row == "...":
                table_display.append(["..."] * len(headers))
            else:
                # Ensure row matches header length
                formatted_row = row + [""] * (len(headers) - len(row)) if len(row) < len(headers) else row[:len(headers)]
                table_display.append(formatted_row)
        
        table_preview = "\n".join(["\t".join([str(cell) for cell in row]) for row in table_display])
        
        # Create schema summary
        schema_summary = []
        for col_name, col_type in table_info.schema.items():
            type_desc = {
                'string': 'VARCHAR(255) - Text data',
                'text': 'TEXT - Long text content', 
                'integer': 'INT - Whole numbers',
                'float': 'FLOAT - Decimal numbers',
                'currency': 'FLOAT - Monetary values (parsed from currency symbols)',
                'percentage': 'FLOAT - Percentage values (stored as decimal: 0.25 for 25%)'
            }.get(col_type.lower(), f'{col_type.upper()} - Custom type')
            schema_summary.append(f"- {col_name}: {type_desc}")
        
        schema_text = "\n".join(schema_summary)
        
        # Add context section to prompt
        context_section = ""
        if context_before or context_after:
            context_section = f"""
        SURROUNDING CONTEXT:
        Text before table: {context_before[:200]}{'...' if len(context_before) > 200 else ''}
        Text after table: {context_after[:200]}{'...' if len(context_after) > 200 else ''}
        """

        prompt = f"""
        Generate a comprehensive table description for database query generation. This description will help an LLM choose the correct table and generate accurate SQL queries.

        TABLE INFORMATION:
        Table Name: {table_info.name}
        Total Rows Stored: {stored_row_count}
        Column Count: {len(headers)}

        SCHEMA DETAILS:
        {schema_text}

        SAMPLE DATA:
        {table_preview}

        {context_section}

        Provide a clear, concise and simple description that would help an LLM understand when and how to use this table for query generation:
    """

        try:
            response = self.model.generate_content(prompt)
            detailed_description = response.text.strip()
            
            # Clean up any markdown formatting if present
            if "```" in detailed_description:
                detailed_description = detailed_description.replace("```", "").strip()
                
            return detailed_description
            
        except Exception as e:
            logger.error(f"Failed to generate detailed description: {e}")
            # Fallback to basic description
            return f"""Table: {table_info.name}
    Columns: {', '.join(headers)}
    Total Rows: {stored_row_count}
    Purpose: Data table with {len(headers)} columns containing structured information.
    Schema: {dict(table_info.schema)}"""



    def test_database_connection(self):
        """Test if database connection works"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute("SELECT version();")
                version = result.fetchone()[0]
                print(f"✓ Connected to PostgreSQL: {version}")
                logger.info(f"Database connection successful: {version}")
                return True
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            logger.error(f"Database connection failed: {e}")
            return False

    def _query_gemini_for_schema(self, table_data: List[List[str]], context_dict: dict, pdf_uuid: str, table_index: int = 1) -> TableSchema:
        """Query Gemini for table schema only (description will be generated later with full data)."""
        # Prepare the table preview (top 3 rows)
        preview_rows = table_data[:3]
        table_preview = "\n".join(["\t".join(row) for row in preview_rows])
        
        prompt = f"""
    Analyze this table data and provide schema information in JSON format.

    Table Preview (first 3 rows):
    {table_preview}

    Please provide a JSON response with:
    1. table_name: A descriptive name for this table (use format: pdf_{pdf_uuid}_descriptive_name)
    2. table_schema: Object mapping SANITIZED column names to SQL types (use: "string", "integer", "float", "text", "currency", "percentage")
    - Column names must follow MySQL rules: letters, digits, underscores only
    - Must start with letter or underscore
    - Replace spaces with underscores
    - Max 64 characters
    3. description: "TBD" (will be generated later with full data)

Schema type guidelines:
- "currency": For monetary values (e.g., $4.34, €10.50, ¥1000)
- "percentage": For percentage values (e.g., 25%, 0.15%)
- "float": For plain decimal numbers
- "integer": For whole numbers
- "string": For text data
- "text": For longer text content

Example response:
{{
    "table_name": "pdf_{pdf_uuid}_financial_summary",
    "table_schema": {{
        "year": "integer",
        "revenue": "currency",
        "profit_margin": "percentage",
        "description": "text"
    }},
    "description": "Financial summary table showing yearly revenue and profit margins"
}}

Respond with valid JSON only:
"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up the response to extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            schema_data = json.loads(response_text)

            # Sanitize column names in the schema
            if 'table_schema' in schema_data:
                original_schema = schema_data['table_schema']
                sanitized_schema = {}
                for col_name, col_type in original_schema.items():
                    sanitized_name = self._sanitize_column_name(col_name)
                    sanitized_schema[sanitized_name] = col_type
                schema_data['table_schema'] = sanitized_schema

            return TableSchema(**schema_data)
            
        except Exception as e:
            logger.error(f"Failed to query Gemini for schema: {e}")
            # Fallback to basic schema
            headers = table_data[0] if table_data else []
            fallback_schema = {
                self._sanitize_column_name(header): "string" 
                for header in headers
            }
            return TableSchema(
                table_name=f"pdf_{pdf_uuid}_table_{table_index}",
                table_schema=fallback_schema,
                description="Auto-generated table schema"
            )

    def _query_gemini_for_continuation(self, current_table_headers: List[str], new_table_preview: List[List[str]], current_table_data: List[List[str]] = None) -> bool:
        """Query Gemini to check if a table is a continuation of the previous one."""
        current_headers_str = "\t".join(current_table_headers)
        new_preview_str = "\n".join(["\t".join(row) for row in new_table_preview[:3]])
        
        # Include current table context with headers and top 3 data rows
        current_table_context = ""
        if current_table_data and len(current_table_data) > 1:
            # Format: headers + top 3 data rows (skip the header row which is at index 0)
            current_preview_rows = [current_table_headers] + current_table_data[1:4]  # headers + top 3 data rows
            current_table_context = "\n".join(["\t".join(row) for row in current_preview_rows])
        else:
            # Fallback to just headers if no data available
            current_table_context = current_headers_str
        
        prompt = f"""
    Determine if this new table data is a continuation of the previous table.

    Current table (headers + top 3 data rows):
    {current_table_context}

    New table preview (first 3 rows):
    {new_preview_str}

    Analyze if this is a continuation (same structure, no headers) or a new table.

    Respond with JSON only:
    - If it's a continuation: {{"status": true}}
    - If it's a new table: {{"status": false, "reason": "explain why it's not a continuation"}}

    Examples:
    - Same column count, data rows only: {{"status": true}}
    - Different column count: {{"status": false, "reason": "Column count mismatch"}}
    - Different data structure: {{"status": false, "reason": "Data structure differs from previous table"}}

    JSON response:
    """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up the response to extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            result = json.loads(response_text)
            
            # Log the reason if it's not a continuation
            if not result.get("status", False):
                reason = result.get("reason", "No reason provided")
                logger.info(f"Table not a continuation: {reason}")
                print(f"  → Not a continuation: {reason}")
            else:
                print(f"  → Confirmed continuation")
                
            return result.get("status", False)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response for continuation: {e}")
            print(f"  → JSON parsing failed, assuming new table")
            return False
        except Exception as e:
            logger.error(f"Failed to query Gemini for continuation: {e}")
            print(f"  → Query failed, assuming new table")
            return False

    def _parse_numeric_value(self, value: str, expected_type: str) -> Optional[float]:
        """
        Parse numeric values with units (currency, percentages, etc.) into clean numbers.
        
        Args:
            value: The string value to parse
            expected_type: The expected data type (currency, percentage, float, integer)
            
        Returns:
            Parsed numeric value or None if parsing fails
        """
        if not value or not value.strip():
            return None
            
        # Clean the value
        cleaned_value = value.strip()
        
        try:
            # Handle currency values
            if expected_type == "currency":
                # Remove currency symbols and common formatting
                import re
                # Common currency symbols: $, €, £, ¥, ₹, etc.
                currency_pattern = r'[\$€£¥₹₽₩¢₦₨₪₫₡₲₴₸₵₶₷₹₺₻₼₽₾₿]'
                cleaned_value = re.sub(currency_pattern, '', cleaned_value)
                # Remove commas used as thousands separators
                cleaned_value = cleaned_value.replace(',', '')
                # Remove spaces
                cleaned_value = cleaned_value.replace(' ', '')
                # Handle parentheses for negative values (accounting format)
                if cleaned_value.startswith('(') and cleaned_value.endswith(')'):
                    cleaned_value = '-' + cleaned_value[1:-1]
                return float(cleaned_value) if cleaned_value else None
                
            # Handle percentage values
            elif expected_type == "percentage":
                if '%' in cleaned_value:
                    cleaned_value = cleaned_value.replace('%', '').strip()
                    # Convert percentage to decimal (25% -> 0.25)
                    return float(cleaned_value) / 100 if cleaned_value else None
                else:
                    # Assume it's already in decimal format
                    return float(cleaned_value) if cleaned_value else None
                    
            # Handle regular numeric values with potential formatting
            elif expected_type in ["float", "integer"]:
                # Remove common formatting characters
                import re
                # Remove everything except digits, decimal points, minus signs, and 'e' for scientific notation
                cleaned_value = re.sub(r'[^\d\.\-e]', '', cleaned_value)
                
                if expected_type == "integer":
                    # For integers, convert to float first then to int to handle decimal formatting
                    float_val = float(cleaned_value) if cleaned_value else None
                    return int(float_val) if float_val is not None else None
                else:
                    return float(cleaned_value) if cleaned_value else None
                    
            # If not a numeric type, return None
            else:
                return None
                
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse '{value}' as {expected_type}: {e}")
            return None

    def _create_pydantic_model(self, schema_info: TableSchema) -> type:
        """Create a Pydantic model from the schema information with custom validators."""
        from pydantic import BaseModel, Field, field_validator
        from typing import Optional, Any
        
        type_mapping = {
            "string": (str, Field(default="")),
            "integer": (Optional[int], Field(default=None)),
            "float": (Optional[float], Field(default=None)),
            "text": (str, Field(default="")),
            "currency": (Optional[float], Field(default=None, description="Monetary value parsed from currency format")),
            "percentage": (Optional[float], Field(default=None, description="Percentage value as decimal (0.25 for 25%)"))
        }
        
        # Create field annotations dictionary
        annotations = {}
        field_defaults = {}
        
        for col_name, col_type in schema_info.table_schema.items():
            python_type, field_info = type_mapping.get(col_type.lower(), (str, Field(default="")))
            annotations[col_name] = python_type
            field_defaults[col_name] = field_info
        
        # Create a base class with the static method
        class BaseTableModel(BaseModel):
            @staticmethod
            def _parse_numeric_value(value: str, expected_type: str) -> Optional[float]:
                """Parse numeric values with units (currency, percentages, etc.) into clean numbers."""
                if not value or not value.strip():
                    return None
                    
                # Clean the value
                cleaned_value = value.strip()
                
                try:
                    # Handle currency values
                    if expected_type == "currency":
                        # Remove currency symbols and common formatting
                        import re
                        # Common currency symbols: $, €, £, ¥, ₹, etc.
                        currency_pattern = r'[\$€£¥₹₽₩¢₦₨₪₫₡₲₴₸₵₶₷₹₺₻₼₽₾₿]'
                        cleaned_value = re.sub(currency_pattern, '', cleaned_value)
                        # Remove commas used as thousands separators
                        cleaned_value = cleaned_value.replace(',', '')
                        # Remove spaces
                        cleaned_value = cleaned_value.replace(' ', '')
                        # Handle parentheses for negative values (accounting format)
                        if cleaned_value.startswith('(') and cleaned_value.endswith(')'):
                            cleaned_value = '-' + cleaned_value[1:-1]
                        return float(cleaned_value) if cleaned_value else None
                        
                    # Handle percentage values
                    elif expected_type == "percentage":
                        if '%' in cleaned_value:
                            cleaned_value = cleaned_value.replace('%', '').strip()
                            # Convert percentage to decimal (25% -> 0.25)
                            return float(cleaned_value) / 100 if cleaned_value else None
                        else:
                            # Assume it's already in decimal format
                            return float(cleaned_value) if cleaned_value else None
                            
                    # Handle regular numeric values with potential formatting
                    elif expected_type in ["float", "integer"]:
                        # Remove common formatting characters
                        import re
                        # Remove everything except digits, decimal points, minus signs, and 'e' for scientific notation
                        cleaned_value = re.sub(r'[^\d\.\-e]', '', cleaned_value)
                        
                        if expected_type == "integer":
                            # For integers, convert to float first then to int to handle decimal formatting
                            float_val = float(cleaned_value) if cleaned_value else None
                            return int(float_val) if float_val is not None else None
                        else:
                            return float(cleaned_value) if cleaned_value else None
                            
                    # If not a numeric type, return None
                    else:
                        return None
                        
                except (ValueError, TypeError):
                    return None
        
        # Create validators dictionary for numeric fields
        validators = {}
        
        for col_name, col_type in schema_info.table_schema.items():
            if col_type.lower() in ["currency", "percentage", "float", "integer"]:
                # Create a closure to capture the column type
                def make_validator(column_type: str):
                    @field_validator(col_name, mode='before')
                    @classmethod
                    def validate_field(cls, v: Any) -> Any:
                        if v is None or v == "":
                            return None
                        if isinstance(v, (int, float)):
                            return v
                        if isinstance(v, str):
                            parsed = cls._parse_numeric_value(v, column_type)
                            if parsed is not None:
                                return parsed
                            # If parsing fails, try basic float conversion
                            try:
                                return float(v)
                            except (ValueError, TypeError):
                                return None
                        return v
                    return validate_field
                
                validator_func = make_validator(col_type.lower())
                # Use a unique name for each validator
                validators[f'validate_{col_name.replace(" ", "_").replace("-", "_")}'] = validator_func
        
        # Create the model class dynamically
        model_attrs = {
            '__annotations__': annotations,
            **field_defaults,
            **validators
        }
        
        # Create the final model class
        DynamicModel = type(
            f"{schema_info.table_name}Model",
            (BaseTableModel,),
            model_attrs
        )
        
        return DynamicModel

    def _convert_schema_to_sqlalchemy(self, schema_info: TableSchema) -> List[Column]:
        """Convert Gemini schema to SQLAlchemy columns."""
        type_mapping = {
            "string": String(255),
            "integer": Integer,
            "float": Float,
            "text": Text,
            "currency": Float,  # Store currency as float (numeric value only)
            "percentage": Float  # Store percentage as float (decimal format)
        }
        
        columns = []
        for col_name, col_type in schema_info.table_schema.items():
            sqlalchemy_type = type_mapping.get(col_type.lower(), String(255))
            columns.append(Column(col_name, sqlalchemy_type))
        
        return columns


#renamed the old extract and store content method to _legacy_extract_and_store_content
    def _legacy_extract_and_store_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        DEPRECATED: Legacy PDF processing method.
        Use optimized_extract_and_store() instead for better text chunking.
        
        Enhanced content extraction with Gemini-powered schema inference.
        Combines extraction and storage into a single intelligent process.
        """
        text_chunks = []
        stored_tables = []
        current_table_info: Optional[TableInfo] = None


        # Generate UUID for unique table naming
        pdf_uuid = str(uuid.uuid4())[:8] 

        logger.info(f"Starting enhanced PDF extraction for file: {pdf_path}")
        print(f"\n=== Enhanced PDF Processing ===")
        print(f"File: {Path(pdf_path).name}")
        print(f"File UUID: {pdf_uuid}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text for chunks
                    text = page.extract_text()
                    if text:
                        sentences = re.split(r'(?<=[.!?])\s+', text)
                        chunk = ""
                        for sentence in sentences:
                            if len(chunk) + len(sentence) < 400:
                                chunk += sentence + " "
                            else:
                                if chunk.strip():
                                    text_chunks.append(chunk.strip())
                                chunk = sentence + " "
                        if chunk.strip():
                            text_chunks.append(chunk.strip())

                    # Extract and process tables
                    page_tables = page.extract_tables()
                    logger.info(f"Found {len(page_tables)} tables on page {page_num}")
                    
                    for table_idx, table in enumerate(page_tables, 1):
                        if not table or not table[0]:
                            continue

                        cleaned_table = [
                            [str(cell) if cell is not None else "" for cell in row]
                            for row in table if any(cell.strip() for cell in row if cell is not None)
                        ]

                        if not cleaned_table:
                            continue

                        # Check for transposition
                        if len(cleaned_table) < len(cleaned_table[0]):
                            cleaned_table = list(map(list, zip(*cleaned_table)))

                        print(f"\nProcessing table {table_idx} on page {page_num}")
                        print(f"Table dimensions: {len(cleaned_table)} rows x {len(cleaned_table[0])} columns")

                        # Check if this continues the previous table
                        if (current_table_info and 
                            len(cleaned_table[0]) == current_table_info.column_count):
                            
                            print("Checking if table continues previous one...")
                            is_continuation = self._query_gemini_for_continuation(
                                list(current_table_info.schema.keys()),
                                cleaned_table
                            )
                            
                            if is_continuation:
                                print("✓ Continuing previous table")
                                current_table_info.data.extend(cleaned_table)
                                continue

                        # Finalize previous table if exists
                        if current_table_info:
                            print(f"Finalizing table: {current_table_info.name}")
                            success = self._store_table_with_schema(current_table_info)
                            if success:
                                # Get updated description from schemas
                                updated_schema = self.schemas.get(current_table_info.name, {})
                                stored_tables.append({
                                    "name": current_table_info.name,
                                    "rows": len(current_table_info.data) - 1,  # Exclude header
                                    "description": updated_schema.get('description', current_table_info.description)
                                })

                        # Process new table with Gemini
                        print("Analyzing new table with Gemini...")
                        context_dict = self._get_context_text(pdf_path, page_num, table_idx)
                        # Generate unique table index across all pages
                        global_table_index = len(stored_tables) + 1
                        schema_info = self._query_gemini_for_schema(cleaned_table, context_dict, pdf_uuid, global_table_index)
                        
                        print(f"✓ Gemini analysis complete:")
                        print(f"  Table name: {schema_info.table_name}")
                        print(f"  Schema: {schema_info.table_schema}")
                        print(f"  Description: {schema_info.description}")

                        # Save initial schema to file (description will be updated after storage)
                        self.schemas[schema_info.table_name] = {
                            "schema": schema_info.table_schema,
                            "description": schema_info.description,
                            "pdf_uuid": pdf_uuid,
                            "created_at": pd.Timestamp.now().isoformat(),
                            "status": "processing"
                        }
                        self._save_schemas()
                        print(f"✓ Saved initial schema for {schema_info.table_name}")

                        # Create new table info
                        current_table_info = TableInfo(
                            name=schema_info.table_name,
                            schema=schema_info.table_schema,
                            description=schema_info.description,
                            data=cleaned_table,
                            column_count=len(cleaned_table[0])
                        )
                        # Store context for later use in description generation
                        current_table_info.context = context_dict

                # Finalize the last table
                if current_table_info:
                    print(f"Finalizing last table: {current_table_info.name}")
                    success = self._store_table_with_schema(current_table_info)
                    if success:
                        # Get updated description from schemas
                        updated_schema = self.schemas.get(current_table_info.name, {})
                        stored_tables.append({
                            "name": current_table_info.name,
                            "rows": len(current_table_info.data) - 1,
                            "description": updated_schema.get('description', current_table_info.description)
                        })

            print(f"\n=== Processing Complete ===")
            print(f"Text chunks extracted: {len(text_chunks)}")
            print(f"Tables stored: {len(stored_tables)}")
            for table in stored_tables:
                print(f"  - {table['name']}: {table['rows']}")
            print("===============================\n")

            return {
                "text_chunks": text_chunks,
                "tables_info": stored_tables,
                "schemas_saved": len(stored_tables),
                "pdf_name": Path(pdf_path).stem,
                "pdf_uuid": pdf_uuid
            }

        except Exception as e:
            logger.error(f"Enhanced PDF extraction failed: {str(e)}")
            print(f"Error: Enhanced PDF extraction failed: {str(e)}")
            raise ValueError(f"Enhanced PDF extraction error: {str(e)}")
    
    def optimized_extract_and_store(self, pdf_path: str) -> Dict[str, Any]:
       
        
        import re
        import uuid
        from pathlib import Path
        from collections import Counter
        import math
        import pandas as pd
        import pdfplumber

        # ----------------------------
        # Tunables (adjust if needed)
        # ----------------------------
        TARGET_WORDS_MIN = 350         # ~750 tokens (rough approx 1 token ~ 1.3 words in English prose)
        TARGET_WORDS_MAX = 650         # ~1200 tokens
        HARD_MAX_WORDS   = 800         # emergency upper bound to avoid huge chunks
        OVERLAP_SENTENCES = 2          # overlap for context preservation between chunks
        SEM_SIM_THRESHOLD = 0.08       # semantic similarity threshold for grouping (0..1, cosine)
        BULLET_PREFIX_RE  = r"^(\s*[-–•*]|(\s*\d+[\).\]]|\s*[A-Z]\))\s+)"
        HEADING_RE = (
            r"^(\s*"
            r"([0-9]+\.){1,4}\s+[^\n]{2,}|"
            r"[IVXLCM]+\.\s+[^\n]{2,}|"
            r"[A-Z][A-Za-z0-9/&\-\s]{0,60}$|"
            r"[A-Z0-9][A-Z0-9\s/&\-]{3,60}$"
            r")"
        )

        STOPWORDS = {
            "the","a","an","and","or","but","if","then","else","when","while","at","by","for",
            "with","about","against","between","into","through","during","before","after",
            "above","below","to","from","up","down","in","out","on","off","over","under",
            "again","further","once","here","there","all","any","both","each","few","more",
            "most","other","some","such","no","nor","not","only","own","same","so","than",
            "too","very","can","will","just","don","should","now"
        }

        def sent_tokenize(text: str) -> list[str]:
            # Conservative sentence splitter; keeps abbreviations fairly safe.
            # Splits at . ! ? followed by space/newline and uppercase start or digit/quote.
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text.strip())
            # Also split overly long parts by semicolons/newlines to avoid giant sentences
            out = []
            for p in parts:
                if len(p) > 1200:
                    out.extend(re.split(r'[;\n]{1,}', p))
                else:
                    out.append(p)
            return [s.strip() for s in out if s.strip()]

        def is_heading(line: str) -> bool:
            s = line.strip()
            if len(s) > 100:  # very long lines are unlikely to be headings
                return False
            return bool(re.match(HEADING_RE, s))

        def is_bullet(line: str) -> bool:
            return bool(re.match(BULLET_PREFIX_RE, line))

        def normalize_words(text: str) -> list[str]:
            words = re.findall(r"[A-Za-z0-9']+", text.lower())
            return [w for w in words if w not in STOPWORDS]

        def bow_vector(text: str) -> Counter:
            return Counter(normalize_words(text))

        def cosine_sim(c1: Counter, c2: Counter) -> float:
            if not c1 or not c2:
                return 0.0
            # dot product
            inter = set(c1.keys()) & set(c2.keys())
            dot = sum(c1[k] * c2[k] for k in inter)
            # magnitude
            mag1 = math.sqrt(sum(v*v for v in c1.values()))
            mag2 = math.sqrt(sum(v*v for v in c2.values()))
            if mag1 == 0 or mag2 == 0:
                return 0.0
            return dot / (mag1 * mag2)

        def sectionize(raw_text: str) -> list[dict]:
            """
            Split page text into sections by headings / blank lines / bullets,
            but keep short 'header lines' attached to their following paragraph.
            """
            lines = [ln.rstrip() for ln in raw_text.splitlines()]
            sections = []
            buf = []
            curr_heading = None

            i = 0
            while i < len(lines):
                ln = lines[i]
                if is_heading(ln):
                    # flush previous section
                    if buf:
                        sections.append({"heading": curr_heading, "text": "\n".join(buf).strip()})
                        buf = []
                    curr_heading = ln.strip()
                elif ln.strip() == "":
                    # paragraph boundary
                    if buf:
                        sections.append({"heading": curr_heading, "text": "\n".join(buf).strip()})
                        buf = []
                else:
                    # keep bullets tightly packed
                    if is_bullet(ln):
                        # ensure previous non-bullet paragraph is flushed
                        if buf and not is_bullet(buf[-1]):
                            sections.append({"heading": curr_heading, "text": "\n".join(buf).strip()})
                            buf = []
                    buf.append(ln)
                i += 1

            if buf:
                sections.append({"heading": curr_heading, "text": "\n".join(buf).strip()})
            # Filter empties
            return [s for s in sections if s["text"]]

        def build_chunks_from_sections(sections: list[dict], page_num: int) -> list[str]:
            """
            For each section: sentence tokenize, then semantically group sentences
            until we reach target size. Add overlaps between consecutive chunks.
            """
            chunks: list[str] = []

            for sec in sections:
                heading = sec["heading"]
                sentences = sent_tokenize(sec["text"])
                if not sentences:
                    continue

                # Precompute sentence vectors (cheap)
                sent_vecs = [bow_vector(s) for s in sentences]

                curr: list[str] = []
                curr_vec = Counter()
                word_count = 0

                def flush(with_overlap=True):
                    nonlocal curr, curr_vec, word_count
                    if not curr:
                        return
                    # Prefix with heading + (Page X) for light context anchoring
                    prefix_bits = []
                    if heading:
                        prefix_bits.append(f"{heading.strip()}")
                    prefix_bits.append(f"(Page {page_num})")
                    prefix = " — ".join(prefix_bits)

                    chunk_text = prefix + "\n" + " ".join(curr)
                    chunks.append(chunk_text.strip())

                    if with_overlap and OVERLAP_SENTENCES > 0:
                        # keep last N sentences as seed for next chunk
                        overlap = curr[-OVERLAP_SENTENCES:]
                        curr = overlap[:]
                        curr_vec = Counter()
                        for s in curr:
                            curr_vec.update(bow_vector(s))
                        word_count = sum(len(normalize_words(s)) for s in curr)
                    else:
                        curr = []
                        curr_vec = Counter()
                        word_count = 0

                for idx, (s, sv) in enumerate(zip(sentences, sent_vecs)):
                    s_words = len(normalize_words(s))
                    # If adding this sentence explodes past HARD_MAX_WORDS, flush first.
                    if word_count + s_words > HARD_MAX_WORDS and curr:
                        flush(with_overlap=True)

                    # Similarity to current context
                    sim = cosine_sim(curr_vec, sv) if curr else 1.0  # first sentence: force add

                    # Heuristic: if current is already healthy size and similarity is low, start new chunk
                    if curr and word_count >= TARGET_WORDS_MIN and sim < SEM_SIM_THRESHOLD:
                        flush(with_overlap=True)

                    # Add sentence
                    curr.append(s)
                    curr_vec.update(sv)
                    word_count += s_words

                    # Adaptive flush when exceeding soft max
                    if word_count >= TARGET_WORDS_MAX:
                        # If next sentence exists and is semantically different, flush now;
                        # otherwise allow slight overflow to keep a coherent idea together.
                        if idx + 1 < len(sentences):
                            next_sim = cosine_sim(curr_vec, sent_vecs[idx + 1])
                            if next_sim < SEM_SIM_THRESHOLD:
                                flush(with_overlap=True)

                # Flush remainder (without extra overlap to avoid duplication at section end)
                flush(with_overlap=False)

            return chunks

        # ----------------------------------------------------------------------
        # Main processing (mirrors your original, with improved text chunking)
        # ----------------------------------------------------------------------
        text_chunks: list[str] = []
        stored_tables = []
        current_table_info = None

        pdf_uuid = str(uuid.uuid4())[:8]
        logger.info(f"[Optimized] Starting PDF extraction for file: {pdf_path}")
        print(f"\n=== Optimized PDF Processing ===")
        print(f"File: {Path(pdf_path).name}")
        print(f"File UUID: {pdf_uuid}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # -------- Text (hybrid chunking) --------
                    text = page.extract_text() or ""
                    if text.strip():
                        sections = sectionize(text)
                        page_chunks = build_chunks_from_sections(sections, page_num)
                        text_chunks.extend(page_chunks)

                    # -------- Tables (same logic as your original) --------
                    page_tables = page.extract_tables() or []
                    logger.info(f"[Optimized] Found {len(page_tables)} tables on page {page_num}")

                    for table_idx, table in enumerate(page_tables, 1):
                        if not table or not table[0]:
                            continue

                        cleaned_table = [
                            [str(cell) if cell is not None else "" for cell in row]
                            for row in table if any((str(cell).strip() if cell is not None else "") for cell in row)
                        ]
                        if not cleaned_table:
                            continue

                        # transpose if looks sideways
                        if len(cleaned_table) < len(cleaned_table[0]):
                            cleaned_table = list(map(list, zip(*cleaned_table)))

                        print(f"\nProcessing table {table_idx} on page {page_num}")
                        print(f"Table dimensions: {len(cleaned_table)} rows x {len(cleaned_table[0])} columns")

                        # Possible continuation of previous table
                        if (current_table_info and len(cleaned_table[0]) == current_table_info.column_count):
                            print("Checking if table continues previous one...")
                            is_continuation = self._query_gemini_for_continuation(
                                list(current_table_info.schema.keys()),
                                cleaned_table
                            )
                            if is_continuation:
                                print("✓ Continuing previous table")
                                current_table_info.data.extend(cleaned_table)
                                continue

                        # Finalize previous table if exists
                        if current_table_info:
                            print(f"Finalizing table: {current_table_info.name}")
                            success = self._store_table_with_schema(current_table_info)
                            if success:
                                updated_schema = self.schemas.get(current_table_info.name, {})
                                stored_tables.append({
                                    "name": current_table_info.name,
                                    "rows": len(current_table_info.data) - 1,
                                    "description": updated_schema.get('description', current_table_info.description)
                                })

                        # New table -> Gemini schema inference
                        print("Analyzing new table with Gemini...")
                        context_dict = self._get_context_text(pdf_path, page_num, table_idx)
                        global_table_index = len(stored_tables) + 1
                        schema_info = self._query_gemini_for_schema(cleaned_table, context_dict, pdf_uuid, global_table_index)

                        print(f"✓ Gemini analysis complete:")
                        print(f"  Table name: {schema_info.table_name}")
                        print(f"  Schema: {schema_info.table_schema}")
                        print(f"  Description: {schema_info.description}")

                        # Save initial schema (status: processing)
                        self.schemas[schema_info.table_name] = {
                            "schema": schema_info.table_schema,
                            "description": schema_info.description,
                            "pdf_uuid": pdf_uuid,
                            "created_at": pd.Timestamp.now().isoformat(),
                            "status": "processing"
                        }
                        self._save_schemas()
                        print(f"✓ Saved initial schema for {schema_info.table_name}")

                        # Create/track current table
                        current_table_info = TableInfo(
                            name=schema_info.table_name,
                            schema=schema_info.table_schema,
                            description=schema_info.description,
                            data=cleaned_table,
                            column_count=len(cleaned_table[0])
                        )
                        current_table_info.context = context_dict

                # Finalize the last table
                if current_table_info:
                    print(f"Finalizing last table: {current_table_info.name}")
                    success = self._store_table_with_schema(current_table_info)
                    if success:
                        updated_schema = self.schemas.get(current_table_info.name, {})
                        stored_tables.append({
                            "name": current_table_info.name,
                            "rows": len(current_table_info.data) - 1,
                            "description": updated_schema.get('description', current_table_info.description)
                        })

            print(f"\n=== Optimized Processing Complete ===")
            print(f"Text chunks extracted: {len(text_chunks)}")
            print(f"Tables stored: {len(stored_tables)}")
            for table in stored_tables:
                print(f"  - {table['name']}: {table['rows']}")
            print("===============================\n")

            return {
                "text_chunks": text_chunks,
                "tables_info": stored_tables,
                "schemas_saved": len(stored_tables),
                "pdf_name": Path(pdf_path).stem,
                "pdf_uuid": pdf_uuid
            }

        except Exception as e:
            logger.error(f"Optimized PDF extraction failed: {str(e)}")
            print(f"Error: Optimized PDF extraction failed: {str(e)}")
            raise ValueError(f"Optimized PDF extraction error: {str(e)}")

    def _store_table_with_schema(self, table_info: TableInfo) -> bool:
        """Store table using Gemini-generated schema and Pydantic validation with enhanced numeric parsing."""
        try:
            print(f"\nStoring table: {table_info.name}")
            # Validate and sanitize column names in schema
            sanitized_schema = {}
            for col_name, col_type in table_info.schema.items():
                sanitized_name = self._sanitize_column_name(col_name)
                sanitized_schema[sanitized_name] = col_type
            
            # Update table_info with sanitized schema
            table_info.schema = sanitized_schema
            # Create Pydantic model for validation
            schema_info = TableSchema(
                table_name=table_info.name,
                table_schema=table_info.schema,
                description=table_info.description
            )
            pydantic_model = self._create_pydantic_model(schema_info)
            
            # Create SQLAlchemy table
            columns = self._convert_schema_to_sqlalchemy(schema_info)
            table = Table(table_info.name, self.metadata, *columns, schema='public')

            # Add debug logging
            print(f"DEBUG: About to create table: {table_info.name}")
            print(f"DEBUG: Columns: {[col.name for col in columns]}")
            self.metadata.create_all(self.engine)
            
            print(f"Created table with schema: {table_info.schema}")

            # Process and validate data with enhanced numeric parsing
            headers = list(table_info.schema.keys())
            data_rows = table_info.data[1:]  # Skip header row
            
            validated_rows = []
            parsing_stats = {"success": 0, "failed": 0, "warnings": []}
            
            for row_idx, row in enumerate(data_rows):
                try:
                    # Ensure row length matches headers
                    row = row + [""] * (len(headers) - len(row)) if len(row) < len(headers) else row[:len(headers)]
                    
                    # Pre-process data with custom parsing for numeric types
                    processed_row_dict = {}
                    for header, value in zip(headers, row):
                        col_type = table_info.schema.get(header, "string").lower()
                        cleaned_value = value.strip() if value else ""
                        
                        if col_type in ["currency", "percentage", "float", "integer"] and cleaned_value:
                            # Use enhanced numeric parsing
                            parsed_value = self._parse_numeric_value(cleaned_value, col_type)
                            if parsed_value is not None:
                                processed_row_dict[header] = parsed_value
                                if row_idx < 5:  # Log first 5 successful conversions for debugging
                                    print(f"  ✓ Parsed '{cleaned_value}' → {parsed_value} ({col_type})")
                            else:
                                # If enhanced parsing fails, try basic conversion
                                try:
                                    if col_type == "integer":
                                        processed_row_dict[header] = int(float(cleaned_value.replace(',', '')))
                                    else:
                                        processed_row_dict[header] = float(cleaned_value.replace(',', ''))
                                    parsing_stats["warnings"].append(f"Row {row_idx+1}: Basic parsing used for '{cleaned_value}' in {header}")
                                except (ValueError, TypeError):
                                    processed_row_dict[header] = None
                                    parsing_stats["warnings"].append(f"Row {row_idx+1}: Failed to parse '{cleaned_value}' in {header}, set to NULL")
                        else:
                            # Non-numeric types or empty values
                            processed_row_dict[header] = cleaned_value if cleaned_value else None
                    
                    # Validate with Pydantic (should pass since we pre-processed)
                    try:
                        validated_row = pydantic_model(**processed_row_dict)
                        validated_rows.append(validated_row.model_dump())
                        parsing_stats["success"] += 1
                    except Exception as pydantic_error:
                        logger.warning(f"Row {row_idx + 1} Pydantic validation failed: {pydantic_error}")
                        parsing_stats["failed"] += 1
                        # Try to salvage the row by setting problematic fields to None
                        salvaged_row = {}
                        for header in headers:
                            try:
                                # Test each field individually
                                test_dict = {header: processed_row_dict.get(header)}
                                pydantic_model(**{h: None for h in headers if h != header}, **test_dict)
                                salvaged_row[header] = processed_row_dict.get(header)
                            except:
                                salvaged_row[header] = None
                                parsing_stats["warnings"].append(f"Row {row_idx+1}: Set {header} to NULL due to validation error")
                        
                        try:
                            validated_row = pydantic_model(**salvaged_row)
                            validated_rows.append(validated_row.model_dump())
                            parsing_stats["success"] += 1
                        except:
                            parsing_stats["failed"] += 1
                            continue
                    
                except Exception as e:
                    logger.warning(f"Row {row_idx + 1} processing failed: {e}")
                    parsing_stats["failed"] += 1
                    continue

            # Report parsing statistics
            total_rows = len(data_rows)
            print(f"\nParsing Statistics:")
            print(f"  Total rows processed: {total_rows}")
            print(f"  Successfully validated: {parsing_stats['success']}")
            print(f"  Failed validation: {parsing_stats['failed']}")
            print(f"  Success rate: {(parsing_stats['success']/total_rows*100):.1f}%")
            
            if parsing_stats["warnings"]:
                print(f"  Warnings: {len(parsing_stats['warnings'])}")
                # Show first 5 warnings
                for warning in parsing_stats["warnings"][:5]:
                    print(f"    - {warning}")
                if len(parsing_stats["warnings"]) > 5:
                    print(f"    ... and {len(parsing_stats['warnings']) - 5} more warnings")

            # Insert validated data
            if validated_rows:
                with self.engine.connect() as conn:
                    conn.execute(insert(table), validated_rows)
                    conn.commit()
                
                print(f"✓ Successfully stored {len(validated_rows)} validated rows")
                logger.info(f"Successfully stored {len(validated_rows)} rows in {table_info.name}")

                # Generate detailed description after successful storage
                print("Generating detailed table description...")
                detailed_description = self._generate_detailed_description(table_info, len(validated_rows))
                print(f"✓ Generated detailed description ({len(detailed_description)} characters)")

                # Update schema with detailed description and mark as complete
                if table_info.name in self.schemas:
                    self.schemas[table_info.name]['description'] = detailed_description
                    self.schemas[table_info.name]['status'] = 'complete'
                    self.schemas[table_info.name]['rows_stored'] = len(validated_rows)
                    self._save_schemas()
                    print(f"✓ Updated schema file with detailed description")
                else:
                    logger.warning(f"Table {table_info.name} not found in schemas when updating description")

                return True
            else:
                print("✗ No valid rows to store")
                logger.warning(f"No valid rows to store in {table_info.name}")
                return False

        except Exception as e:
            logger.error(f"Error storing table {table_info.name}: {str(e)}")
            print(f"✗ Error storing table {table_info.name}: {str(e)}")
            return False

    def get_stored_schemas(self) -> Dict:
        """Get all stored table schemas."""
        return self.schemas

    def get_table_info(self, table_name: str) -> Optional[Dict]:
        """Get information about a specific table."""
        return self.schemas.get(table_name)

    # Backward compatibility methods
    def extract_content(self, pdf_path: str) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        result = self.optimized_extract_and_store(pdf_path)
        # Convert to legacy format
        return {
            "text_chunks": result["text_chunks"],
            "tables": [],  # Tables are now stored directly
            "table_names": [info["name"] for info in result["tables_info"]]
        }

    def store_table(self, table_data: List[List[str]], table_name: str) -> bool:
        """Legacy method for backward compatibility."""
        logger.warning("Using legacy store_table method. Consider using optimized_extract_and_store instead.")
        
        if not table_data:
            return False
            
        # Create basic table info for legacy support
        headers = table_data[0] if table_data else []
        basic_schema = {
            self._sanitize_column_name(header) if header else f"col_{i}": "string" 
            for i, header in enumerate(headers)
        }
        table_info = TableInfo(
            name=table_name,
            schema=basic_schema,
            description="Legacy table",
            data=table_data,
            column_count=len(table_data[0])
        )
        
        return self._store_table_with_schema(table_info)