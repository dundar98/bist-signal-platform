"""
BIST100 Universe Validator.

This module enforces the CRITICAL CONSTRAINT that the system
MUST ONLY train on BIST100 stocks. All data requests pass through
this validator.
"""

import logging
from typing import List, Set
from datetime import date

from utils import normalize_ticker

logger = logging.getLogger(__name__)


# Current BIST100 constituents as of December 2024
# This list should be updated periodically or fetched from an official source
BIST100_SYMBOLS: Set[str] = {
    # Banks
    "AKBNK", "GARAN", "HALKB", "ISCTR", "YKBNK", "VAKBN", "TSKB", "QNBFB", "ALBRK", "ICBCT",
    # Holdings
    "KCHOL", "SAHOL", "DOHOL", "ECZYT", "SISE", "TAVHL", "TKFEN", "GLYHO", "KOZAL", "KOZAA",
    # Industrials
    "EREGL", "KRDMD", "ISDMR", "TUPRS", "PETKM", "SASA", "BRISA", "FROTO", "TOASO", "OTKAR",
    "TTRAK", "DOAS", "ARCLK", "VESTL", "KLMSN", "CEMTS", "CIMSA", "ADEL", "AEFES", "CCOLA",
    # Airlines & Transport
    "THYAO", "PGSUS", "CLEBI", "BEYAZ", "RYSAS",
    # Retail & Consumer
    "BIMAS", "MGROS", "SOKM", "MAVI", "VAKKO", "DESA", "BIZIM",
    # Telecom & Tech
    "TCELL", "TTKOM", "LOGO", "INDES", "LINK", "KAREL", "ESCOM", "NETAS",
    # Energy & Utilities
    "AKSEN", "ENKAI", "ODAS", "AYDEM", "AKSA", "ZOREN", "AKENR", "EUPWR",
    # Real Estate & Construction
    "EKGYO", "ISGYO", "EMLAK", "ENTRA", "KLGYO", "HLGYO", "TRGYO",
    # Healthcare & Pharma
    "ECILC", "SELEC", "DEVA",
    # Other
    "ASELS", "KORDS", "ULKER", "TATGD", "PNLSN", "TMSN", "SMRTG", "MPARK", "GESAN",
    "KONTR", "OYAKC", "BUCIM", "BAGFS", "BTCIM", "GOLTS", "HEKTS", "IEYHO", "ISMEN",
    "KERVT", "MIATK", "PAPIL", "QUAGR", "RGYAS", "SRVGY", "TBORG", "TKURU", "TURSG",
    "VERUS", "YEOTK", "YATAS", "ALARK", "AGHOL", "ANHYT", "ANSGR", "ARDYZ", "AYCES",
    "BANVT", "BERA", "BIOEN", "BRYAT", "CANTE", "DITAS", "EBEBK", "EGEEN", "EGGUB",
    "ENJSA", "ESEN", "GEDZA", "GOODY", "GOZDE", "GUBRF", "HURGZ", "IPEKE", "ITTFH",
    "JANTS", "KARSN", "KATMR", "KAYSE", "KERVN", "KMPUR", "KRONT", "KRTEK", "KTLEV",
    "KUTPO", "MAGEN", "MAKIM", "MEGAP", "MERCN", "METRO", "MOBTL", "MRSHL", "NUGYO",
    "OBASE", "OFSYM", "ORGE", "OSMEN", "OSTIM", "PARSN", "PENGD", "POLHO", "PRKAB",
    "PSDTC", "RODRG", "ROYAL", "RUBNS", "SAFKR", "SAMAT", "SANFM", "SARKY", "SEGYO",
    "SEKUR", "SILVR", "SKBNK", "SKYMD", "SMART", "SNKRN", "SODSN", "SONME", "SURGY",
    "SUWEN", "TRGYO", "TUKAS", "TUREX", "ULUUN", "USAK", "VAKFN", "YYLGD", "ZRGYO",
}


class BIST100ValidationError(Exception):
    """Raised when a non-BIST100 symbol is encountered."""
    pass


class BIST100Validator:
    """
    Validates that symbols are in the BIST100 universe.
    
    This class is the gatekeeper for all data operations.
    No data should be loaded without passing through validation.
    """
    
    def __init__(self, symbols: Set[str] = None):
        """
        Initialize the validator.
        
        Args:
            symbols: Optional custom set of BIST100 symbols.
                    If None, uses the default BIST100_SYMBOLS.
        """
        self._symbols = symbols if symbols is not None else BIST100_SYMBOLS.copy()
        self._rejected_symbols: List[str] = []
        logger.info(f"BIST100Validator initialized with {len(self._symbols)} symbols")
    
    def is_valid_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is in the BIST100 universe.
        
        Args:
            symbol: Stock symbol to check (without exchange suffix)
            
        Returns:
            True if symbol is in BIST100, False otherwise
        """
        # Normalize: remove common suffixes and convert to uppercase
        normalized = self._normalize_symbol(symbol)
        return normalized in self._symbols
    
    def validate_symbol(self, symbol: str) -> str:
        """
        Validate a single symbol and return the normalized form.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            Normalized symbol if valid
            
        Raises:
            BIST100ValidationError: If symbol is not in BIST100
        """
        normalized = self._normalize_symbol(symbol)
        
        if normalized not in self._symbols:
            self._rejected_symbols.append(symbol)
            logger.warning(f"REJECTED: Symbol '{symbol}' is NOT in BIST100 universe")
            raise BIST100ValidationError(
                f"Symbol '{symbol}' is not in BIST100. "
                f"Only BIST100 stocks are allowed for training."
            )
        
        return normalized
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """
        Validate a list of symbols.
        
        Args:
            symbols: List of stock symbols to validate
            
        Returns:
            List of validated, normalized symbols
            
        Raises:
            BIST100ValidationError: If any symbol is not in BIST100
        """
        validated = []
        invalid = []
        
        for symbol in symbols:
            normalized = self._normalize_symbol(symbol)
            if normalized in self._symbols:
                validated.append(normalized)
            else:
                invalid.append(symbol)
                self._rejected_symbols.append(symbol)
        
        if invalid:
            logger.error(f"REJECTED: {len(invalid)} symbols not in BIST100: {invalid}")
            raise BIST100ValidationError(
                f"The following symbols are NOT in BIST100 and cannot be used: {invalid}. "
                f"Only BIST100 stocks are allowed for training."
            )
        
        logger.info(f"VALIDATED: {len(validated)} symbols confirmed in BIST100")
        return validated
    
    def filter_valid_symbols(self, symbols: List[str]) -> List[str]:
        """
        Filter and return only valid BIST100 symbols (non-raising version).
        
        Args:
            symbols: List of stock symbols to filter
            
        Returns:
            List of validated symbols (invalid ones are silently dropped)
        """
        valid = []
        for symbol in symbols:
            normalized = self._normalize_symbol(symbol)
            if normalized in self._symbols:
                valid.append(normalized)
            else:
                logger.debug(f"Filtered out non-BIST100 symbol: {symbol}")
        
        return valid
    
    def get_all_symbols(self) -> List[str]:
        """
        Get all BIST100 symbols.
        
        Returns:
            Sorted list of all BIST100 symbols
        """
        return sorted(self._symbols)
    
    def get_rejected_symbols(self) -> List[str]:
        """
        Get list of all rejected symbols during this session.
        
        Returns:
            List of rejected symbols (for audit)
        """
        return self._rejected_symbols.copy()
    
    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize a symbol to standard format using the project-wide normalizer."""
        return normalize_ticker(symbol)
    
    def add_symbol(self, symbol: str) -> None:
        """
        Add a symbol to the valid set (for dynamic updates).
        
        Args:
            symbol: Symbol to add
        """
        normalized = self._normalize_symbol(symbol)
        self._symbols.add(normalized)
        logger.info(f"Added {normalized} to BIST100 universe")
    
    def remove_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from the valid set.
        
        Args:
            symbol: Symbol to remove
        """
        normalized = self._normalize_symbol(symbol)
        self._symbols.discard(normalized)
        logger.info(f"Removed {normalized} from BIST100 universe")


# Singleton instance for convenience
_default_validator: BIST100Validator = None


def get_validator() -> BIST100Validator:
    """Get the default BIST100 validator instance."""
    global _default_validator
    if _default_validator is None:
        _default_validator = BIST100Validator()
    return _default_validator


def validate_bist100(symbols: List[str]) -> List[str]:
    """
    Convenience function to validate symbols against BIST100.
    
    Args:
        symbols: List of symbols to validate
        
    Returns:
        List of validated symbols
        
    Raises:
        BIST100ValidationError: If any symbol is invalid
    """
    return get_validator().validate_symbols(symbols)
