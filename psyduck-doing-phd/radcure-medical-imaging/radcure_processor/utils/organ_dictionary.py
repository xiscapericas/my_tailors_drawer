"""Organ dictionary management for tracking organ indices."""

import json
import os
from typing import Dict, Optional


class OrganDictionary:
    """Manages organ-to-index mapping dictionary."""
    
    def __init__(self, dictionary_path: Optional[str] = None):
        """
        Initialize organ dictionary.
        
        Parameters
        ----------
        dictionary_path : str, optional
            Path to JSON file containing organ dictionary.
            If None, starts with default dictionary.
        """
        self.dictionary_path = dictionary_path
        if dictionary_path and os.path.exists(dictionary_path):
            with open(dictionary_path, "r") as f:
                self.dictionary = json.load(f)
            print(f"Loaded dictionary from: {dictionary_path}")
        else:
            self.dictionary = {
                'background': 0,
                'other-tissue': 1
            }
            if dictionary_path:
                print(f"Dictionary file not found, starting with default")
    
    def get(self, organ_name: str) -> Optional[int]:
        """
        Get index for an organ name.
        
        Parameters
        ----------
        organ_name : str
            Name of the organ
        
        Returns
        -------
        int or None
            Organ index or None if not found
        """
        return self.dictionary.get(organ_name)
    
    def add_organ(self, organ_name: str) -> int:
        """
        Add a new organ to the dictionary with next available index.
        
        Parameters
        ----------
        organ_name : str
            Name of the organ to add
        
        Returns
        -------
        int
            Index assigned to the organ
        """
        if organ_name in self.dictionary:
            return self.dictionary[organ_name]
        
        max_index = max(self.dictionary.values()) if self.dictionary else -1
        new_index = max_index + 1
        self.dictionary[organ_name] = new_index
        return new_index
    
    def add_tumor_index(self) -> int:
        """
        Add tumor (GTVp) index to dictionary.
        
        Returns
        -------
        int
            Index assigned to tumor
        """
        return self.add_organ('GTVp')
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save dictionary to JSON file.
        
        Parameters
        ----------
        path : str, optional
            Path to save. If None, uses dictionary_path.
        """
        save_path = path or self.dictionary_path
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(self.dictionary, f, indent=4)
            print(f"Dictionary saved to: {save_path}")
    
    def get_max_index(self) -> int:
        """
        Get maximum index in dictionary.
        
        Returns
        -------
        int
            Maximum index value
        """
        return max(self.dictionary.values()) if self.dictionary else 0
    
    def __getitem__(self, key: str) -> int:
        """Allow dictionary-like access."""
        return self.dictionary[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if organ is in dictionary."""
        return key in self.dictionary
    
    def copy(self) -> Dict[str, int]:
        """Return a copy of the dictionary."""
        return self.dictionary.copy()

