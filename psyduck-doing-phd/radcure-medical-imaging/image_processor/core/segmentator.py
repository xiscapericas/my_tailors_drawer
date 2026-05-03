"""TotalSegmentator wrapper for running segmentation tasks."""

import os
from totalsegmentator.python_api import totalsegmentator
from typing import List, Optional


class TotalSegmentatorWrapper:
    """Wrapper for TotalSegmentator API."""
    
    def __init__(
        self,
        fast: bool = False,
        nr_thr_resamp: Optional[int] = None,
        nr_thr_saving: Optional[int] = None,
    ):
        """
        Initialize TotalSegmentator wrapper.
        
        Parameters
        ----------
        fast : bool
            Whether to use fast mode
        nr_thr_resamp : int, optional
            Passed to totalsegmentator() when set (lower uses less RAM during resampling).
        nr_thr_saving : int, optional
            Passed to totalsegmentator() when set (lower uses less RAM when saving).
        """
        self.fast = fast
        self.nr_thr_resamp = nr_thr_resamp
        self.nr_thr_saving = nr_thr_saving
    
    def run_task(
        self,
        path_to_input: str,
        output_path: str,
        task_type: str
    ) -> bool:
        """
        Run a single TotalSegmentator task.
        
        Parameters
        ----------
        path_to_input : str
            Path to input NIfTI file
        output_path : str
            Directory for saving masks
        task_type : str
            Task type (e.g., 'head_glands_cavities')
        
        Returns
        -------
        bool
            True if successful
        """
        print(f'Running TotalSegmentator task: {task_type}')
        kwargs = dict(
            input=path_to_input,
            output=output_path,
            task=task_type,
            fast=self.fast,
        )
        if self.nr_thr_resamp is not None:
            kwargs["nr_thr_resamp"] = self.nr_thr_resamp
        if self.nr_thr_saving is not None:
            kwargs["nr_thr_saving"] = self.nr_thr_saving
        totalsegmentator(**kwargs)
        return True
    
    def run_tasks(
        self,
        radcure_case_id: str,
        input_nifti_path: str,
        output_base_folder: str,
        tasks_to_run: List[str]
    ) -> str:
        """
        Run multiple TotalSegmentator tasks.
        
        Parameters
        ----------
        radcure_case_id : str
            Case ID
        input_nifti_path : str
            Path to input NIfTI file
        output_base_folder : str
            Base folder for outputs
        tasks_to_run : List[str]
            List of task types to run
        
        Returns
        -------
        str
            Path to total segmentator output folder
        """
        total_segmentator_output = os.path.join(
            output_base_folder, "total_segmentator_output"
        )
        os.makedirs(total_segmentator_output, exist_ok=True)
        
        for task_type in tasks_to_run:
            print(f"Running task: {task_type}")
            task_dir = os.path.join(total_segmentator_output, task_type)
            if not os.path.exists(task_dir):
                os.makedirs(task_dir, exist_ok=True)
                self.run_task(
                    path_to_input=input_nifti_path,
                    output_path=task_dir,
                    task_type=task_type
                )
            else:
                print(f"Task already run for {task_type}")
        
        return total_segmentator_output + "/"

