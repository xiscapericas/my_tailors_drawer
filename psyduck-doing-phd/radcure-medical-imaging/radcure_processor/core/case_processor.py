"""Main case processor that orchestrates the entire RADCURE processing pipeline."""

import os
import numpy as np
from typing import List, Optional, Dict
from radcure_processor.io.aws_handler import AWSHandler
from radcure_processor.io.file_handler import FileHandler
from radcure_processor.io.nifti_handler import NIfTIHandler
from radcure_processor.core.dicom_handler import DICOMHandler
from radcure_processor.core.segmentator import TotalSegmentatorWrapper
from radcure_processor.core.mask_generator import MaskGenerator
from radcure_processor.utils.organ_dictionary import OrganDictionary
from radcure_processor.utils.image_processing import ImageProcessor
from radcure_processor.visualization.visualizer import MedicalImageVisualizer


class CaseProcessor:
    """Main class for processing RADCURE cases."""
    
    def __init__(
        self,
        main_path: str,
        aws_bucket_name: str,
        aws_folder: str,
        organ_dictionary_path: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: str = 'eu-west-1',
        total_segmentator_tasks: Optional[List[str]] = None,
        reverse_slices: bool = False,
        slice_expansion: int = 5
    ):
        """
        Initialize case processor.
        
        Parameters
        ----------
        main_path : str
            Main path for storing processed cases
        aws_bucket_name : str
            AWS S3 bucket name
        aws_folder : str
            AWS S3 folder prefix
        organ_dictionary_path : str, optional
            Path to organ dictionary JSON file
        aws_access_key_id : str, optional
            AWS access key ID
        aws_secret_access_key : str, optional
            AWS secret access key
        aws_region : str
            AWS region name
        total_segmentator_tasks : List[str], optional
            List of TotalSegmentator tasks to run. Default includes head/neck tasks.
        reverse_slices : bool
            Whether to reverse slice order
        slice_expansion : int
            Number of slices to add above and below tumor region
        """
        self.main_path = main_path
        self.main_path_retrain = os.path.join(main_path, 'TotalSegmentatorRetrain')
        self.reverse_slices = reverse_slices
        self.slice_expansion = slice_expansion
        
        # Initialize components
        self.aws_handler = AWSHandler(
            bucket_name=aws_bucket_name,
            aws_folder=aws_folder,
            access_key_id=aws_access_key_id,
            secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        
        self.file_handler = FileHandler()
        self.nifti_handler = NIfTIHandler()
        self.dicom_handler = DICOMHandler()
        
        self.segmentator = TotalSegmentatorWrapper(fast=False)
        self.organ_dictionary = OrganDictionary(organ_dictionary_path)
        self.mask_generator = MaskGenerator(self.organ_dictionary)
        self.visualizer = MedicalImageVisualizer()
        
        # Default TotalSegmentator tasks
        if total_segmentator_tasks is None:
            self.tasks_to_run = [
                'head_glands_cavities',
                'head_muscles',
                'headneck_bones_vessels',
                'headneck_muscles',
                'oculomotor_muscles',
                'craniofacial_structures'
            ]
        else:
            self.tasks_to_run = total_segmentator_tasks
        
        # Create directories
        os.makedirs(self.main_path_retrain, exist_ok=True)
    
    def process_case(self, radcure_case_id: str) -> Dict[str, str]:
        """
        Process a single RADCURE case through the entire pipeline.
        
        Parameters
        ----------
        radcure_case_id : str
            Case ID (e.g., 'RADCURE-0005')
        
        Returns
        -------
        Dict[str, str]
            Dictionary with paths to output files
        """
        print(f'Starting processing for {radcure_case_id}')
        
        local_folder = os.path.join(self.main_path_retrain, radcure_case_id)
        radcure_case_id_zip = radcure_case_id + '.zip'
        zip_path = os.path.join(self.main_path_retrain, radcure_case_id_zip)
        
        try:
            # Step 1: Download zip
            print('Step 1: Downloading zip')
            if not os.path.exists(zip_path):
                self.aws_handler.download_case(
                    radcure_case_id,
                    self.main_path_retrain
                )
            
            # Step 2: Unzip
            print('Step 2: Unzipping')
            if not os.path.exists(local_folder):
                self.file_handler.unzip_file(zip_path, local_folder)
            
            # Step 3: Get DICOM paths
            print('Step 3: Getting DICOM paths')
            dicom_folder_path = self.file_handler.get_dicom_path(
                local_folder, radcure_case_id
            )
            ct_mask_paths = self.file_handler.get_ct_and_mask_paths(dicom_folder_path)
            dicom_folder_ct_path = ct_mask_paths['ct_path']
            dicom_folder_mask_path = ct_mask_paths['mask_path']
            
            # Step 4: Convert to NIfTI
            print('Step 4: Converting to NIfTI')
            nifti_output_path = self.nifti_handler.convert_dicom_to_nifti(
                dicom_folder_ct_path,
                radcure_case_id,
                local_folder
            )
            
            # Step 5: Load tumor mask and get slices
            print('Step 5: Loading tumor mask and getting slices')
            contours = self.dicom_handler.load_tumor_mask(
                dicom_folder_ct_path,
                dicom_folder_mask_path
            )
            non_zero_tumor_mask = ImageProcessor.get_non_zero_slices(contours)
            
            # Expand slices
            start_slices_mask = max(min(non_zero_tumor_mask) - self.slice_expansion, 0)
            end_slices_mask = max(non_zero_tumor_mask) + self.slice_expansion
            non_zero_tumor_mask_expanded = list(
                range(start_slices_mask, end_slices_mask + 1)
            )
            
            # Reverse if needed
            if self.reverse_slices:
                non_zero_tumor_mask_expanded = [
                    contours.shape[2] - idx for idx in non_zero_tumor_mask_expanded
                ]
                non_zero_tumor_mask_expanded = non_zero_tumor_mask_expanded[::-1]
            
            # Verify slice count
            ct_slices_to_keep = self.file_handler.get_ct_slices_to_keep(
                non_zero_tumor_mask_expanded,
                dicom_folder_ct_path
            )
            if len(ct_slices_to_keep) != len(non_zero_tumor_mask_expanded):
                raise Exception(
                    f'Mismatch: {len(ct_slices_to_keep)} CT slices vs '
                    f'{len(non_zero_tumor_mask_expanded)} mask slices'
                )
            
            # Step 5b: Save and align tumor mask with CT
            print('Step 5b: Saving and aligning tumor mask with CT')
            mask_nifti_path = os.path.join(local_folder, f'{radcure_case_id}_tumor_mask_aligned.nii.gz')
            self.nifti_handler.save_and_align_mask_with_ct(
                contours,
                nifti_output_path,
                mask_nifti_path
            )
            
            # Step 6: Run TotalSegmentator
            print('Step 6: Running TotalSegmentator')
            total_segmentator_output = self.segmentator.run_tasks(
                radcure_case_id,
                nifti_output_path,
                local_folder,
                self.tasks_to_run
            )
            
            # Step 7: Generate background
            print('Step 7: Generating background masks')
            background_array_int = self.mask_generator.generate_background_array(
                non_zero_tumor_mask_expanded,
                nifti_output_path
            )
            
            # Step 8: Generate combined mask
            print('Step 8: Generating combined mask')
            combined_mask_array, updated_dict = self.mask_generator.generate_combined_mask(
                non_zero_tumor_mask_expanded,
                background_array_int,
                total_segmentator_output
            )
            
            # Step 9: Add tumor to mask
            print('Step 9: Adding tumor to mask')
            combined_mask_array_tumor = self.mask_generator.update_combined_mask_with_tumor(
                mask_nifti_path,
                non_zero_tumor_mask_expanded,
                combined_mask_array
            )
            
            # Step 10: Generate CT images
            print('Step 10: Generating CT images')
            ct_img_array = self.mask_generator.generate_ct_images(
                nifti_output_path,
                non_zero_tumor_mask_expanded
            )
            
            # Step 11: Save results
            print('Step 11: Saving results')
            results_path = os.path.join(local_folder, 'output')
            results_path_images = os.path.join(results_path, 'image')
            results_path_labels = os.path.join(results_path, 'labels')
            os.makedirs(results_path_images, exist_ok=True)
            os.makedirs(results_path_labels, exist_ok=True)
            
            # Save NIfTI files
            image_path = self.nifti_handler.save_as_nii(
                ct_img_array,
                results_path_images,
                radcure_case_id,
                dtype_needed=False
            )
            label_path = self.nifti_handler.save_as_nii(
                combined_mask_array_tumor,
                results_path_labels,
                radcure_case_id,
                dtype_needed=True
            )
            
            # Step 12: Generate visualization PDF
            print('Step 12: Generating visualization PDF')
            pdf_path = os.path.join(results_path, f'{radcure_case_id}.pdf')
            self.visualizer.show_two_niis_side_by_side(
                image_path,
                label_path,
                save_pdf_path=pdf_path,
                show=False
            )
            
            # Step 13: Update dictionary
            print('Step 13: Updating organ dictionary')
            if self.organ_dictionary.dictionary_path:
                self.organ_dictionary.save()
            
            return {
                'image_path': image_path,
                'label_path': label_path,
                'pdf_path': pdf_path,
                'status': 'success'
            }
            
        except Exception as e:
            print(f'Error processing {radcure_case_id}: {e}')
            # Cleanup on failure
            self.file_handler.cleanup_case_files(local_folder, zip_path)
            raise
    
    def process_multiple_cases(
        self,
        case_ids: Optional[List[str]] = None,
        processed_cases_file: Optional[str] = None,
        failed_cases_file: Optional[str] = None
    ) -> None:
        """
        Process multiple cases.
        
        Parameters
        ----------
        case_ids : List[str], optional
            List of case IDs to process. If None, processes all available cases.
        processed_cases_file : str, optional
            Path to file for logging successful cases
        failed_cases_file : str, optional
            Path to file for logging failed cases
        """
        if case_ids is None:
            # Get all cases from AWS
            all_cases = self.aws_handler.list_cases()
            
            # Filter out already processed cases
            try:
                already_processed = os.listdir(self.main_path_retrain)
                max_id = max([
                    int(e.split('-')[1].split('.')[0])
                    for e in already_processed
                    if e.startswith('RADCURE-') and ('.zip' in e or os.path.isdir(
                        os.path.join(self.main_path_retrain, e)
                    ))
                ])
                case_ids = [
                    c for c in all_cases
                    if c not in already_processed
                    and int(c.split('-')[1]) > max_id
                ]
            except:
                case_ids = all_cases
            
            print(f'Pending cases to process: {len(case_ids)}')
        
        for radcure_case_id in case_ids:
            try:
                result = self.process_case(radcure_case_id)
                if processed_cases_file:
                    with open(processed_cases_file, "a") as logf:
                        logf.write(radcure_case_id + "\n")
                print(f"✓ Successfully processed: {radcure_case_id}")
            except Exception as e:
                error_message = str(e)
                print(f"✗ Failed to process {radcure_case_id}: {error_message}")
                if failed_cases_file:
                    with open(failed_cases_file, "a") as logf:
                        logf.write(f"{radcure_case_id}: {error_message}\n")

    def download_and_visualize_case(
        self,
        radcure_case_id: str,
        slice_indices: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Dict[str, str]:
        """
        Download a RADCURE case and visualize the DICOM CT directly.
        
        Parameters
        ----------
        radcure_case_id : str
            Case ID (e.g., 'RADCURE-0005')
        slice_indices : List[int], optional
            Specific slice indices to visualize. If None, shows slices at 25%, 50%, 75%
        save_path : str, optional
            Path to save the visualization image
        show : bool
            If True, displays the figure interactively
        
        Returns
        -------
        Dict[str, str]
            Dictionary with paths to downloaded files and visualization
        """
        print(f'Downloading and visualizing {radcure_case_id}')
        
        local_folder = os.path.join(self.main_path_retrain, radcure_case_id)
        radcure_case_id_zip = radcure_case_id + '.zip'
        zip_path = os.path.join(self.main_path_retrain, radcure_case_id_zip)
        
        try:
            # Step 1: Download zip
            print('Step 1: Downloading zip')
            if not os.path.exists(zip_path):
                self.aws_handler.download_case(
                    radcure_case_id,
                    self.main_path_retrain
                )
            
            # Step 2: Unzip
            print('Step 2: Unzipping')
            if not os.path.exists(local_folder):
                self.file_handler.unzip_file(zip_path, local_folder)
            
            # Step 3: Get DICOM paths
            print('Step 3: Getting DICOM paths')
            dicom_folder_path = self.file_handler.get_dicom_path(
                local_folder, radcure_case_id
            )
            ct_mask_paths = self.file_handler.get_ct_and_mask_paths(dicom_folder_path)
            dicom_folder_ct_path = ct_mask_paths['ct_path']
            
            # Step 4: Visualize DICOM
            print('Step 4: Visualizing DICOM')
            if save_path is None:
                save_path = os.path.join(local_folder, f'{radcure_case_id}_dicom_preview.png')
            
            ct_volume = self.visualizer.visualize_dicom_series(
                dicom_folder_ct_path,
                slice_indices=slice_indices,
                save_path=save_path,
                show=show
            )
            
            return {
                'case_folder': local_folder,
                'zip_path': zip_path,
                'dicom_ct_path': dicom_folder_ct_path,
                'visualization_path': save_path,
                'status': 'success'
            }
            
        except Exception as e:
            print(f'Error downloading/visualizing {radcure_case_id}: {e}')
            raise

