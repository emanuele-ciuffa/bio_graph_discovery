from dataclasses import dataclass

@dataclass
class Dp_report:
    def __init__(self,
                 selected_organism=None,
                 selected_gene_form=None,
                 dataset_count_by_organism=None,
                 dataset_count_by_gene_form=None,
                 not_recognized_elements_count=None,
                 dataset_triplets_count=None,
                 dataset_validated_triplets_count=None,
                 max_reaction_depth=None,
                 distinct_chemical_count=None,
                 distinct_protein_count=None,
                 distinct_reaction_count=None):
        """
        Build a report for Data Preparation
        :param selected_organism: selected organism (specie) \n
        :param selected_gene_form: selected gene form \n
        :param dataset_count_by_organism: number of records of dataset filtering the selected organism \n
        :param dataset_count_by_gene_form: number of records of dataset filtering the selected organism and gene form \n
        :param not_recognized_elements_count: records that have not been categorized as chemical, protein or interaction \n
        :param dataset_triplets_count: number of records of extracted semantic triplets (before validation) \n
        :param dataset_validated_triplets_count: number of records of validated semantic triplets  (after validation filtering) \n
        :param max_reaction_depth: max reaction depth (max nested level)
        :param distinct_chemical_count: number of distinct chemicals detected \n
        :param distinct_protein_count: number of distinct protein detected \n
        :param distinct_reaction_count: number of distinct reaction detected \n
        """

        self.selected_organism = selected_organism,
        self.selected_gene_form = selected_gene_form,
        self.dataset_count_by_organism = dataset_count_by_organism,
        self.dataset_count_by_gene_form = dataset_count_by_gene_form,
        self.not_recognized_elements_count = not_recognized_elements_count,
        self.dataset_triplets_count = dataset_triplets_count,
        self.dataset_validated_triplets_count = dataset_validated_triplets_count,
        self.max_reaction_depth = max_reaction_depth,
        self.distinct_chemical_count = distinct_chemical_count,
        self.distinct_protein_count = distinct_protein_count,
        self.distinct_reaction_count = distinct_reaction_count
