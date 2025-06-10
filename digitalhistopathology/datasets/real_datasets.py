#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import glob
import math
import os

import numpy as np
import seaborn as sns


from digitalhistopathology.engineered_features.engineered_features import EngineeredFeatures
from digitalhistopathology.embeddings.gene_embedding import GeneEmbedding
from digitalhistopathology.embeddings.image_embedding import ImageEmbedding


from digitalhistopathology.datasets.spatial_dataset import SpatialDataset


class HER2Dataset(SpatialDataset):
    """HER2-positive breast cancer dataset from https://www.nature.com/articles/s41467-021-26271-2."""

    PALETTE = {
        "invasive cancer": "red",
        "cancer in situ": "orange",
        "immune infiltrate": "yellow",
        "breast glands": "green",
        "connective tissue": "blue",
        "adipose tissue": "cyan",
        "undetermined": "lightgrey",
    }
    ORDER_LABEL = [
        "invasive cancer",
        "cancer in situ",
        "immune infiltrate",
        "breast glands",
        "connective tissue",
        "adipose tissue",
        "undetermined",
    ]

    def __init__(self, patches_folder=None, saving_emb_folder=None):
        """
        Args:
            patches_folder (str, optional): Folder that contains patches. Defaults to None.
            saving_emb_folder (str, optional): Path to folder where to save the embeddings. Defaults to None.
        """

        super().__init__(patches_folder)
        self.saving_emb_folder = saving_emb_folder
        self.name = "HER2"
        super().__init__(patches_folder)
        # Image embeddings
        self.init_patches_filenames()
        self.label_filenames = sorted(
            glob.glob("../data/HER2_breast_cancer/meta/*.tsv")
        )

        # Gene embeddings
        AREA_SPOT_HER2_PIXEL2 = 8000  # from QPath
        self.spot_diameter = 2 * int(np.sqrt(AREA_SPOT_HER2_PIXEL2 / math.pi))
        self.genes_count_filenames = sorted(
            glob.glob("../data/HER2_breast_cancer/count-matrices/*")
        )
        self.genes_spots_filenames = sorted(
            glob.glob("../data/HER2_breast_cancer/spot-selections/*")
        )
        self.images_filenames = sorted(
            glob.glob("../data/HER2_breast_cancer/images/HE/*")
        )
        self.samples_names = [
            f.split("/")[-1].split(".")[0][0:2] for f in self.images_filenames
        ]

    def get_image_embeddings(self, model, filename="ie", emb_path=None):
        """Compute the image embedding or load it if it already exists.

        Args:
            model (models.PretrainedModel): Pretrained deep learning vision encoder.
            filename (str, optional): Filename of the .h5ad image embedding. Defaults to "ie".

        Returns:
            image_embedding.ImageEmbedding: image embedding
        """
        self.init_patches_filenames()
        ie = ImageEmbedding(
            patches_filenames=self.patches_filenames,
            patches_info_filename=self.patches_info_filename,
            label_files=self.label_filenames,
            pretrained_model=model,
            name=self.name + "_" + model.name,
        )
        try:
            if os.path.exists(self.saving_emb_folder):
                loading_file = os.path.join(
                    self.saving_emb_folder, "{}.h5ad".format(filename)
                )
            elif emb_path is not None:
                loading_file = os.path.join(emb_path, "{}.h5ad".format(filename))
            else:
                loading_file = (
                    "../results/embeddings/images_embeddings/save/{}/{}.h5ad".format(
                        model.name.lower(), self.saving_emb_folder
                    )
                )
            ie.load_embeddings(loading_file)
            if "label" not in ie.emb.obs.columns:
                ie.add_label()
            print("Fill emb with: {}".format(loading_file))
            print(ie.emb)
        except Exception as e:
            print("Cannot load images embeddings: {}".format(e))
        return ie

    def get_gene_embeddings(
        self,
        compute_emb=False,
        preprocessing=False,
        spot_normalization=False,
        load=False,
        filename="ge",
    ):
        """Compute the gene embedding or load it if it already exists.

        Args:
            compute_emb (bool, optional): If we need compute the gene embedding. Defaults to False.
            preprocessing (bool, optional): If preprocessing is done. Defaults to False.
            spot_normalization (bool, optional): If spots normalization is done if preprocessing. Defaults to False.
            load (bool, optional): If we need to load gene embedding from file. Defaults to False.
            filename (str, optional): Filename of the .h5ad gene embedding. Defaults to "ge".

        Returns:
            gene_embedding.GeneEmebdding: gene embedding
        """
        # Filter only spot patches
        patches_filenames = [
            f for f in self.patches_filenames if "spot" in f.split("/")[-1]
        ]
        ge = GeneEmbedding(
            spot_diameter_fullres=self.spot_diameter,
            samples_names=self.samples_names,
            genes_count_filenames=self.genes_count_filenames,
            spots_filenames=self.genes_spots_filenames,
            image_filenames=self.images_filenames,
            label_files=self.label_filenames,
            patches_filenames=patches_filenames,
            name=self.name,
            st_method="old_st",
        )
        if compute_emb:
            ge.compute_embeddings()
            if preprocessing:
                ge.emb = ge.preprocessing(spot_norm=spot_normalization)
            print(ge.emb)
        elif load:
            try:
                loading_file = os.path.join(
                    self.saving_emb_folder, "{}.h5ad".format(filename)
                )
                ge.load_embeddings(
                    loading_file,
                    columns_ast=[],
                    columns_numeric=[
                        "imagecol",
                        "imagerow",
                        "x",
                        "y",
                        "total_genes_count",
                        "predicted_label",
                    ],
                )
                if "label" not in ge.emb.obs.columns:
                    ge.add_label()
                print("Fill emb with: {}".format(loading_file))
                print(ge.emb)
            except Exception as e:
                print("Cannot load genes embeddings: {}".format(e))
        return ge

    def get_engineered_features(self, remove_nan=True, filename="ef", emb_path=None):
        """Compute the engineered features or load it if it already exists.

        Args:
            remove_nan (bool, optional): If rows with NaN. Defaults to True.
            filename (str, optional): Filename of the .h5ad engineered features. Defaults to "ef".

        Returns:
            engineered_features.EngineeredFeatures: engineered features
        """
        ef = EngineeredFeatures(
            patches_info_filename=self.patches_info_filename,
            name=self.name,
            label_files=self.label_filenames,
        )

        if (self.saving_emb_folder is not None) and (
            os.path.exists(self.saving_emb_folder)
        ):
            loading_file = os.path.join(
                self.saving_emb_folder, "{}.h5ad".format(filename)
            )
        elif emb_path is not None:
            loading_file = os.path.join(emb_path, "{}.h5ad".format(filename))
        else:
            loading_file = (
                "../results/embeddings/engineered_features/save/{}.h5ad".format(
                    self.saving_emb_folder
                )
            )

        try:
            if (self.saving_emb_folder is not None) and (
                os.path.exists(self.saving_emb_folder)
            ):
                loading_file = os.path.join(
                    self.saving_emb_folder, "{}.h5ad".format(filename)
                )
            elif emb_path is not None:
                loading_file = os.path.join(emb_path, "{}.h5ad".format(filename))
            else:
                loading_file = (
                    "../results/embeddings/engineered_features/save/{}.h5ad".format(
                        self.saving_emb_folder
                    )
                )

            print(f"loading_file: {loading_file}", flush=True)
            ef.load_embeddings(loading_file)

            if remove_nan:
                # remove rows with nan
                print("Remove Nan...")
                print(ef.emb.shape)
                ef.emb = ef.emb[~np.isnan(ef.emb.X).any(axis=1), :]
                print(ef.emb.shape)
            if "label" not in ef.emb.obs.columns:
                ef.add_label()
            print("Fill emb with: {}".format(loading_file))
            print(ef.emb)
        except Exception as e:
            print("Cannot load engineered features: {}".format(e))
        return ef

    def get_palette_2():
        palette = sns.color_palette(palette="bright")
        palette_2 = dict()
        palette_2[0] = palette[0]
        palette_2[1] = palette[5]
        palette_2[2] = palette[1]
        palette_2[3] = palette[2]
        palette_2[4] = palette[6]
        palette_2[5] = palette[3]
        palette_2[6] = palette[4]
        palette_2[7] = palette[7]
        palette_2[8] = palette[8]
        palette_2[9] = palette[9]
        palette_2[10] = sns.color_palette(palette="colorblind")[5]
        return palette_2



class TNBCDataset(SpatialDataset):
    PALETTE = {
        "invasive cancer": "#017801",
        "Necrosis": "#000000",
        "Fat tissue": "#000080",
        "Vessels": "#dc0000",
        "Lactiferous duct": "#9980e6",
        "in situ": "#ccffcc",
        "Lymphoid nodule": "#80801a",
        "Lymphocyte": "#c4417f",
        "Stroma": "#ff9980",
        "Nerve": "#4d8080",
        "Heterologous elements": "#808080",
    }

    ORDER_LABEL = [
        "invasive cancer",
        "Necrosis",
        "Fat tissue",
        "Vessels",
        "Lactiferous duct",
        "in situ",
        "Lymphoid nodule",
        "Lymphocyte",
        "Stroma",
        "Nerve",
        "Heterologous elements",
    ]

    def __init__(self, patches_folder=None, saving_emb_folder=None, dataDir=None):
        """
        Args:
            patches_folder (str, optional): Folder that contains patches. Defaults to None.
            saving_emb_folder (str, optional): Path to folder where to save the embeddings. Defaults to None.
            dataDir (str, optional): Path to the folder that contains the images. Defaults to None.
        """
        self.dataDir = dataDir
        super().__init__(patches_folder)
        self.saving_emb_folder = saving_emb_folder
        self.name = "TNBC"
        # Image embeddings
        self.init_patches_filenames()
        self.label_filenames = sorted(glob.glob("../results/TNBC/labels/*.csv"))
        self.spot_diameter = 348
        self.genes_count_filenames = sorted(
            glob.glob("../results/TNBC/count-matrices/*.csv")
        )
        self.genes_spots_filenames = self.label_filenames
        self.images_filenames = [
            os.path.join(
                self.dataDir, "Images", "imagesHD", os.path.basename(name)[:-4] + ".jpg"
            )
            for name in self.genes_count_filenames
        ]
        self.samples_names = [
            f.split("/")[-1].split(".")[0] for f in self.images_filenames
        ]

    def get_image_embeddings(self, model, filename="ie", emb_path=None):
        """Compute the image embedding or load it if it already exists.

        Args:
            model (models.PretrainedModel): Pretrained deep learning vision encoder.
            filename (str, optional): Filename of the .h5ad image embedding. Defaults to "ie".

        Returns:
            image_embedding.ImageEmbedding: image embedding
        """
        self.init_patches_filenames()
        ie = ImageEmbedding(
            patches_filenames=self.patches_filenames,
            patches_info_filename=self.patches_info_filename,
            label_files=self.label_filenames,
            pretrained_model=model,
            name=self.name + "_" + model.name,
        )
        try:
            if os.path.exists(self.saving_emb_folder):
                loading_file = os.path.join(
                    self.saving_emb_folder, "{}.h5ad".format(filename)
                )
            elif emb_path is not None:
                loading_file = os.path.join(emb_path, "{}.h5ad".format(filename))
            else:
                loading_file = (
                    "../results/embeddings/images_embeddings/save/{}/{}.h5ad".format(
                        model.name.lower(), self.saving_emb_folder
                    )
                )
            ie.load_embeddings(loading_file)
            if "label" not in ie.emb.obs.columns:
                ie.add_label()
            print("Fill emb with: {}".format(loading_file))
            print(ie.emb)
        except Exception as e:
            print("Cannot load images embeddings: {}".format(e))
        return ie

    def get_gene_embeddings(
        self,
        compute_emb=False,
        preprocessing=False,
        spot_normalization=False,
        load=False,
        filename="ge",
    ):
        """Compute the gene embedding or load it if it already exists.

        Args:
            compute_emb (bool, optional): If we need compute the gene embedding. Defaults to False.
            preprocessing (bool, optional): If preprocessing is done. Defaults to False.
            spot_normalization (bool, optional): If spots normalization is done if preprocessing. Defaults to False.
            load (bool, optional): If we need to load gene embedding from file. Defaults to False.
            filename (str, optional): Filename of the .h5ad gene embedding. Defaults to "ge".

        Returns:
            gene_embedding.GeneEmebdding: gene embedding
        """
        ge = GeneEmbedding(
            spot_diameter_fullres=self.spot_diameter,
            samples_names=self.samples_names,
            genes_count_filenames=self.genes_count_filenames,
            spots_filenames=self.genes_spots_filenames,
            image_filenames=self.images_filenames,
            label_files=self.label_filenames,
            patches_filenames=self.patches_filenames,
            name=self.name,
            st_method="old_st",
        )
        if compute_emb:
            ge.compute_embeddings()
            if preprocessing:
                ge.emb = ge.preprocessing(spot_norm=spot_normalization)
            print(ge.emb)
        elif load:
            try:
                loading_file = os.path.join(
                    self.saving_emb_folder, "{}.h5ad".format(filename)
                )
                ge.load_embeddings(
                    loading_file,
                    columns_ast=[],
                    columns_numeric=[
                        "imagecol",
                        "imagerow",
                        "x",
                        "y",
                        "total_genes_count",
                        "predicted_label",
                    ],
                )
                if "label" not in ge.emb.obs.columns:
                    ge.add_label()
                print("Fill emb with: {}".format(loading_file))
                print(ge.emb)
            except Exception as e:
                print("Cannot load genes embeddings: {}".format(e))
        return ge

    def get_engineered_features(self, remove_nan=True, filename="ef", emb_path=None):
        """Compute the engineered features or load it if it already exists.

        Args:
            remove_nan (bool, optional): If rows with NaN. Defaults to True.
            filename (str, optional): Filename of the .h5ad engineered features. Defaults to "ef".

        Returns:
            engineered_features.EngineeredFeatures: engineered features
        """
        ef = EngineeredFeatures(
            patches_info_filename=self.patches_info_filename,
            name=self.name,
            label_files=self.label_filenames,
        )

        if (self.saving_emb_folder is not None) and (
            os.path.exists(self.saving_emb_folder)
        ):
            loading_file = os.path.join(
                self.saving_emb_folder, "{}.h5ad".format(filename)
            )
        elif emb_path is not None:
            loading_file = os.path.join(emb_path, "{}.h5ad".format(filename))
        else:
            loading_file = (
                "../results/embeddings/engineered_features/save/{}.h5ad".format(
                    self.saving_emb_folder
                )
            )

        try:
            if (self.saving_emb_folder is not None) and (
                os.path.exists(self.saving_emb_folder)
            ):
                loading_file = os.path.join(
                    self.saving_emb_folder, "{}.h5ad".format(filename)
                )
            elif emb_path is not None:
                loading_file = os.path.join(emb_path, "{}.h5ad".format(filename))
            else:
                loading_file = (
                    "../results/embeddings/engineered_features/save/{}.h5ad".format(
                        self.saving_emb_folder
                    )
                )

            print(f"loading_file: {loading_file}", flush=True)
            ef.load_embeddings(loading_file)

            if remove_nan:
                # remove rows with nan
                print("Remove Nan...")
                print(ef.emb.shape)
                ef.emb = ef.emb[~np.isnan(ef.emb.X).any(axis=1), :]
                print(ef.emb.shape)
            if "label" not in ef.emb.obs.columns:
                ef.add_label()
            print("Fill emb with: {}".format(loading_file))
            print(ef.emb)
        except Exception as e:
            print("Cannot load engineered features: {}".format(e))
        return ef

    def get_palette_2():
        palette = sns.color_palette(palette="bright")
        palette_2 = dict()
        palette_2[0] = palette[0]
        palette_2[1] = palette[5]
        palette_2[2] = palette[1]
        palette_2[3] = palette[2]
        palette_2[4] = palette[6]
        palette_2[5] = palette[3]
        palette_2[6] = palette[4]
        palette_2[7] = palette[7]
        palette_2[8] = palette[8]
        palette_2[9] = palette[9]
        palette_2[10] = sns.color_palette(palette="colorblind")[5]
        return palette_2