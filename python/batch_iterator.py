import os
from typing import List, Callable, Literal, Tuple

import numpy as np
import pandas as pd
import xgboost
import pickle

class Iterator(xgboost.DataIter):
    """A custom iterator for loading files in batches."""

    def __init__(
        self, device: Literal["cpu", "cuda"], file_paths: List[Tuple[str, str]], metadata_path=None
    ) -> None:
        self.device = device
        self._file_paths = file_paths
        # self._file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
        self.metadata_path = metadata_path if metadata_path is not None else None
        if (self.metadata_path is not None) and (os.path.exists(self.metadata_path)):
            with open(self.metadata_path, "rb") as metadata_file:
                metadata = pickle.load(metadata_file)
        else:
            metadata = None
        self.metadata = metadata
        self._it = 0
        # XGBoost will generate some cache files under the current directory with the
        # prefix "cache"
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def load_file(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single batch of data."""
        X_path = self._file_paths[self._it]
        # When the `ExtMemQuantileDMatrix` is used, the device must match. GPU cannot
        # consume CPU input data and vice-versa.
        if self.device == "cpu":
            df = pd.read_parquet(X_path).sample(frac=1, ignore_index=True, random_state=42)
            # X = pd.read_parquet(X_path)
            # if self.metadata is not None:
            #     X['BDTWeight']=1
            #     for year in metadata['years']:
            #         for 
                
            # else:
            #     w = abs(X['finalWeight'])
            
            # w = abs(X['finalWeight'])
            w = abs(df['BDT_cat_weight'])*1e3
            y = df['y']
            # w = X['BDT_cat_weight']*1e-3 #CHANGED!!!
            
            #ORIGINAL Version:
            X = df.drop(['category','y', 'weight', 'finalWeight', 'sum_genWeight', 
                         'weight_nonorm', 'BDT_weight', 'BDT_cat_weight', 'BDT_era_weight', 
                         'FatJet0_pt', 'FatJet0_msd', 'BDT_Ncat_weight'#'MC_name'
                       ], axis=1)

            #FOR DEBUGGING WITH EAF Version:
           #  omit_cols = ['isSignal', 'weight','weight_final', 'BDT_score',
           #   'category', 'FatJet0_pt', 'FatJet0_msd', 'FatJet0_msdmatched',
           #   'MC_name', 'y', 'sumW', 'genWeight',
           #   'Photon110EB_TightID_TightIso', 'Photon30EB_TightID_TightIso',
           #   'FatJet0_pnetMass', 'FatJet0_pnetTXbb', 'FatJet0_pnetTXgg',
           #   'FatJet0_pnetTXcc', 'FatJet0_pnetTXqq', 'FatJet0_pnetXbbXcc', 'FatJet0_pnetTQCD',
           #   'FatJet0_ParTPQCD', 'FatJet0_ParTPXbb', 'FatJet0_ParTPXcc',
           #    'FatJet0_ParTPXqq', 'FatJet0_ParTPXcs', 'FatJet0_ParTPXbbVsQCD',
           #   'FatJet0_ParTPTopbWq', 'FatJet0_ParTPTopbWqq',
           #   'Photon200', 'FatJet0_ParTPXccVsQCD', 'FatJet0_ParTPXbbXcc', 'FatJet0_ParTmassX2p',
           #   'FatJet1_pnetMass', 'FatJet1_pnetTXbb', 'FatJet1_pnetTXcc',
           #   'FatJet1_pnetTXqq', 'FatJet1_pnetTXgg',
           #   'Photon0_eta', 'Photon0_phi', 'Photon0_pt',
           #   'AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35',
           # 'QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65',
           # 'AK8PFJet425_SoftDropMass40', 'PFJet500', 'PFHT1050', 'BDT_cat_weight']
           #  X = df.drop(omit_cols, axis=1)
            
            # X['year'] = X['year'].astype(np.int64)
            # for col in X.columns:
            #     print(col, type(X[col].iloc[0]))
            # y = pd.read_parquet(X_path, columns = ['y']).sample(frac=1, ignore_index=True, random_state=42)
            # print("nans in y: ", y.isna().sum())
            # y = np.load(y_path)
            # print(
            #     "batch:",
            #     self._it,
            #     # "N =", len(df),
            #     "sum weights =", w.sum(),
            #     # "mean weight =", w.mean(),
            #     "min =", w.min(),
            #     "max =", w.max(),
            #     # "class weighted sums =",
            #     # df.groupby("y")["BDT_cat_weight"].sum().to_dict()
            #  )
        else:
            import cupy as cp

            X = cp.load(X_path)
            y = cp.load(y_path)

        assert X.shape[0] == y.shape[0]
        return X, w, y

    def next(self, input_data: Callable) -> bool:
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function
        is called by XGBoost during the construction of ``DMatrix``

        """
        if self._it == len(self._file_paths):
            # return False to let XGBoost know this is the end of iteration
            return False

        # input_data is a keyword-only function passed in by XGBoost and has the similar
        # signature to the ``DMatrix`` constructor.
        X, w, y = self.load_file()
        input_data(data=X, label=y, weight=w)
        self._it += 1
        return True

    def reset(self) -> None:
        """Reset the iterator to its beginning"""
        self._it = 0