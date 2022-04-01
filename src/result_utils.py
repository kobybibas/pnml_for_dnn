import torch
import pandas as pd
import logging
import wandb
import os.path as osp

logger = logging.getLogger(__name__)


class ResultTracker:
    def __init__(self, out_dir: str, wandb_h) -> None:
        # Create logging table
        self.res_df = pd.DataFrame(
            columns=[
                "test_idx",
                "pnml_nf",
                "pnml_regret",
                "pnml_logloss",
                "pnml_is_correct",
                "pnml_probs",
                "genie_probs",
                "erm_logloss",
                "erm_is_correct",
                "erm_probs",
            ]
        )
        self.wandb_h = wandb_h
        self.out_dir = out_dir
        self.out_path = osp.join(self.out_dir, "res_df.pkl")

    def calc_performance(self, probs, true_label, learner_name: str):
        is_correct = torch.argmax(probs) == true_label
        logloss = -torch.log(probs[true_label])
        return {
            learner_name + "is_correct": is_correct.item(),
            learner_name + "logloss": logloss.item(),
        }

    def calc_test_sample_performance(
        self, test_idx, test_true_label, genie_probs, erm_probs
    ):
        pnml_nf = torch.sum(genie_probs)
        pnml_regret = torch.log(pnml_nf)
        pnml_probs = genie_probs / pnml_nf

        # pNML
        pnml_dict = self.calc_performance(
            pnml_probs, test_true_label, learner_name="pnml_"
        )
        pnml_dict["pnml_nf"] = pnml_nf.item()
        pnml_dict["pnml_regret"] = pnml_regret.item()
        pnml_dict["pnml_probs"] = pnml_probs.tolist()
        pnml_dict["genie_probs"] = genie_probs.tolist()

        # ERM
        erm_dict = self.calc_performance(
            erm_probs, test_true_label, learner_name="erm_"
        )
        erm_dict["erm_probs"] = erm_probs.tolist()

        # Combine
        res_dict = {**pnml_dict, **erm_dict}
        res_dict["test_idx"] = test_idx
        res_dict["test_true_label"] = test_true_label
        self.res_df = self.res_df.append(res_dict, ignore_index=True)

    def finish_sample(self):

        logger.info(self.res_df.iloc[-1])

        # Save to file
        self.res_df.to_pickle(self.out_path)

        # Upload
        test_table = wandb.Table(dataframe=self.res_df)
        self.wandb_h.run.log({"test_table": test_table})
        self.wandb_h.save(self.out_path)


     
