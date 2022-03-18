
class ResultTracker:
    def __init__(self,out_dir:str) -> None:
        # Create logging table
        res_df = pd.DataFrame(
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

        self.curr_sample_dict = {}


        # Save file

    def save_to_file():
        pass

    def upload_to_w_and_b():
        pass

    def add_erm_res():
        pass
    def add_pnml_res():
        pass

    def finish_sample():
        pass