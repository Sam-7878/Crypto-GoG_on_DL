# fraud_detection/mc_extension/score_extractor.py
import os, pathlib, torch, pickle, argparse
from torch_geometric.loader import DataLoader
from fraud_detection.shared.base_model import GoGModel
from fraud_detection.shared.utils import (
    set_global_seed, ensure_dir, save_pickle, load_yaml, get_logger, select_device
)

LOGGER = get_logger("ScoreExtractor")

class ScoreExtractor:
    def __init__(self, ckpt_path: str, device=None):
        self.device = select_device() if device is None else device
        LOGGER.info(f"Loading GoG checkpoint from {ckpt_path}")
        self.model = GoGModel.load_from_checkpoint(ckpt_path).to(self.device)

    @torch.no_grad()
    def extract(self, pyg_data):
        """
        하나의 PyG Data 에 대해
        Returns
            node_scores: tensor [N]  (fraud probability)
            graph_score: float       (mean fraud prob)
        """
        pyg_data = pyg_data.to(self.device)
        prob = self.model.predict_proba(pyg_data)  # [N]
        return prob.cpu(), float(prob.mean())

    def run_dataset(self, dataset, batch_size=1):
        """
        dataset: list[PyG Data]
        Returns
            node_list: list[tensor]  # 각 요소 shape [N_i]
            graph_list: list[float]  # 각 요소 shape 1
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        node_list, graph_list = [], []
        for idx, data in enumerate(loader):
            node_scores, graph_score = self.extract(data)
            node_list.append(node_scores)
            graph_list.append(graph_score)
            if (idx + 1) % 100 == 0:
                LOGGER.info(f"Processed {idx+1}/{len(loader)} graphs")
        return node_list, graph_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="fraud_detection/mc_extension/config.yaml")
    parser.add_argument("--out_dir", type=str, default=None, help="override config score_dir")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_global_seed(cfg["global_seed"])

    out_dir = args.out_dir or cfg["score_dir"]
    ensure_dir(out_dir)

    # --- 데이터 로드 ---
    test_data_path = cfg["test_data"]
    LOGGER.info(f"Loading dataset from {test_data_path}")
    # dataset = torch.load(test_data_path)          # list[PyG Data]
    # 변경
    # PyTorch 2.6+ 기본값이 torch.load(..., weights_only=True) 로 바뀜.
    # import torch_geometric.data
    # torch.serialization.add_safe_globals([torch_geometric.data.Data,
    #                                     torch_geometric.data.DataEdgeAttr])
    # weights_only=False 로 우회
    dataset = torch.load(test_data_path, weights_only=False)  # list[PyG Data]

    if not isinstance(dataset, list):
        dataset = [dataset]

    # --- 모델 & 추출 ---
    extractor = ScoreExtractor(cfg["gog_ckpt"])
    node_scores, graph_scores = extractor.run_dataset(
        dataset, batch_size=cfg.get("score_batch", 1)
    )

    # --- 저장 ---
    node_file = os.path.join(out_dir, "node_scores.pkl")
    graph_file = os.path.join(out_dir, "graph_scores.pkl")
    save_pickle(node_scores, node_file)
    save_pickle(graph_scores, graph_file)
    LOGGER.info(f"Saved node_scores to {node_file}")
    LOGGER.info(f"Saved graph_scores to {graph_file}")


if __name__ == "__main__":
    main()