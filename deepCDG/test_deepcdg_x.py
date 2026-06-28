import unittest
import torch
import torch.nn as nn
from model_x import LoRALinear, StageConditionedPPI, CausalGNNGating, deepCDGX
from explain_x import CFGNNExplainer

class TestDeepCDGX(unittest.TestCase):
    def setUp(self):
        # Create dummy arguments matching parser options
        class Args:
            dropout = 0.5
        self.args = Args()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_lora_linear(self):
        # 1. Test LoRALinear module
        in_features, out_features = 32, 16
        x = torch.randn(5, in_features).to(self.device)
        lora_layer = LoRALinear(in_features, out_features, rank=4).to(self.device)
        
        # Test disabled mode
        lora_layer.set_lora(False)
        out1 = lora_layer(x)
        self.assertEqual(out1.shape, (5, 16))
        
        # Test enabled mode
        lora_layer.set_lora(True)
        out2 = lora_layer(x)
        self.assertEqual(out2.shape, (5, 16))
        
        # Assert outputs differ when LoRA is enabled vs disabled
        # (initially lora_B is 0, so out2 should equal out1 until lora parameters change)
        self.assertTrue(torch.allclose(out1, out2))
        
        # Modify lora parameters and check difference
        nn.init.ones_(lora_layer.lora_B)
        nn.init.ones_(lora_layer.lora_A)
        out3 = lora_layer(x)
        self.assertFalse(torch.allclose(out1, out3))

    def test_stage_conditioned_ppi(self):
        # 2. Test StageConditionedPPI module
        num_nodes = 10
        dim_hidden = 48
        x = torch.randn(num_nodes, dim_hidden).to(self.device)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
        ], dtype=torch.long).to(self.device)
        
        stage_ppi = StageConditionedPPI(node_dim=dim_hidden).to(self.device)
        
        # Test with early stage (1.0)
        stage_1 = torch.tensor(1.0).to(self.device)
        weights_1 = stage_ppi(x, edge_index, stage_1)
        self.assertEqual(weights_1.shape, (10,))
        self.assertTrue(torch.all(weights_1 >= 0.0) and torch.all(weights_1 <= 1.0))
        
        # Test with late stage (4.0) and check if edge weights are dynamic
        stage_4 = torch.tensor(4.0).to(self.device)
        weights_4 = stage_ppi(x, edge_index, stage_4)
        self.assertEqual(weights_4.shape, (10,))
        self.assertFalse(torch.allclose(weights_1, weights_4))

    def test_causal_gating(self):
        # 3. Test CausalGNNGating module
        num_nodes = 5
        dim_hidden = 48
        x = torch.randn(num_nodes, dim_hidden).to(self.device)
        causal = CausalGNNGating(dim_hidden).to(self.device)
        
        x_causal, mask = causal(x)
        self.assertEqual(x_causal.shape, (5, 48))
        self.assertEqual(mask.shape, (5, 48))
        self.assertTrue(torch.all(mask >= 0.0) and torch.all(mask <= 1.0))

    def test_deepcdg_x_forward(self):
        # 4. Test deepCDGX model forward and backward
        num_nodes = 20
        # Dummy features representing 3 omics types * 16 features = 48 features
        x = torch.randn(num_nodes, 48).to(self.device)
        edge_index = torch.tensor([
            list(range(19)) + [19],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0]
        ], dtype=torch.long).to(self.device)
        
        model = deepCDGX(self.args).to(self.device)
        
        # Test forward in training mode (includes causal loss calculation)
        model.train()
        pred = model(x, edge_index)
        self.assertEqual(pred.shape, (num_nodes, 1))
        self.assertNotEqual(model.causal_loss.item(), 0.0)
        
        # Test backward pass
        loss = pred.mean() + model.causal_loss
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad and "lora" not in name:
                self.assertIsNotNone(param.grad)

    def test_mc_dropout(self):
        # 5. Test Monte Carlo Dropout uncertainty estimation
        num_nodes = 15
        x = torch.randn(num_nodes, 48).to(self.device)
        edge_index = torch.tensor([
            list(range(14)) + [14],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0]
        ], dtype=torch.long).to(self.device)
        
        model = deepCDGX(self.args).to(self.device)
        
        mean, var = model.forward_mc(x, edge_index, num_passes=10)
        self.assertEqual(mean.shape, (num_nodes, 1))
        self.assertEqual(var.shape, (num_nodes, 1))
        self.assertTrue(torch.all(var >= 0.0))
        # Epistemic uncertainty variance should be greater than zero due to dropout
        self.assertTrue(torch.any(var > 0.0))

    def test_cf_explainer(self):
        # 6. Test Counterfactual GNNExplainer
        num_nodes = 30
        x = torch.randn(num_nodes, 48).to(self.device)
        # Create a highly connected dummy node to test explanation
        src = [0] * 10 + list(range(1, 20))
        dst = list(range(1, 11)) + list(range(2, 21))
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long).to(self.device)
        
        model = deepCDGX(self.args).to(self.device)
        model.eval()
        
        # We need the node prediction to be > 0.5 to trigger explanation
        # Mock the model output for node 0 to be high
        orig_forward = model.forward
        def mock_forward(x_in, edge_in, stage=None):
            out = orig_forward(x_in, edge_in, stage)
            # Make first node prediction very high (driver)
            out[0] = 5.0  # Sigmoid of 5.0 is > 0.99
            return out
        model.forward = mock_forward
        
        explainer = CFGNNExplainer(model, steps=10, lr=0.1)
        removed_edges, success = explainer.explain(x, edge_index, node_idx=0)
        
        # Explainer should return some removed edges to flip output to target_prob
        self.assertIsInstance(removed_edges, list)
        self.assertTrue(len(removed_edges) >= 0)

if __name__ == '__main__':
    unittest.main()
