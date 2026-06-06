import pytest
import torch
from dataclasses import dataclass
from typing import NamedTuple

from nn.concept_encoder import (
    ConceptEncoderConfig,
    ConceptEncoder,
)

@dataclass
class SpecialTokens:
    """Special token IDs used across tests."""
    PAD: int = 0    # [PAD]
    CLS: int = 1    # [CLS] [bos]
    BOS: int = 1    # [BOS] [bos]
    SEP: int = 2    # [SEP] [eos]
    EOS: int = 2    # [EOS]
    MASK: int = 3   # [MASK]

class MLMBatch(NamedTuple):
    """Represents a batch of data for masked language modeling."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

@dataclass
class ModelConfig:
    """Base configuration for model testing."""
    vocab_size: int
    concept_num: int = 4
    hidden_size: int = 8
    num_hidden_layers: int = 1
    num_attention_heads: int = 1
    intermediate_size: int = 16
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1


class ModelConfigs:
    """Predefined model configurations for testing."""
    @staticmethod
    def tiny_1l_1h():
        return ModelConfig(
            vocab_size=16,
            concept_num=4,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=16
        )
    
    @staticmethod
    def small_2l_1h():
        return ModelConfig(
            vocab_size=16,
            concept_num=8,
            hidden_size=12,
            num_hidden_layers=2,
            num_attention_heads=1,
            intermediate_size=16
        )
    @staticmethod
    def small_2l_2h():
        return ModelConfig(
            vocab_size=16,
            concept_num=8,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16
        )
    
    @staticmethod
    def medium_3l_4h():
        return ModelConfig(
            vocab_size=16,
            concept_num=8,
            hidden_size=16,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=32
        )

class TestBase:
    """Base class for all encoder tests."""
    @pytest.fixture
    def tokens(self):
        return SpecialTokens()
    
    def create_encoder_config(self, model_config: ModelConfig, tokens: SpecialTokens) -> ConceptEncoderConfig:
        """Creates ConceptEncoderConfig from ModelConfig and SpecialTokens."""
        return ConceptEncoderConfig(
            vocab_size=model_config.vocab_size,
            concept_num=model_config.concept_num,
            hidden_size=model_config.hidden_size,
            num_hidden_layers=model_config.num_hidden_layers,
            num_attention_heads=model_config.num_attention_heads,
            intermediate_size=model_config.intermediate_size,
            hidden_dropout_prob=model_config.hidden_dropout_prob,
            attention_probs_dropout_prob=model_config.attention_probs_dropout_prob,
            pad_token_id=tokens.PAD,
            eos_token_id=tokens.SEP,
            bos_token_id=tokens.CLS,
            cls_token_id=tokens.CLS,
            sep_token_id=tokens.SEP,
            mask_token_id=tokens.MASK,
        )
    
    def create_mlm_batch(self, model_config: ModelConfig, tokens: SpecialTokens) -> MLMBatch:
        """Creates a test batch for masked language modeling."""
        # Create labels with tokens in range [4, vocab_size-1]
        labels = torch.tensor([
            [tokens.CLS, 4, 5, 6, 7, 8, 9, tokens.SEP],
            [tokens.CLS, 5, 6, 7, tokens.SEP, 0, 0, 0],
        ], dtype=torch.long)
        
        model_input_ids = labels.clone()
        model_input_ids[:, 2] = tokens.MASK  # Mask the third token
        attention_mask = (labels != tokens.PAD).long()
        
        return MLMBatch(model_input_ids, attention_mask, labels)

class TestConceptEncoder(TestBase):
    """Tests for the base ConceptEncoder model."""
    
    @pytest.mark.parametrize("config_factory", [
        ModelConfigs.tiny_1l_1h,
        ModelConfigs.small_2l_1h,
        ModelConfigs.small_2l_2h,
        ModelConfigs.medium_3l_4h,
    ])
    def test_encoder_output_shapes(self, config_factory, tokens):
        """Test encoder output shapes across different configurations."""
        model_config = config_factory()
        encoder_config = self.create_encoder_config(model_config, tokens)
        model = ConceptEncoder(encoder_config)
        model.eval()
        
        batch = self.create_mlm_batch(model_config, tokens)
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        labels = batch.labels
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        # check the shapr of encoder output which is the concept_representation
        expected_shape = (
            batch_size,
            model_config.concept_num, # concept_num
            model_config.hidden_size # hidden_size
        )
        assert outputs.last_hidden_state.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {outputs.last_hidden_state.shape}"
        )
        
    @pytest.mark.parametrize("config_factory", [
        ModelConfigs.tiny_1l_1h,
        ModelConfigs.small_2l_1h,   
        ModelConfigs.small_2l_2h,
        ModelConfigs.medium_3l_4h
                                                ])
    def test_all_hidden_states_output(self, config_factory, tokens):
        """Test that all hidden states from layers are properly returned."""
        model_config = config_factory()
        encoder_config = self.create_encoder_config(model_config, tokens)
        model = ConceptEncoder(encoder_config)
        model.eval()
        
        batch = self.create_mlm_batch(model_config, tokens)
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Check if we get hidden states for each layer + input embeddings
        assert len(outputs.hidden_states) == model_config.num_hidden_layers + 1
        
        # Check shapes of hidden states
        for hidden_state in outputs.hidden_states:
            assert hidden_state.shape == (
                batch_size,
                model_config.concept_num,
                model_config.hidden_size
            )
            
    @pytest.mark.skip(reason="Need to figure it out which attention states should be returned")     
    @pytest.mark.parametrize("config_factory", [
        ModelConfigs.tiny_1l_1h,
        ModelConfigs.small_2l_1h,   
        ModelConfigs.small_2l_2h,
        ModelConfigs.medium_3l_4h
    ])
    def test_all_attention_states_output(self, config_factory, tokens):
        """Test that all attention states from layers are properly returned."""
        model_config = config_factory()
        encoder_config = self.create_encoder_config(model_config, tokens)
        model = ConceptEncoder(encoder_config)
        model.eval()
        
        batch = self.create_mlm_batch(model_config, tokens)
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Check if we get hidden states for each layer + input embeddings
        assert len(outputs.attentions) == model_config.num_hidden_layers + 1
        
        # Check shapes of hidden states
        for layer_attention in outputs.attentions:
            assert layer_attention.shape == (
                batch_size,
                model_config.num_attention_heads,
                seq_length,
            )
    

class TestConceptEncoderEmbeddings(TestBase):
    """Tests for the embedding components of ConceptEncoder."""
    
    @pytest.mark.parametrize("config_factory", [ModelConfigs.tiny_1l_1h])
    def test_token_embeddings_shape(self, config_factory, tokens):
        """Test token embeddings output the correct shape."""
        model_config = config_factory()
        encoder_config = self.create_encoder_config(model_config, tokens)
        model = ConceptEncoder(encoder_config)
        
        batch = self.create_mlm_batch(model_config, tokens)
        input_ids = batch.input_ids
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        
        embeddings = model.token_embeddings(input_ids)
        
        expected_shape = (batch_size, seq_length, model_config.hidden_size)
        assert embeddings.shape == expected_shape
    
    @pytest.mark.parametrize("config_factory", [ModelConfigs.tiny_1l_1h])
    def test_concept_embeddings_initialization(self, config_factory, tokens):
        """Test concept embeddings are properly initialized."""
        model_config = config_factory()
        encoder_config = self.create_encoder_config(model_config, tokens)
        model = ConceptEncoder(encoder_config)
        
        # Get concept embeddings directly
        concept_ids = torch.arange(model_config.concept_num)
        embeddings = model.concept_embeddings(concept_ids)
        
        expected_shape = (model_config.concept_num, model_config.hidden_size)
        assert embeddings.shape == expected_shape
        
        # Check if embeddings are not zero and properly initialized
        assert not torch.allclose(embeddings, torch.zeros_like(embeddings))
        assert torch.abs(embeddings.mean()) < 0.1  # Roughly centered around 0

class TestConceptEncoderAttention(TestBase):
    """Tests for attention mechanisms in ConceptEncoder."""
    
    @pytest.mark.parametrize("config_factory", [ModelConfigs.tiny_1l_1h])
    def test_attention_mask_effect(self, config_factory, tokens):
        """Test that attention mask properly masks out padded tokens."""
        model_config = config_factory()
        encoder_config = self.create_encoder_config(model_config, tokens)
        model = ConceptEncoder(encoder_config)
        model.eval()
        
        # Create two identical sequences but with different attention masks
        input_ids = torch.tensor([[tokens.CLS, 4, 5, tokens.SEP]], dtype=torch.long)
        
        # First mask: attend to all tokens
        mask1 = torch.ones_like(input_ids)
        # Second mask: ignore the last token
        mask2 = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
        
        with torch.no_grad():
            output1 = model(input_ids, attention_mask=mask1)
            output2 = model(input_ids, attention_mask=mask2)
        
        # Outputs should be different due to different attention masks
        assert not torch.allclose(output1.last_hidden_state, output2.last_hidden_state)
