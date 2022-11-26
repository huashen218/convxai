import more_itertools as mit
import logging
import random
import math
import numpy as np
from torch.utils import data
from transformers import (
    BertTokenizerFast
)
from allennlp.data.batch import Batch
from allennlp.nn import util

import torch
import torch.nn.functional as F
from torch import backends

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZERS_PARALLELISM=False

class MaskError(Exception):
    pass



class Feature:
    def __init__(self, pad_length=50, tokenizer=None):
        # self.pad_length = pad_length
        if tokenizer is None:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
        self.pad_id, self.cls_id, self.sep_id = self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]
            )

    def extract(self, sents):
        results = [self.tokenizer.encode(s, add_special_tokens=False) for s in sents]
        # results = [self.tokenizer.encode(sents, add_special_tokens=False)]

        # turn to matrix with padding
        # matrix = np.ones([len(results), self.pad_length], dtype=np.int32) * self.pad_id
        # for i, res in enumerate(results):
        #     length = min(len(res), self.pad_length)
        #     matrix[i, :length] = res[:length]

        matrix = np.ones([len(results), len(results[0])], dtype=np.int32)
        for i, res in enumerate(results):
            length = len(res) 
            matrix[i, :length] = res[:length]

        cls_matrix = np.ones([len(results), 1]) * self.cls_id
        sep_matrix = np.ones([len(results), 1]) * self.sep_id
        matrix = np.hstack([cls_matrix, matrix, sep_matrix])
        
        return matrix



class PredDataset(data.Dataset):
    """For AspectPredict Writing Model
    """
    def __init__(self, x, bucket_num=None):
        self.x = x.astype(np.int64)

    def __len__(self):
        return self.x.shape[0] 

    def __getitem__(self, index):
        return self.x[index]




class Masker():
    """ 
    Class used to mask inputs for Editors.
    Two subclasses: RandomMasker and GradientMasker
    
    mask_frac: float 
        Fraction of input tokens to mask.
    editor_to_wrapper: allennlp.data.tokenizers.tokenizer 
        Wraps around Editor tokenizer.
        Has capabilities for mapping Predictor tokens to Editor tokens.
    max_tokens: int
        Maximum number of tokens a masked input should have.
    """
    
    def __init__(
            self,
            mask_frac, 
            editor_tok_wrapper, 
            max_tokens
        ):
        self.mask_frac = mask_frac
        self.editor_tok_wrapper = editor_tok_wrapper
        self.max_tokens = max_tokens
        
    def _get_mask_indices(self, editor_toks):
        """ Helper function to get indices of Editor tokens to mask. """
        raise NotImplementedError("Need to implement this in subclass")

    def get_all_masked_strings(self, editable_seg):
        """ Returns a list of masked inps/targets where each inp has 
        one word replaced by a sentinel token.
        Used for calculating fluency. """

        editor_toks = self.editor_tok_wrapper.tokenize(editable_seg)
        masked_segs = [None] * len(editor_toks)
        labels = [None] * len(editor_toks)

        for i, token in enumerate(editor_toks):
            token_start, token_end = token.idx, token.idx_end
            masked_segs[i] = editable_seg[:token_start] + \
                    Masker._get_sentinel_token(0) + editable_seg[token_end:]
            labels[i] = Masker._get_sentinel_token(0) + \
                    editable_seg[token_start:token_end] + \
                    Masker._get_sentinel_token(1)
        
        return masked_segs, labels       

    def _get_sentinel_token(idx):
        """ Helper function to get sentinel token based on given idx """

        return "<extra_id_" + str(idx) + ">"

    def _get_grouped_mask_indices(
            self, editable_seg, pred_idx, editor_mask_indices, 
            editor_toks, **kwargs):
        """ Groups consecutive mask indices.
        Applies heuristics to enable better generation:
            - If > 27 spans, mask tokens b/w neighboring spans as well.
                (See Appendix: observed degeneration after 27th sentinel token)
            - Mask max of 100 spans (since there are 100 sentinel tokens in T5)
        """

        if editor_mask_indices is None:
            editor_mask_indices = self._get_mask_indices(editable_seg, editor_toks, pred_idx, **kwargs)

        new_editor_mask_indices = set(editor_mask_indices)
        grouped_editor_mask_indices = [list(group) for group in \
                mit.consecutive_groups(sorted(new_editor_mask_indices))]

        if len(grouped_editor_mask_indices) > 27:
            for t_idx in editor_mask_indices:
                if t_idx + 2 in editor_mask_indices:
                    new_editor_mask_indices.add(t_idx + 1)
        
        grouped_editor_mask_indices = [list(group) for group in \
                mit.consecutive_groups(sorted(new_editor_mask_indices))]

        if len(grouped_editor_mask_indices) > 27:
            for t_idx in editor_mask_indices:
                if t_idx + 3 in editor_mask_indices:
                    new_editor_mask_indices.add(t_idx + 1)
                    new_editor_mask_indices.add(t_idx + 2)

        new_editor_mask_indices = list(new_editor_mask_indices)
        grouped_editor_mask_indices = [list(group) for group in \
                mit.consecutive_groups(sorted(new_editor_mask_indices))]
        
        grouped_editor_mask_indices = grouped_editor_mask_indices[:99]
        return grouped_editor_mask_indices

    def get_masked_string(
            self, editable_seg, pred_idx, 
            editor_mask_indices = None, **kwargs):
        """ Gets masked string masking tokens w highest predictor gradients.
        Requires mapping predictor tokens to Editor tokens because edits are
        made on Editor tokens. """
        editor_toks = self.editor_tok_wrapper.tokenize(editable_seg)

        ####### editable_seg = Hospitalizations decreased in Australia and Singapore but increased in Taiwan , Republic of China .
        ####### editor_toks = [▁Hospital, ization, s, ▁decreased, ▁in, ▁Australia, ▁and, ▁Singapore, ▁but, ▁increased, ▁in, ▁Taiwan, ▁, ,, ▁Republic, ▁of, ▁China, ▁, ., </s>]
        ####### editor_mask_indices = None

        grpd_editor_mask_indices = self._get_grouped_mask_indices(
                editable_seg, pred_idx, editor_mask_indices, 
                editor_toks, **kwargs)

        span_idx = len(grpd_editor_mask_indices) - 1
        label = Masker._get_sentinel_token(len(grpd_editor_mask_indices))
        masked_seg = editable_seg

        # Iterate over spans in reverse order and mask tokens
        for span in grpd_editor_mask_indices[::-1]:

            span_char_start = editor_toks[span[0]].idx
            span_char_end = editor_toks[span[-1]].idx_end
            end_token_idx = span[-1]

            # If last span tok is last t5 tok, heuristically set char end idx
            if span_char_end is None and end_token_idx == len(editor_toks)-1:
                span_char_end = span_char_start + 1

            if not span_char_end > span_char_start:
                raise MaskError
                
            label = Masker._get_sentinel_token(span_idx) + \
                    masked_seg[span_char_start:span_char_end] + label
            masked_seg = masked_seg[:span_char_start] + \
                    Masker._get_sentinel_token(span_idx) + \
                    masked_seg[span_char_end:]
            span_idx -= 1    

        return grpd_editor_mask_indices, editor_mask_indices, masked_seg, label
            

            

class GradientMasker(Masker):
    """ Masks spans based on gradients of Predictor wrt. given predicted label.

    mask_frac: float 
        Fraction of input tokens to mask.
    editor_to_wrapper: allennlp.data.tokenizers.tokenizer 
        Wraps around Editor tokenizer.
        Has capabilities for mapping Predictor tokens to Editor tokens.
    max_tokens: int
        Maximum number of tokens a masked input should have.
    grad_type: str, one of ["integrated_l1", "integrated_signed", 
            "normal_l1", "normal_signed", "normal_l2", "integrated_l2"]
        Specifies how gradient value should be calculated
            Integrated vs. normal:
                Integrated: https://arxiv.org/pdf/1703.01365.pdf
                Normal: 'Vanilla' gradient
            Signed vs. l1 vs. l2:
                Signed: Sum gradients over embedding dimension.
                l1: Take l1 norm over embedding dimension.
                l2: Take l2 norm over embedding dimension.
    sign_direction: One of [-1, 1, None]
        When grad_type is signed, determines whether we want to get most 
        negative or positive gradient values. 
        This should depend on what label is being used 
        (pred_idx argument to get_masked_string).
        For example, Stage One, we want to mask tokens that push *towards* 
        gold label, whereas during Stage Two, we want to mask tokens that 
        push *away* from the target label. 
        Sign direction plays no role if only gradient *magnitudes* are used 
        (i.e. if grad_type is not signed, but involves taking the l1/l2 norm.) 
    num_integrated_grad_steps: int
        Hyperparameter for integrated gradients. 
        Only used when grad_type is one of integrated types.
    """
    
    def __init__(
            self, 
            mask_frac, 
            editor_tok_wrapper, 
            predictor, 
            max_tokens, 
            grad_type = "normal_l2", 
            sign_direction = None,
            num_integrated_grad_steps = 10
        ):
        super().__init__(mask_frac, editor_tok_wrapper, max_tokens)
        
        self.predictor = predictor
        self.grad_type = grad_type
        self.num_integrated_grad_steps = num_integrated_grad_steps
        self.sign_direction = sign_direction

        if ("signed" in self.grad_type and sign_direction is None):
            error_msg = "To calculate a signed gradient value, need to " + \
                    "specify sign direction but got None for sign_direction"
            raise ValueError(error_msg)

        if sign_direction not in [1, -1, None]:
            error_msg = f"Invalid value for sign_direction: {sign_direction}"
            raise ValueError(error_msg)
        

        # temp_tokenizer = self.predictor._dataset_reader._tokenizer

        # # Used later to avoid skipping special tokens like <s>
        # self.predictor_special_toks = \
        #         temp_tokenizer.sequence_pair_start_tokens + \
        #         temp_tokenizer.sequence_pair_mid_tokens + \
        #         temp_tokenizer.sequence_pair_end_tokens + \
        #         temp_tokenizer.single_sequence_start_tokens + \
        #         temp_tokenizer.single_sequence_end_tokens

        self.temp_tokenizer = self.predictor.tokenizer

        self.predictor_special_toks = self.temp_tokenizer.all_special_tokens

    def batch_dataloader(self, input_string):
        feature = Feature(tokenizer=self.temp_tokenizer)
        x_text = feature.extract(input_string[:])
        dataset = PredDataset(x_text)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        return dataloader


    def _register_embedding_gradient_hooks(self, model, embeddings_gradients):
        def hook_layers(module, grad_in, grad_out):
            embeddings_gradients.append(grad_out[0])
        embedding_layer = model.bert.embeddings.word_embeddings
        hook = embedding_layer.register_backward_hook(hook_layers)
        return hook


    def _get_gradients_by_prob(self, instance, pred_idx):
        """ Helper function to get gradient values of predicted logit 
        Largely copied from Predictor class of AllenNLP """


        original_param_name_to_requires_grad_dict = {}
        # for param_name, param in self.predictor._model.named_parameters():
        #     original_param_name_to_requires_grad_dict[param_name] = \
        #             param.requires_grad
        #     param.requires_grad = True
        for param_name, param in self.predictor.model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = \
                    param.requires_grad
            param.requires_grad = True



        embedding_gradients: List[Tensor] = []

        ###### Reference: https://stackoverflow.com/questions/67142267/gradient-based-saliency-of-input-words-in-a-pytorch-model-from-transformers-libr
        # hooks: List[RemovableHandle] = self.predictor.model._register_embedding_gradient_hooks([embedding_gradients)
        hooks: List[RemovableHandle] = self._register_embedding_gradient_hooks(self.predictor.model, embedding_gradients)
        # hooks  = self._register_embedding_gradient_hooks(self.predictor.model, embedding_gradients)
        # print("-====>>>>>hooks", hooks)
        # exit()

        # instances = [instance]
        # dataset = Batch(instances)
        # dataset.index_instances(self.predictor._model.vocab)
        # dataset_tensor_dict = util.move_to_device(
        #         dataset.as_tensor_dict(), self.predictor.cuda_device)


        ###### instance = ["Hospitalizations decreased in Australia and Singapore but increased in Taiwan , Republic of China ."]
        dataloader = self.batch_dataloader([instance])
        x_batch = next(iter(dataloader)).to(device)

        with backends.cudnn.flags(enabled=False):
            # outputs = self.predictor._model.make_output_human_readable(
            #     self.predictor._model.forward(**dataset_tensor_dict) 
            # )

            y_batch = torch.tensor([1]*x_batch.size(0)).to(device)
            outputs = self.predictor.model(x_batch, labels=y_batch)

            # Differs here
            # prob = outputs["logits"][0][pred_idx] 
            prob = outputs[1][0][int(pred_idx)] ###### outputs[1] tensor([[-0.1297, -0.9615,  0.0416,  1.8316,  0.0538]]

            self.predictor.model.zero_grad()
            prob.backward()


        # for hook in hooks:
        #     hook.remove()
        hooks.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):     ###### len = 1, [0].shape = embedding_gradients torch.Size([1, 18, 768])
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        # Restore original requires_grad values of the parameters
        for param_name, param in self.predictor.model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]


        ######
        """
        grad_dict = {'grad_input_1': array([[[-0.00725925,  0.04185553, -0.02047579, ..., -0.01079098,                                                                                                                                 
                -0.00394276,  0.01109453],                                                                                                                                                                                              
                [-0.01863867,  0.13642767, -0.04722746, ..., -0.13016962,                                                                                                                                                                
                -0.00461602,  0.10560728],                                                                                                                                                                                              
                [-0.00047336,  0.03735472, -0.00956504, ..., -0.04319321,                                                                                                                                                                
                0.00550868,  0.01700047],                                                                                                                                                                                              
                ...,                                                                                                                                                                                                                     
                [ 0.04194318, -0.00398021, -0.01697866, ...,  0.00703355,                                                                                                                                                                
                -0.00270116,  0.0241398 ],                                                                                                                                                                                              
                [ 0.02588669,  0.00276013,  0.00086447, ...,  0.00280573,                                                                                                                                                                
                0.0006188 , -0.03371754],                                                                                                                                                                                              
                [ 0.03003237, -0.00248498,  0.01639194, ..., -0.04627804,                                                                                                                                                                
                0.01179953,  0.06639014]]], dtype=float32)} 
        """

        # del dataset_tensor_dict
        torch.cuda.empty_cache()
        return grad_dict, outputs
    
    
    def _get_word_positions(self, predic_tok, predic_tok_start, editor_toks):
        """ Helper function to map from (sub)tokens of Predictor to 
        token indices of Editor tokenizer. Assumes the tokens are in order.
        Raises MaskError if tokens cannot be mapped 
            This sometimes happens due to inconsistencies in way text is 
            tokenized by different tokenizers. """

        ###### ==>>predic_tok> singapore
        ###### ==>>>>>editor_toks [▁Hospital, ization, s, ▁decreased, ▁in, ▁Australia, ▁and, ▁Singapore, ▁but, ▁increased, ▁in, ▁Taiwan, ▁, ,, ▁Republic, ▁of, ▁China, ▁, ., </s>]


        return_word_idx = None
        # predic_tok_start = predic_tok.idx
        # predic_tok_end = predic_tok.idx_end
        predic_tok_start = predic_tok_start
        predic_tok_end = predic_tok_start + 1
   

        if predic_tok_start is None or predic_tok_end is None:
           return [], [], [] 
        
        class Found(Exception): pass
        try:
            for word_idx, word_token in reversed(list(enumerate(editor_toks))):
                if editor_toks[word_idx].idx is None:
                    continue
                    
                # Ensure predic_tok start >= start of last Editor tok
                if word_idx == len(editor_toks) - 1:
                    if predic_tok_start >= word_token.idx:
                        return_word_idx = word_idx
                        raise Found
                        
                # For all other Editor toks, ensure predic_tok start 
                # >= Editor tok start and < next Editor tok start
                elif predic_tok_start >= word_token.idx:
                    for cand_idx in range(word_idx + 1, len(editor_toks)):
                        if editor_toks[cand_idx].idx is None:
                            continue
                        elif predic_tok_start < editor_toks[cand_idx].idx:
                            return_word_idx = word_idx
                            raise Found
        except Found:
            pass 

        if return_word_idx is None:
            return [], [], []

        last_idx = return_word_idx
        if predic_tok_end > editor_toks[return_word_idx].idx_end:
            for next_idx in range(return_word_idx, len(editor_toks)):
                if editor_toks[next_idx].idx_end is None:
                    continue
                if predic_tok_end <= editor_toks[next_idx].idx_end:
                    last_idx = next_idx
                    break
 
            return_indices = []
            return_starts = []
            return_ends = []

            for cand_idx in range(return_word_idx, last_idx + 1):
                return_indices.append(cand_idx)
                return_starts.append(editor_toks[cand_idx].idx)
                return_ends.append(editor_toks[cand_idx].idx_end)
            if not predic_tok_start >= editor_toks[return_word_idx].idx:
                raise MaskError

            # Sometimes BERT tokenizers add extra tokens if spaces at end
            if last_idx != len(editor_toks)-1 and \
                    predic_tok_end > editor_toks[last_idx].idx_end:
                    raise MaskError

            return return_indices, return_starts, return_ends

        return_tuple = ([return_word_idx], 
                            [editor_toks[return_word_idx].idx], 
                            [editor_toks[return_word_idx].idx_end])
        return return_tuple
        
    # Copied from AllenNLP integrated gradient
    def _integrated_register_forward_hook(self, alpha, embeddings_list):
        """ Helper function for integrated gradients """

        def forward_hook(module, inputs, output):
            if alpha == 0:
                embeddings_list.append(
                        output.squeeze(0).clone().detach().cpu().numpy())

            output.mul_(alpha)

        # embedding_layer = util.find_embedding_layer(self.predictor._model)
        embedding_layer = util.find_embedding_layer(self.predictor.model)
        handle = embedding_layer.register_forward_hook(forward_hook)
        ###### embedding_layer, Embedding(31090, 768, padding_idx=0)
        ###### handle <torch.utils.hooks.RemovableHandle object at 0x7f0d32d8a410>
        return handle


    # def _register_embedding_list_hook(self, model, embeddings_list):
    #     def forward_hook(module, inputs, output):
    #         embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    #     embedding_layer = model.bert.embeddings.word_embeddings
    #     handle = embedding_layer.register_forward_hook(forward_hook)
    #     return handle


    # Copied from AllenNLP integrated gradient
    def _get_integrated_gradients(self, instance, pred_idx, steps):
        """ Helper function for integrated gradients """

        ig_grads: Dict[str, Any] = {}

        # List of Embedding inputs
        embeddings_list: List[np.ndarray] = []

        # Exclude the endpoint because we do a left point integral approx 
        for alpha in np.linspace(0, 1.0, num=steps, endpoint=False):
            # Hook for modifying embedding value
            handle = self._integrated_register_forward_hook(alpha, embeddings_list)    ###### handle = self._register_embedding_list_hook(self.predictor.model, embeddings_list)

            grads = self._get_gradients_by_prob(instance, pred_idx)[0]
            handle.remove()

            # Running sum of gradients
            if ig_grads == {}:
                ig_grads = grads
            else:
                for key in grads.keys():
                    ig_grads[key] += grads[key]

        # Average of each gradient term
        for key in ig_grads.keys():
            ig_grads[key] /= steps

        # Gradients come back in reverse order of order sent into the network
        embeddings_list.reverse()

        # Element-wise multiply average gradient by the input
        for idx, input_embedding in enumerate(embeddings_list):
            key = "grad_input_" + str(idx + 1)
            ig_grads[key] *= input_embedding

        return ig_grads    
        
        
    def get_important_editor_tokens(
            self, editable_seg, pred_idx, editor_toks, 
            labeled_instance=None, 
            predic_tok_start_idx=None, 
            predic_tok_end_idx=None, 
            num_return_toks=None,
            max_length = None):
        """ Gets Editor tokens that correspond to Predictor toks 
        with highest gradient values (with respect to pred_idx).

        editable_seg:
            Original inp to mask.
        pred_idx:
            Index of label (in Predictor label space) to take gradient of. 
        editor_toks: 
            Tokenized words using Editor tokenizer
        labeled_instance:
            Instance object for Predictor
        predic_tok_start_idx:
            Start index of Predictor tokens to consider masking. 
            Helpful for when we only want to mask part of the input, 
                as in RACE (only mask article). In this case, editable_seg 
                will contain a subinp of the original input, but the 
                labeled_instance used to get gradient values will correspond 
                to the whole original input, and so predic_tok_start_idx
                is used to line up gradient values with tokens of editable_seg.
        predic_tok_end_idx:
            End index of Predictor tokens to consider masking. 
            Similar to predic_tok_start_idx.
        num_return_toks: int
            If set to value k, return k Editor tokens that correspond to 
                Predictor tokens with highest gradients. 
            If not supplied, use self.mask_frac to calculate # tokens to return
        """

        integrated_grad_steps = self.num_integrated_grad_steps

        # max_length = self.predictor._dataset_reader._tokenizer._max_length
        max_length = max_length
        temp_tokenizer = self.predictor.tokenizer

        all_predic_dis = temp_tokenizer.encode(editable_seg)   ###### all_predic_dis [102, 13972, 30113, 2664, 121, 7266, 137, 15252, 563, 1175, 121, 11338, 422, 9789, 131, 3640, 205, 103]
        # all_predic_toks = temp_tokenizer.tokenize(editable_seg)   ###### all_predic_toks = ['hospitalization', '##s', 'decreased', 'in', 'australia', 'and', 'singapore', 'but', 'increased', 'in', 'taiwan', ',', 'republic', 'of', 'china', '.']
        all_predic_toks = temp_tokenizer.convert_ids_to_tokens(all_predic_dis)  ###### all_predic_toks = ['[CLS]', 'hospitalization', '##s', 'decreased', 'in', 'australia', 'and', 'singapore', 'but', 'increased', 'in', 'taiwan', ',', 'republic', 'of', 'china', '.', '[SEP]']
        ###### editor_toks = [▁Hospital, ization, s, ▁decreased, ▁in, ▁Australia, ▁and, ▁Singapore, ▁but, ▁increased, ▁in, ▁Taiwan, ▁, ,, ▁Republic, ▁of, ▁China, ▁, ., </s>]


        # # TODO: Does NOT work for RACE
        # # If labeled_instance is not supplied, create one
        # if labeled_instance is None:
        #     labeled_instance = self.predictor.json_to_labeled_instances(
        #             {"sentence": editable_seg})[0]

        grad_type_options = ["integrated_l1", "integrated_signed", "normal_l1",  "normal_signed", "normal_l2", "integrated_l2"]
        if self.grad_type not in grad_type_options:
            raise ValueError("Invalid value for grad_type")

        # Grad_magnitudes is used for sorting; highest values ordered first. 
        # -> For signed, to only mask most neg values, multiply by -1
        labeled_instance = editable_seg

        if self.grad_type == "integrated_l1":
            # grads = self._get_integrated_gradients(editable_seg, pred_idx, steps = integrated_grad_steps)
            # grad = grads["grad_input_1"][0]
            # grad_signed = np.sum(abs(grad), axis = 1) 
            # grad_magnitudes = grad_signed.copy()

            grads = self._get_integrated_gradients(
                    labeled_instance, pred_idx, steps = integrated_grad_steps)
            grad = grads["grad_input_1"][0]
            grad_signed = np.sum(abs(grad), axis = 1) 
            grad_magnitudes = grad_signed.copy()

        elif self.grad_type == "integrated_signed":
            grads = self._get_integrated_gradients(
                    labeled_instance, pred_idx, steps = integrated_grad_steps)
            grad = grads["grad_input_1"][0]
            grad_signed = np.sum(grad, axis = 1)
            grad_magnitudes = self.sign_direction * grad_signed
        
        elif self.grad_type == "integrated_l2":
            grads = self._get_integrated_gradients(
                    labeled_instance, pred_idx, steps = integrated_grad_steps)
            grad = grads["grad_input_1"][0]
            grad_signed = [g.dot(g) for g in grad]
            grad_magnitudes = grad_signed.copy()

        elif self.grad_type == "normal_l1":
            grads = self._get_gradients_by_prob(labeled_instance, pred_idx)[0]
            grad = grads["grad_input_1"][0]
            grad_signed = np.sum(abs(grad), axis = 1) 
            grad_magnitudes = grad_signed.copy()

        elif self.grad_type == "normal_signed":
            grads = self._get_gradients_by_prob(labeled_instance, pred_idx)[0]
            grad = grads["grad_input_1"][0]
            grad_signed = np.sum(grad, axis = 1)
            grad_magnitudes = self.sign_direction * grad_signed

        elif self.grad_type == "normal_l2":
            grads = self._get_gradients_by_prob(labeled_instance, pred_idx)[0]
            grad = grads["grad_input_1"][0]
            grad_signed = [g.dot(g) for g in grad]
            grad_magnitudes = grad_signed.copy()

        # Include only gradient values for editable parts of the inp
        if predic_tok_end_idx is not None:
            if predic_tok_start_idx is not None:
                grad_magnitudes = grad_magnitudes[
                        predic_tok_start_idx:predic_tok_end_idx]
                grad_signed = grad_signed[
                        predic_tok_start_idx:predic_tok_end_idx]
            else:
                grad_magnitudes = grad_magnitudes[:predic_tok_end_idx]
                grad_signed = grad_signed[:predic_tok_end_idx]
        

        # Order Predictor tokens from largest to smallest gradient values 
        ordered_predic_tok_indices = np.argsort(grad_magnitudes)[::-1]    ###### grad_magnitudes shape= (18,)
        """
        grad_magnitudes [0.35318244 1.7696457  0.9514374  1.3205127  0.66492414 1.3386885
                        0.6169287  1.7747654  1.1276003  1.1889602  0.742771   1.5146408
                        0.55210245 1.6286447  0.65258837 1.1351242  0.46439698 0.6163024 ]
        ordered_predic_tok_indices: [ 7  1 13 11  5  3  9 15  8  2 10  4 14  6 17 12 16  0]
        """

        # List of tuples of (start, end) positions in the original inp to mask
        ordered_word_indices_by_grad = [self._get_word_positions(
            all_predic_toks[idx], idx, editor_toks)[0] \
                    for idx in ordered_predic_tok_indices \
                    if all_predic_toks[idx] not in self.predictor_special_toks]    ###### all_predic_toks: 18   ### ordered_word_indices_by_grad: [[0], [0], [1], [1], [0], [0], [1], [2], [1], [0], [1], [0], [1], [0], [1], [2, 3]]

        ordered_word_indices_by_grad = [item for sublist in \
                ordered_word_indices_by_grad for item in sublist]   ###### ordered_word_indices_by_grad: [2, 1, 0, 1, 1, 0, 1, 0, 1, 1, 2]   

        # Sanity checks
        if predic_tok_end_idx is not None:
            if predic_tok_start_idx is not None:
                assert(len(grad_magnitudes) == \
                        predic_tok_end_idx - predic_tok_start_idx)
            else:
                assert(len(grad_magnitudes) == predic_tok_end_idx)
        elif max_length is not None and (len(grad_magnitudes)) >= max_length:
            assert(max_length == (len(grad_magnitudes)))
        else:
            assert(len(all_predic_toks) == (len(grad_magnitudes)))
        
        # Get num words to return
        if num_return_toks is None:
            num_return_toks = math.ceil(
                    self.mask_frac * len(ordered_word_indices_by_grad))
        highest_editor_tok_indices = []
        for idx in ordered_word_indices_by_grad:
            if idx not in highest_editor_tok_indices:
                highest_editor_tok_indices.append(idx)
                if len(highest_editor_tok_indices) == num_return_toks:
                    break
        
        highest_predic_tok_indices = ordered_predic_tok_indices[:num_return_toks]
        return highest_editor_tok_indices
    


    def _get_mask_indices(self, editable_seg, editor_toks, pred_idx, **kwargs):
        """ Helper function to get indices of Editor tokens to mask. """
        
        editor_mask_indices = self.get_important_editor_tokens(
                editable_seg, pred_idx, editor_toks, **kwargs)
        return editor_mask_indices 






# class RandomMasker(Masker):
#     """ Masks randomly chosen spans. """ 
    
#     def __init__(
#             self, 
#             mask_frac, 
#             editor_tok_wrapper, 
#             max_tokens
#         ):
#         super().__init__(mask_frac, editor_tok_wrapper, max_tokens)
   
#     def _get_mask_indices(self, editable_seg, editor_toks, pred_idx, **kwargs):
#         """ Helper function to get indices of Editor tokens to mask. """
        
#         num_tokens = min(self.max_tokens, len(editor_toks))

#         return random.sample(
#                 range(num_tokens), math.ceil(self.mask_frac * num_tokens))
    
