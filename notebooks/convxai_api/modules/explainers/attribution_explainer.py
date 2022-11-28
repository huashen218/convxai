import torch
import numpy as np
from torch import backends
from allennlp.nn import util

from convxai.writing_models.models import *
from convxai.writing_models.dataloaders import *


class AttributionExplainer(object):
    
    def __init__(
            self, 
            diversity_model, 
            max_tokens=512, 
            grad_type = "integrated_l2", 
            sign_direction = None,
            num_integrated_grad_steps = 20
    ):
        super().__init__()
        self.predictor  = diversity_model
        self.grad_type = grad_type
        self.num_integrated_grad_steps = num_integrated_grad_steps
        self.sign_direction = sign_direction
        self.max_tokens = max_tokens

        if ("signed" in self.grad_type and sign_direction is None):
            error_msg = "To calculate a signed gradient value, need to " + \
                    "specify sign direction but got None for sign_direction"
            raise ValueError(error_msg)

        if sign_direction not in [1, -1, None]:
            error_msg = f"Invalid value for sign_direction: {sign_direction}"
            raise ValueError(error_msg)

        self.temp_tokenizer = self.predictor.tokenizer
        self.predictor_special_toks = self.temp_tokenizer.all_special_tokens




    def batch_dataloader(self, input_string):
        feature = Feature(tokenizer=self.temp_tokenizer, pad_length = None)
        x_text = feature.extract(input_string[:])
        dataset = PredDataset(x_text)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        return dataloader



    def _get_word_positions(self, predic_tok, predic_tok_start, editor_toks):
        """ Helper function to map from (sub)tokens of Predictor to 
        token indices of Editor tokenizer. Assumes the tokens are in order.
        Raises MaskError if tokens cannot be mapped 
            This sometimes happens due to inconsistencies in way text is 
            tokenized by different tokenizers. """
        return_word_idx = None
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
        for param_name, param in self.predictor.model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = \
                    param.requires_grad
            param.requires_grad = True

        embeddings_gradients: List[Tensor] = []
        hooks: List[RemovableHandle] = self._register_embedding_gradient_hooks(self.predictor.model, embeddings_gradients)

        ###### instance = ["Hospitalizations decreased in Australia and Singapore but increased in Taiwan , Republic of China ."]
        dataloader = self.batch_dataloader([instance])
        x_batch = next(iter(dataloader)).to(device)

        softmax = nn.Softmax(dim=1)

        with backends.cudnn.flags(enabled=False):
            y_batch = torch.tensor([1]*x_batch.size(0)).to(device)
            outputs = self.predictor.model(x_batch, labels=y_batch)
            loss, y_pred = outputs[0:2]

            pred_idx = torch.argmax(y_pred, dim=1) if pred_idx is None else pred_idx
            prob = outputs[1][0][int(pred_idx)]

            self.predictor.model.zero_grad()
            prob.backward()

        hooks.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embeddings_gradients):
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        # Restore original requires_grad values of the parameters
        for param_name, param in self.predictor.model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

        del x_batch, y_batch
        torch.cuda.empty_cache()
        return grad_dict, outputs


    # Copied from AllenNLP integrated gradient
    def _integrated_register_forward_hook(self, alpha, embeddings_list):
        """ Helper function for integrated gradients """

        def forward_hook(module, inputs, output):
            if alpha == 0:
                embeddings_list.append(
                        output.squeeze(0).clone().detach().cpu().numpy())

            output.mul_(alpha)

        embedding_layer = util.find_embedding_layer(self.predictor.model)
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle



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
            self, editable_seg, pred_idx=None, editor_toks=None, 
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
        all_predic_dis = temp_tokenizer.encode(editable_seg)
        all_predic_toks = temp_tokenizer.convert_ids_to_tokens(all_predic_dis)

        grad_type_options = ["integrated_l1", "integrated_signed", "normal_l1", 
                "normal_signed", "normal_l2", "integrated_l2"]
        if self.grad_type not in grad_type_options:
            raise ValueError("Invalid value for grad_type")


        # Grad_magnitudes is used for sorting; highest values ordered first. 
        # -> For signed, to only mask most neg values, multiply by -1

        labeled_instance = editable_seg
        if self.grad_type == "integrated_l1":
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

        ### remove the [CLS] and [SEP] token.
        grad_magnitudes = grad_magnitudes[1:-1]
        all_predic_toks = all_predic_toks[1:-1]

        # Order Predictor tokens from largest to smallest gradient values 
        ordered_predic_tok_indices = np.argsort(grad_magnitudes)[::-1]    ###### grad_magnitudes shape= (18,)
        
        assert (len(all_predic_toks) == (len(ordered_predic_tok_indices)))
        assert(len(all_predic_toks) == (len(grad_magnitudes)))

        return all_predic_toks, ordered_predic_tok_indices



    def get_sorted_important_tokens(self, input, pred_idx):
        all_predic_toks, ordered_predic_tok_indices = self.get_important_editor_tokens(input, pred_idx)
        return  all_predic_toks, ordered_predic_tok_indices
