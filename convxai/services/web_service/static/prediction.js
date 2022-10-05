var BlockEmbed = Quill.import('blots/block/embed');
let Inline = Quill.import('blots/inline');




// class Predict extends Inline {
//     static create(props) {
//         let node = super.create();
//         node.setAttribute('id', props.id);
//         node.setAttribute('type', "predict-"+props.type);
//         return node;
//     }

    
//     static formats(node) {
//         return {
//             id: node.getAttribute("id"),
//             type: node.getAttribute("type"),
//             // style="display:none"
//         }
//     }
//     }
//     Predict.blotName = 'predict';
//     Predict.tagName = 'span';
//     Predict.className = 'predict';
                    
// Quill.register(Predict);







class PredictDiversity extends Inline {
    static create(props) {
        let node = super.create();
        node.setAttribute('id', props.id);
        node.setAttribute('type', "predict-"+props.type);
        node.setAttribute('category', "predicts");
        node.setAttribute('data_tooltip', props.data_tooltip);
        return node;
    }

    
    static formats(node) {
        return {
            id: node.getAttribute("id"),
            type: node.getAttribute("type"),
        }
    }
}
        PredictDiversity.blotName = 'predict-diversity';
        // PredictDiversity.tagName = 'span';
        PredictDiversity.tagName = 'p';
        // PredictDiversity.className = 'predicts predict-model-writing-1';
        PredictDiversity.className = 'predict-model-writing-1';
Quill.register(PredictDiversity);




// class PredictQuality extends Inline {
//     static create(props) {
//         let node = super.create();
//         node.setAttribute('id', props.id);
//         node.setAttribute('type', "predict-"+props.type);
//         node.setAttribute('style', "display:none")
//         node.setAttribute('category', "predicts")
//         return node;
//     }

    
//     static formats(node) {
//         return {
//             id: node.getAttribute("id"),
//             type: node.getAttribute("type"),
//         }
//     }
// }
//         PredictQuality.blotName = 'predict-quality';
//         PredictQuality.tagName = 'span';
//         // PredictQuality.className = 'predicts predict-model-writing-2';
//         PredictQuality.className = 'predict-model-writing-2';          
// Quill.register(PredictQuality);



