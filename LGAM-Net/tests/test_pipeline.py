"""Offline integration tests, using tiny random encoders and generated images."""
import json
import tempfile
import unittest
from pathlib import Path
import torch
from PIL import Image
from data.collate import BatchCollator
from data.records import read_records, label_of
from data.weak_supervision import build_entity_pool, make_weak_triple
from models.lgam_net import LGAMNet
from losses.objective import objective
from engine.trainer import train
from engine.evaluator import evaluate

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, root):
        return {'seed':42,'device':'cpu','data':{'root':str(root),'train_json':str(root/'train.json'),
                'val_json':str(root/'val.json')},'model':{'encoder':'tiny','dim':32,'views':4},
                'train':{'epochs':1,'batch_size':4,'accumulation':2,'workers':0,'lr':.001,'amp':False,
                         'output':str(root/'run')},'loss':{'alignment':.1,'consistency':.05,'temperature':.07}}

    def fixture(self, root):
        def record(i):
            Image.new('RGB',(40,32),(i*30,80,120)).save(root/f'{i}.jpg')
            return {'img_local_path':f'{i}.jpg','articles':[
                {'caption':f'Person{i} visits City{i}.','entity_list':[[f'Person{i}','PERSON'],[f'City{i}','GPE']]},
                {'caption':f'A portrait of Person{i}.','entity_list':[[f'Person{i}','PERSON']]}]}
        tr = [record(i) for i in range(3)]
        va = [record(3),record(4)]
        (root/'train.json').write_text(json.dumps(tr))
        (root/'val.json').write_text('\n'.join(map(json.dumps,va)))
        te = [{'img_local_path':f'{i}.jpg','caption1':'A person visits a city.',
               'caption2':'A portrait of a person.','context_label':i%2} for i in range(2)]
        (root/'test.json').write_text(json.dumps(te))
        return tr

    def test_training_checkpoint_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            config=self.config(root)
            ckpt=train(config)
            result=evaluate(ckpt,root/'test.json',root,root/'results','cpu',batch_size=2)
            self.assertEqual(result['n'],2)
            self.assertEqual(sum(map(sum,result['confusion_matrix'])),2)
            self.assertTrue((root/'results'/'predictions.csv').exists())
            # Resume at an epoch boundary; same configured run already finished.
            train(config,root/'run'/'last.pt')

    def test_symmetry_backward_and_masks(self):
        config={'encoder':'tiny','dim':32,'views':4}
        rows=[dict(image=Image.new('RGB',(32,32)),caption1='A violin player',caption2='A musician on stage',
                   label=i,weight=1,group=i,sample_id=str(i),origin='fixture') for i in (0,1)]
        batch=BatchCollator(config)(rows)
        model=LGAMNet(config).eval()
        out=model(batch)
        swapped=dict(batch)
        for base in ['ids','mask','local_mask']:
            swapped[base+'1'],swapped[base+'2']=batch[base+'2'],batch[base+'1']
        torch.testing.assert_close(out['logits'],model(swapped)['logits'],atol=1e-6,rtol=1e-5)
        loss,_=objective(out,batch,{'alignment':.1})
        loss.backward()
        self.assertTrue(torch.isfinite(model.enhance.gamma.grad))
        self.assertIsNotNone(model.head[0].weight.grad)
        for ra,lfe in [(True,False),(False,True),(False,False)]:
            m=LGAMNet(dict(config,use_ra=ra,use_lfe=lfe))
            self.assertTrue(torch.isfinite(m(batch)['logits']).all())

    def test_pseudo_labels_and_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); records=self.fixture(root)
            pools=build_entity_pool(records)
            a,b,y,w,origin=make_weak_triple(records[0],pools,42,True)
            self.assertEqual(y,1); self.assertGreater(w,0)
            self.assertTrue(origin.startswith('entity_swap:'))
            self.assertEqual(len(read_records(root/'val.json')),2)
            self.assertEqual(label_of({'label':'not-ooc'}),0)
            self.assertEqual(label_of({'context_label':1}),1)

    def test_real_clip_interface_offline(self):
        from transformers import CLIPConfig, CLIPModel
        from models.encoders import CLIPEncoder
        with tempfile.TemporaryDirectory() as tmp:
            config=CLIPConfig(text_config={'vocab_size':100,'hidden_size':32,'intermediate_size':64,
                'num_hidden_layers':1,'num_attention_heads':4,'max_position_embeddings':16,'eos_token_id':2,'bos_token_id':1,'pad_token_id':0},
                vision_config={'hidden_size':32,'intermediate_size':64,'num_hidden_layers':1,
                'num_attention_heads':4,'image_size':32,'patch_size':8},projection_dim=16)
            CLIPModel(config).save_pretrained(tmp)
            encoder=CLIPEncoder(tmp,True,True)
            ids=torch.tensor([[1,7,2,0],[1,8,2,0]])
            p,ig,t,tg,grid=encoder(torch.randn(1,3,32,32),ids,ids.ne(0).long())
            self.assertEqual(p.shape,(1,16,32)); self.assertEqual(tg.shape,(2,16))
            self.assertEqual(grid,(4,4)); self.assertFalse(ig.requires_grad)

if __name__=='__main__':
    unittest.main()
