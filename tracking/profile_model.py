import argparse
import torch
from thop import profile
from thop.utils import clever_format
import time
import importlib
from torch import nn
import os


#current_cwd=os.getcwd()
current_cwd=os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


def parse_args():

    parser=argparse.ArgumentParser()

    parser.add_argument(
        '--script',
        type=str,
        default='hptrack'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='hptrack_b224_got'
    )

    return parser.parse_args()



# ===============================
# backbone wrapper
# ===============================

class BackboneWrapper(nn.Module):

    def __init__(self,network):
        super().__init__()
        self.network=network


    def forward(
            self,
            template_list,
            search_list,
            anno_list,
            his_search,
            his_hp
    ):

        back_out,_,_=self.network.forward_backbone(
            template_list,
            search_list,
            anno_list,
            his_search,
            his_hp
        )

        return back_out



# ===============================
# head wrapper
# ===============================

class HeadWrapper(nn.Module):

    def __init__(self,network):
        super().__init__()

        self.network=network


    def forward(self,back_out):

        out=self.network.forward_head(
            back_out=back_out
        )

        return out["score_map"]



def evaluate(
        model,
        template_list,
        search_list,
        anno_list,
        his_search,
        his_hp
):


    print("==========================")
    print("Calculating Backbone")
    print("==========================")


    backbone=BackboneWrapper(model)

    backbone.eval()


    macs1,params1=profile(
        backbone,
        inputs=(
            template_list,
            search_list,
            anno_list,
            his_search,
            his_hp
        ),
        verbose=False
    )


    print(
        "Backbone:",
        clever_format(
            [macs1,params1],
            "%.3f"
        )
    )



    # 得到back_out
    with torch.no_grad():

        back_out,_,_=model.forward_backbone(
            template_list,
            search_list,
            anno_list,
            his_search,
            his_hp
        )



    print("==========================")
    print("Calculating Head")
    print("==========================")


    head=HeadWrapper(model)

    head.eval()


    macs2,params2=profile(
        head,
        inputs=(back_out,),
        verbose=False
    )


    print(
        "Head:",
        clever_format(
            [macs2,params2],
            "%.3f"
        )
    )



    total_macs=macs1+macs2
    total_params=params1+params2


    print("==========================")
    print("Total")
    print("==========================")


    print(
        "Overall:",
        clever_format(
            [total_macs,total_params],
            "%.3f"
        )
    )





    # =====================
    # speed
    # =====================

    print("testing speed...")


    warmup=50
    test=200


    with torch.no_grad():


        for i in range(warmup):

            back_out,_,_=model.forward_backbone(
                template_list,
                search_list,
                anno_list,
                his_search,
                his_hp
            )

            _=model.forward_head(
                back_out
            )


        torch.cuda.synchronize()


        start=time.time()


        for i in range(test):

            back_out,_,_=model.forward_backbone(
                template_list,
                search_list,
                anno_list,
                his_search,
                his_hp
            )

            _=model.forward_head(
                back_out
            )


        torch.cuda.synchronize()

        end=time.time()


    latency=(end-start)/test*1000


    print(
        "Latency %.2f ms"%(latency)
    )

    print(
        "FPS %.2f"%(1000/latency)
    )





def get_data(bs,size):

    return torch.randn(
        bs,
        3,
        size,
        size
    )




if __name__=="__main__":


    device="cuda:0"

    torch.cuda.set_device(device)



    args=parse_args()



    yaml_fname=current_cwd + \
        "/experiments/%s/%s.yaml"%(args.script,args.config)



    config_module=importlib.import_module(
        "lib.config.%s.config"%args.script
    )


    cfg=config_module.cfg


    config_module.update_config_from_file(
        yaml_fname
    )



    model_module=importlib.import_module(
        "lib.models.hptrack"
    )


    model=model_module.build_hptrack(
        cfg
    )



    model=model.to(device)

    model.eval()



    bs=1


    template_size=cfg.TEST.TEMPLATE_SIZE
    search_size=cfg.TEST.SEARCH_SIZE


    template=get_data(
        bs,
        template_size
    ).to(device)



    search=get_data(
        bs,
        search_size
    ).to(device)



    num_template=cfg.TEST.NUM_TEMPLATES


    template_list=[
        template
        for _ in range(num_template)
    ]


    search_list=[
        search
    ]



    anno_list=[
        torch.tensor(
            [[0.5,0.5,0.5,0.5]]
        ).float().to(device)
        for _ in range(num_template)
    ]



    num_layers=cfg.MODEL.BACKBONE.NUM_LAYERS


    his_search=[
        None
    ]*num_layers


    his_hp=[
        None
    ]*num_layers



    evaluate(
        model,
        template_list,
        search_list,
        anno_list,
        his_search,
        his_hp
    )