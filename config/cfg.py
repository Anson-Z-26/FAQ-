import os

path= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gcfg= {
    fs: { 
        "path":path,
        "pswd":"6f1d27162e086449c3ff744702076a96ba8d3a215d6d4317784937155cff7dad",
        "logo":"",
        "slogan":"让知识流动起来",
        "aicode":"",
    },
    llm: {
        "ctx_size": 20480,
        "temperature": 0.6,
        "top_p":0.95,
        "top_k": 50,
        embed_api:"",
        embed_key:"",
        rerank_api:"",
        rerank_size:4096,
        vector_size:1024
    },
    full_index: {
        ignore_fsize: 100,
        ignore_ftype:".cdr,.svg,.ttf,.js,.css,.eot,.map,.scss,.png,.mp4,.iso,.png,jpg,.jpeg,.gif,.so,.dll,.exe,.pyc,.qcow2,.gz,.pcap,.bin,.db",
        chunk_size: 4096,
        chunk_overlap: 100,
    },
    dingtalk:{
        app_key:"",
        app_secret:"",
        exclusiveLogin:"false",
        exclusiveCorpId:"",
    },
    wechat:{
        app_key:"",
        app_secret:"",
    },
    mem:{
        agent_catalogs:{
            "知识库": "orange",
            "知识加工":"blue",
            "文档生成":"green",
            "内容审核":"red",
        },
    },
    ocr:{
        "paddleocr_addr":"",
        "max_size":5242880,
        "min_size":4096
    }
}
