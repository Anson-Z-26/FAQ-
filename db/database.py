from peewee import SqliteDatabase, Model, CharField, TextField, ForeignKeyField, DateTimeField,IntegerField,AutoField
import datetime
import time

from init import *

# 创建一个 SQLite 数据库实例
db = SqliteDatabase(f"{gcfg['fs']['path']}/db/k_database.db")

# 连接到数据库
db.connect()

#并发操作
db.execute_sql("PRAGMA journal_mode=WAL;")


class BaseModel(Model):
    class Meta:
        database = db

#知识库文档
class Doc(BaseModel):
    base = CharField() #知识库
    abs_path = CharField() #绝对路径
    f_name = CharField()
    f_up_at = DateTimeField(default=datetime.datetime.now) #文件更新时间
    f_size = IntegerField(default=0) #文件大小
    f_sec = IntegerField(default=1) #点赞数
    author= CharField(null = True) #作者
    ctype = CharField(null = True) #Content-Type
    catalog = CharField(null = True)
    level = IntegerField(default=1)
    meta = TextField(null = True)
    m_time = DateTimeField(default=datetime.datetime.now) #扫描时间

#知识百科
class BaiKe(BaseModel):
    id = AutoField(primary_key=True)
    title = CharField()
    catalog = CharField()
    html = TextField()
    c_time = DateTimeField(default=datetime.datetime.now)
    m_time = DateTimeField(default=datetime.datetime.now)
    #浏览次数
    view_count = IntegerField(default=0)
    #全文id
    full_id = IntegerField(default=0)
    creator = TextField(default="管理员")
    modifier = TextField(default="管理员")


#知识百科的分类
class BaiKe_Catalog(BaseModel):
    id = AutoField(primary_key=True)
    catalog = CharField()
    demo = CharField()
    c_time = DateTimeField(default=datetime.datetime.now)
    #百科条数
    doc_count = IntegerField(default=0)

#知识广场的知识体的仓库，助手，智能体
class KAgent(BaseModel):
    id = AutoField(primary_key=True)
    title = CharField() #标题
    demo = CharField() #介绍
    guide = CharField() #引导语
    kasn = CharField() #生成一个随机数的,用来标识KAgent
    atype = IntegerField(default=0) #agent 类型,0-检索知识库,1-动态检索,2-上传文件，100-不检索
    files = TextField(null = True) #文件列表
    prompt = TextField() #提示词
    c_time = DateTimeField(default=datetime.datetime.now)
    m_time = DateTimeField(default=datetime.datetime.now)
    icon = CharField(null = True) # 图标
    catalog = CharField(null = True) #分类
    baike = CharField(null = True) #百科
    oth = TextField(null = True) #未来
    creator = TextField(default="管理员")
    modifier = TextField(default="管理员")

#用户列表
class User(BaseModel):
    userid = TextField() #自动生成，可保唯一,
    username = TextField() #显示名
    telphone = TextField() #手机号码，需要唯一
    email = CharField(null = True)
    city = CharField(null = True) #个人的订阅的智能体
    org = CharField(null = True)
    dep = CharField(null = True)   #个人的服务能力设置，如访问次数等
    title = CharField(null = True) #查找键：OAuth通道为手机号的hash(hashed_tel)，账户体系为登录名的hash(hashed_account)
    password = CharField(null = True) #账户密码 hashed_password(明文)；为空表示未设置密码(如OAuth通道创建的账号)
    ref = CharField(null = True) #注册方式
    c_time = DateTimeField(default=datetime.datetime.now)


def _ensure_column(table, column, ddl):
    """
    幂等的“加列”迁移。
    表结构只在 __main__ 的 create_tables 里创建，应用启动时不会建表（靠init.py拷贝自带库），
    peewee 也不会给已存在的表加列，所以新列必须在模块导入时补齐。
    gunicorn 多worker会并发导入本模块，此处需容忍竞态。
    """
    def cols_of():
        return [c[1] for c in db.execute_sql(f"PRAGMA table_info({table});").fetchall()]

    try:
        cols = cols_of()
    except Exception as e:
        print("读取表结构失败", table, e)
        return
    if not cols:
        #新库还没有表，交给 create_tables 直接建成带新列的表
        print(f"迁移跳过: {table} 表不存在")
        return
    if column in cols:
        return

    for _ in range(3):
        try:
            db.execute_sql(f"ALTER TABLE {table} ADD COLUMN {ddl};")
            print(f"迁移完成: {table}.{column}")
            return
        except Exception as e:
            #并发下后到的worker会撞 duplicate column name，或撞 database is locked
            print(f"迁移重试 {table}.{column}: {e}")
            time.sleep(0.5)

    if column not in cols_of():
        print(f"警告：{table}.{column} 迁移失败，登录/注册将不可用")


_ensure_column(User._meta.table_name, "password", "password VARCHAR(255) NULL")


# 创建一个 SQLite 数据库实例
log_db = SqliteDatabase(f"{gcfg['fs']['path']}/db/logs.db")

# 连接到数据库
log_db.connect()

#并发操作
log_db.execute_sql("PRAGMA journal_mode=WAL;")


class LogBaseModel(Model):
    class Meta:
        database = log_db


#操作日志
class Login(LogBaseModel):
    userid = TextField(null = True)
    username = TextField(null = True)
    action = TextField(null = True) #登录，退出，新增删除等操作
    pt  =TextField(null = True) #访问的平台或知识体的sn,或内容
    ref = CharField(null = True) #登录方式
    client_ip = CharField(null = True)
    c_time = DateTimeField(default=datetime.datetime.now)

#知识对话日志
class QA(LogBaseModel):
    userid = TextField(null = True)
    username = TextField(null = True)
    question = TextField()
    answer = TextField()
    ref = CharField(null = True)
    client_ip = CharField(null = True)
    c_time = DateTimeField(default=datetime.datetime.now)

#超级个人助理agent日志
class SuperManAgent(LogBaseModel):
    chatid = TextField(null = True)
    username = TextField(null = True)
    question = TextField()
    answer = TextField() #规划的路径
    prompt = TextField() #可用的prompt
    score = IntegerField(default=60) #自动评分
    human = IntegerField(default=0) #手工点赞100或点踩-100
    c_time = DateTimeField(default=datetime.datetime.now)


#视觉识别的文档中的图片
class VL_img(LogBaseModel):
    doc_base = TextField()
    doc_path = TextField()
    img_path = TextField(null = True) #图片的路径默认在/mnt/vl_img下
    img_name = TextField(null = True)
    img_type = TextField(null = True) #
    img_md = TextField(null = True)
    img_size = TextField(null = True) #width x height
    c_time = DateTimeField(default=datetime.datetime.now)

#自动创建
log_db.create_tables([QA,Login,SuperManAgent,VL_img])


def init_data():
    baike = BaiKe.create(title="k3GPT大模型是什么",catalog="词条/产品词条",html="""<p>k3GPT是一个大模型AI驱动的知识共享平台，通过知识搜索、知识对话、知识创作、知识广场等功能，实现知识从汇聚沉淀、加工创作，到流动共享的全过程管理，因此称为知识knowledge x 3的GPT平台, 名曰: k3GPT. <br>  
            让私域知识文档轻松的聚合沉淀，让人更好的使用和创造知识，让知识流动起来！让工作高效起来！</p>""")
    #baike = BaiKe.get_by_id(1)
    #print(baike.title)

    c1 = BaiKe_Catalog.create(catalog="词条",demo="对特定词汇、短语或主题进行解释和定义的条目，如人物、产品等")
    c1 = BaiKe_Catalog.create(catalog="词条/产品词条",demo="关于产品的词条,包括名称、功能、开份计划等")
    c1 = BaiKe_Catalog.create(catalog="词条/人物词条",demo="关于人物的词条，包括简历，职务等")
    c1 = BaiKe_Catalog.create(catalog="词条/管理词条",demo="关于管理制度的词条,包括定义，解释等")
    c1 = BaiKe_Catalog.create(catalog="词条/活动词条",demo="关于活动的词条,包括时间,主题,人员等")
    c1 = BaiKe_Catalog.create(catalog="词条/技术词条",demo="关于一项技术的描述和讲解")
    c1 = BaiKe_Catalog.create(catalog="词条/标准规范",demo="关于标准规范类的词条")
    c1 = BaiKe_Catalog.create(catalog="大模型/提示词模板",demo="应用和agent使用的提示词的模板")
    c1 = BaiKe_Catalog.create(catalog="大模型/微调数据",demo="可以用于大模型回答结果进行微调的数据")
    c1 = BaiKe_Catalog.create(catalog="工作汇报",demo="关于工作中的月报、周报、日报等")
    c1 = BaiKe_Catalog.create(catalog="参考资料",demo="工作中所用的到参考资料")
    c1 = BaiKe_Catalog.create(catalog="会议纪要",demo="各种会议的纪要、备忘")
    c1 = BaiKe_Catalog.create(catalog="个人空间/管理员",demo="存放管理员的个人数据")
    c1 = BaiKe_Catalog.create(catalog="海报生成",demo="可直接用于生成海报的内容")
    c1 = BaiKe_Catalog.create(catalog="PPT生成",demo="可直接用于生成PPT的内容")

    #初始化知识体
    import secrets
    kasn= secrets.token_urlsafe(8)
    k1 = KAgent.create(atype=1,title="企业文化",demo="企业介绍、发展历程、愿景目标、价值观等内容",guide="你好，我是企业文化知识助手，可以回答你关于企业介绍、历程、愿景、价值观等方面的问题。",kasn=kasn,prompt="""
    你是一个企业文化知识助手，你可以根据提供的上下文信息来回答用户的问题。
    先判断一下文件名称和问题是否相关，再考虑要不要使用这段上下文。
    同时上下文中排在前面的可信度越高，越应该优先采纳。
    如果提供的上下文不足以回答问题时，你可以回答'对不起，暂时没有找到相关的信息，无法回答你的问题'
    """)

    kasn= secrets.token_urlsafe(8)
    k2 = KAgent.create(title="规章制度",demo="企业的考勤、绩效、福利等制度",guide="你好，我是企业文化知识助手，可以回答你关于企业介绍、历程、愿景、价值观等方面的问题。",kasn=kasn,prompt="""
    你是一个企业文化知识助手，你可以根据提供的上下文信息来回答用户的问题。
    先判断一下文件名称和问题是否相关，再考虑要不要使用这段上下文。
    同时上下文中排在前面的可信度越高，越应该优先采纳。
    如果提供的上下文不足以回答问题时，你可以回答'对不起，暂时没有找到相关的信息，无法回答你的问题'
    """)

    kasn= secrets.token_urlsafe(8)
    k3 = KAgent.create(atype=1,title="产品手册",demo="公司所有的产品的手册，包括产品PPT，技术白皮书，规格说明书等",guide="你好，我是企业文化知识助手，可以回答你关于企业介绍、历程、愿景、价值观等方面的问题。",kasn=kasn,prompt="""
    你是一个企业文化知识助手，你可以根据提供的上下文信息来回答用户的问题。
    先判断一下文件名称和问题是否相关，再考虑要不要使用这段上下文。
    同时上下文中排在前面的可信度越高，越应该优先采纳。
    如果提供的上下文不足以回答问题时，你可以回答'对不起，暂时没有找到相关的信息，无法回答你的问题'
    """)

    kasn= secrets.token_urlsafe(8)
    k3 = KAgent.create(atype=2,title="合同审核",files="PPT助手",
                       demo="审核合同的交付过程和潜在的法律风险",
                       guide="你好，我是企业文化知识助手，可以回答你关于企业介绍、历程、愿景、价值观等方面的问题。",
                       kasn=kasn,
                       prompt="""
# 角色：
你是一位专业的律师，执业领域是审查合同
# 动作：
- 你可以根据提供的上下文信息来回答用户的问题。
- 先判断一下文件名称和问题是否相关，再考虑要不要使用这段上下文。                同时上下文中排在前面的可信度越高，越应该优先采纳
# 任务：
你要以审查主体的利益最大化为原则，对合同上下文进行审查和分析，并提供严格的评分。具体目标:
1.对提交的合同文本进行全面审查，指出存在的问题和风险；
2.提供具体的改进建议，帮助改进和完善合同条款;
3.根据法律规范和个人建议，修改并完善合同的具体条款；
4.向客户提供专业的法律服务和支持。
# 约束条件:
1.必须遵守现行有效的法律法规，不能引用已废止的法律条文；
2.所有合同条款的设计应当符合最新的法律法规及相关政策的规定，要使用准确无误的专业术语和地名；
3.结合特定行业的特点(如XXX行业)，确保合同内容贴合实际情况；
4,考虏并反映审查主体的需求，始终站在其角度满足利益最大化。
    """)

    kasn= secrets.token_urlsafe(8)
    k3 = KAgent.create(title="商机管理",demo="产品、项目商机信息的汇总、识别、挖掘等",guide="你好，我是企业文化知识助手，可以回答你关于企业介绍、历程、愿景、价值观等方面的问题。",kasn=kasn,prompt="""
    你是一个企业文化知识助手，你可以根据提供的上下文信息来回答用户的问题。
    先判断一下文件名称和问题是否相关，再考虑要不要使用这段上下文。
    同时上下文中排在前面的可信度越高，越应该优先采纳。
    如果提供的上下文不足以回答问题时，你可以回答'对不起，暂时没有找到相关的信息，无法回答你的问题'
    """)

    kasn= secrets.token_urlsafe(8)
    k3 = KAgent.create(title="人事管理",demo="员工档案信息的管理、薪酬制度、激励政策等",guide="你好，我是企业文化知识助手，可以回答你关于企业介绍、历程、愿景、价值观等方面的问题。",kasn=kasn,prompt="""
    你是一个企业文化知识助手，你可以根据提供的上下文信息来回答用户的问题。
    先判断一下文件名称和问题是否相关，再考虑要不要使用这段上下文。
    同时上下文中排在前面的可信度越高，越应该优先采纳。
    如果提供的上下文不足以回答问题时，你可以回答'对不起，暂时没有找到相关的信息，无法回答你的问题'
    """)

    kasn= secrets.token_urlsafe(8)
    k4 = KAgent.create(atype=2,title="PPT助手",demo="支持多轮对话和生成PPT",
                       guide="你好，我是一个PPT助手，我可以回答问题，也可以输出成一个PPT。请问我有什么可以帮助你的？",
                       kasn=kasn,
                       files="PPT助手",
                       prompt="""
# 你是一个知识助手
## 现在是{now}，可以根据此作出时间上的判断
## 你可以根据提供的上下文信息和历史问答信息来回答用户的问题。
- 同时上下文中排在前面的可信度越高，越应该优先采纳。
- 如果提供的上下文不足以回答问题时，你可以回答
     "对不起，我无法回答改问题"
- 如果用户问题表示对上次回答不满意时，就不要参考历史回答信息，只立即历史用户提问就好
## 生成PPT 
  如果用户需要生成一个PPT，请按如下的格式进行生成，直接生成不要解释
 - PPT的主标题使用<h1>标记(必须)，如:<h1>PPT的主标题</h1>
 - PPT的副标题使用<h3>标记(必须)，如:<h3>PPT的副标题</h3>
 - PPT的子章节使用<h2>标记，相同子章节的单页使用相同的<h2>标记和不同的<h3>副标做区分，相同子章节的<h2>标记可以输出一词，后续的<h3>标记自动继承前面的<h2>标记
 - 单页的主标题使用h2标记(必须)，如:<h2>生产问题</h2><h2>第一行动计划</h2>
 - 单页的副标题使用h3标记(可选)，如：<h3>行动计划概述</h3>
 - 单页的内容按如下层次展开，一级<h5>标记，二级<li>标记，三级和普通内容<p>标记
 - 注意<li>中不能包含<h5>标记，因为<h5>的级别比<li>高
 - 单页的内容较多时，级数多于8项时，可以拆分成多页的PPT。使用不同的<h3>副标题编排内容即可。
 - 单页中可以使用<blockquote>标记来表示引用内容或重要内容
 - 记住：不用生成PPT的目录，智能体会根据<h2>标记自动生成成目录

    """)

    kasn= secrets.token_urlsafe(8)
    k4 = KAgent.create(atype=2,title="海报文案助手",demo="支持海报文案的生成",
                       guide="你好！我是一个海报生成助手，我可以根据你提供的文字和要求来生成一个海报的文案。",
                       kasn=kasn,
                       files="海报文案助手",
                       prompt="""
# 你是一个海报文案生成助手
## 现在是{now}，可以根据此作出时间上的判断 
## 海报文案结构标记
  海报文案结构的具体标记，请按如下的格式进行生成
 - 海报的主标题使用h1标记(必须)，如:<h1>海报的主标题</h1>
 - 海报的副标题使用h3标记(必须)，如:<h3>海报的副标题</h3>
 - 海报内容是一个列表，可以有多个内容块组成
 - 每个内容块有一个主标题和一个副标题
 - 海报内容块的主标题使用h2标记(必须)，如:<h2>行动计划</h2>
 - 海报内容块的副标题使用h3标记(可选)，如：<h3>执行到底</h3>
 - 内容块信息列表至多分为三级,一级用<h5>标记，二级用<li>标记,三级用<p>标记
 - 海报内容的摘要用<h4>标记
 - 海报的报尾需要使用<hr>表示，注脚内容用<h5>标记支持多个

##  海报文案样例
 <h1>海报主标题</h1>
 <h3>海报副标题</h1>
 <h2>海报内容1主标题</h2>
 <h3>海报内容1副标题</h3>
 <h5>一级内容</h5>
 <li>二级内容1</li>
 <p>三级级内容1<p>
 <h2>海报内容2主标题</h2>
 <h3>海报内容2副标题</h3>
 <h4>海报摘要</h4>
 <hr>
 <h5>海报报尾</h5>
    """)

    kasn= secrets.token_urlsafe(8)
    k5 = KAgent.create(title="智能翻译",demo="自动实现中文到英文，英文到中文的的翻译",guide="你好，我是中英文语言翻译的专家，直接输入中文则翻译输出英文，输入英文则翻译输出中文",atype=100,kasn=kasn,prompt="""
    你是一个中英文语言翻译的专家，直接输入中文则翻译输出英文，输入英文则翻译输出中文
    """)

if __name__ == '__main__':
   
    print(f"{gcfg['fs']['path']}/db/k_database.db")

    #创建表
    #db.drop_tables([Doc,BaiKe,BaiKe_Catalog,KAgent,User])

    db.drop_tables([Doc])

    db.create_tables([Doc,BaiKe,BaiKe_Catalog,KAgent,User])

    init_data()

    db.close()

    #log_db.close()