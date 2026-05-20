import sys,os  
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'backend'))  
os.chdir(os.path.dirname(os.path.abspath(__file__)))  
from app.rag.pipeline import get_rag  
rag = get_rag()  
count = rag.build_knowledge_base(reset=True)  
print(f'SUCCESS: {count} records') 
