from typing import List, Optional, Dict
from pydantic import BaseModel
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI

class FinancialMetrics(BaseModel):
    # Original metrics
    last_12m_revenue: Optional[float]
    revenue_growth: Optional[float]
    burn_rate: Optional[float]
    existing_debt: Optional[float]
    total_raised: Optional[float]
    last_raised: Optional[float]
    ebitda_margin: Optional[float]
    
    # New financial metrics
    total_funding: Optional[float]
    latest_round: Optional[str]
    cash: Optional[float]
    ltm_gross_margin: Optional[float]
    working_capital: Optional[float]
    quick_ratio: Optional[float]
    customer_acquisition_cost: Optional[float]
    lifetime_value: Optional[float]
    
    # Company information
    founded_date: Optional[str]
    total_employees: Optional[float]  # Using float to handle "1,600+" type values
    investors: Optional[str]
    management: Optional[str]
    headquarters: Optional[str]
    industry: Optional[str]
    
    # Market metrics
    market_share: Optional[float]
    tam: Optional[float]  # Total Addressable Market
    competitor_count: Optional[float]

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
index_name = "financial-docs"

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def get_vectorstore(namespace: str):
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )

def extract_financial_metrics(documents, llm):
    template = """Extract the following financial and company metrics from the documents. Return each metric on a new line in the format 'Metric Name: Value'. Return NA if not found or null (dont mention the reason).
    
    Financial Metrics:
    - Last 12 Months Revenue
    - Revenue Growth
    - Burn Rate
    - Existing Debt
    - Total Raised
    - Last Raised
    - EBITDA Margin
    - Total Funding
    - Latest Round
    - Cash
    - LTM Gross Margin
    - Working Capital
    - Quick Ratio
    - Customer Acquisition Cost
    - Customer Lifetime Value
    
    Company Information:
    - Founded Date
    - Total Employees
    - Investors
    - Management
    - Headquarters
    - Industry
    
    Market Position:
    - Market Share
    - Total Addressable Market
    - Number of Competitors
    
    Document content: {content}
    """
    print("Documents:", documents[0].page_content)
    combined_content = ""
    for i in range(len(documents)):
        combined_content += documents[i].page_content
        
    print("Combined content length:", len(combined_content))
    print("-----------------------------------")
    response = llm.chat.completions.create(
        model="llama-3.2-3b-instruct",
        messages=[
            {
                "role": "system",
                "content": template.format(content=combined_content)
            },
            {
                "role": "user",
                "content": "Please extract the metrics from the document."
            }
        ]
    )
    print("-----------------------------------")
    
    return parse_metrics(response.choices[0].message.content)

def parse_metrics(response_text: str) -> FinancialMetrics:
    def clean_number(value: str) -> Optional[float]:
        if not value or value.upper() == 'NA':
            return None
        
        value = re.sub(r'[$£€¥\s,]', '', value)
        
        if '%' in value:
            try:
                return float(value.replace('%', '')) / 100
            except ValueError:
                return None
        
        multiplier = 1
        if value.upper().endswith('M'):
            multiplier = 1_000_000
            value = value[:-1]
        elif value.upper().endswith('B'):
            multiplier = 1_000_000_000
            value = value[:-1]
        
        try:
            return float(value) * multiplier
        except ValueError:
            return None

    def clean_text(value: str) -> Optional[str]:
        if not value or value.upper() == 'NA':
            return None
        return value.strip()

    patterns = {
        'last_12m_revenue': r'Last 12 Months Revenue:?\s*([^\n]+)',
        'revenue_growth': r'Revenue Growth:?\s*([^\n]+)',
        'burn_rate': r'Burn Rate:?\s*([^\n]+)',
        'existing_debt': r'Existing Debt:?\s*([^\n]+)',
        'total_raised': r'Total Raised:?\s*([^\n]+)',
        'last_raised': r'Last Raised:?\s*([^\n]+)',
        'ebitda_margin': r'EBITDA Margin:?\s*([^\n]+)',
        'total_funding': r'Total Funding:?\s*([^\n]+)',
        'latest_round': r'Latest Round:?\s*([^\n]+)',
        'cash': r'Cash:?\s*([^\n]+)',
        'ltm_gross_margin': r'LTM Gross Margin:?\s*([^\n]+)',
        'working_capital': r'Working Capital:?\s*([^\n]+)',
        'quick_ratio': r'Quick Ratio:?\s*([^\n]+)',
        'customer_acquisition_cost': r'Customer Acquisition Cost:?\s*([^\n]+)',
        'lifetime_value': r'Customer Lifetime Value:?\s*([^\n]+)',
        'founded_date': r'Founded Date:?\s*([^\n]+)',
        'total_employees': r'Total Employees:?\s*([^\n]+)',
        'investors': r'Investors:?\s*([^\n]+)',
        'management': r'Management:?\s*([^\n]+)',
        'headquarters': r'Headquarters:?\s*([^\n]+)',
        'industry': r'Industry:?\s*([^\n]+)',
        'market_share': r'Market Share:?\s*([^\n]+)',
        'tam': r'Total Addressable Market:?\s*([^\n]+)',
        'competitor_count': r'Number of Competitors:?\s*([^\n]+)'
    }
    
    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE)
        value = match.group(1).strip() if match else 'NA'
        
        if key in ['investors', 'management', 'headquarters', 'industry', 'founded_date', 'latest_round']:
            metrics[key] = clean_text(value)
        else:
            metrics[key] = clean_number(value)
    
    return FinancialMetrics(**metrics)

def generate_insights(metrics: FinancialMetrics) -> List[str]:
    insights = []
    
    if metrics.last_12m_revenue is not None and metrics.revenue_growth is not None:
        if metrics.revenue_growth > 0.5:
            insights.append("Strong revenue growth indicates market validation and scaling success")
        elif metrics.revenue_growth < 0:
            insights.append("Declining revenue suggests need for strategic review")
    
    if metrics.burn_rate is not None and metrics.total_raised is not None:
        runway = metrics.total_raised / metrics.burn_rate if metrics.burn_rate > 0 else float('inf')
        if runway < 12:
            insights.append(f"Current runway of {runway:.1f} months suggests need for fundraising planning")
        elif runway > 24:
            insights.append("Healthy runway provides operational flexibility")
    
    # Add more insights based on other metrics...
    
    return insights

async def generate_dashboard(request: Dict[str, List[str]], namespace: str):
    try:
        # Get namespace-specific vectorstore
        vectorstore = get_vectorstore(namespace)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
        documents = []
        
        for topic in request.get('topics', []):
            docs = retriever.get_relevant_documents(topic)
            documents.extend(docs)
        
        print(f"Documents retrieved for namespace {namespace}", documents)
        metrics = extract_financial_metrics(documents, llm)
        print(f"Metrics extracted for namespace {namespace}", metrics)
        insights = generate_insights(metrics)
        print(f"Insights generated for namespace {namespace}", insights)
        
        return {
            "metrics": metrics.dict(),
            "insights": insights,
            "status": "success",
            "namespace": namespace
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating dashboard: {str(e)}"
        )
  
  
        
def generate_report_content(topics, documents, llm):
    prompt = """Generate a detailed financial report covering: {topics}
    Make it structured with sections. Include specific numbers and insights.
    
    Context from documents: {context}
    """
    
    context = ""
    print("Generating report content...")
    for i in range(len(documents)):
        context += documents[i].page_content
    print("Context length:", len(context))
    print("Context:", context[:200])
    print("Topics:", topics)
    print("Generating report...")
    print("Prompt:", prompt.format(topics=topics, context=context))
    response = llm.chat.completions.create(
        model="llama-3.2-3b-instruct",
        messages=[
            {
                "role": "system",
                "content": prompt.format(topics=topics, context=context)
            },
            {"role": "user", "content": ""}
        ]
    )
    print("Report content generated.", response)
    return response

def classify_document(content: str, llm: OpenAI) -> str:
    print("Classifying document content:", content)
    prompt = """Classify this financial document into one of these categories:
    - Balance Sheet
    - Income Statement
    - Cash Flow Statement
    - Tax Return
    - Invoice
    - Bank Statement
    - Financial Report
    - Other

    Document content: {content}
    
    Return only the category name.
    """
    
    response = llm.chat.completions.create(
        model="lmstudio-community/llama-3.2-3b-instruct",
        messages=[
            {"role": "system", "content": prompt.format(content=content)},
            {"role": "user", "content": ""}
        ]
    )
    return response.choices[0].message.content



