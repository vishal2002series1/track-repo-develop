# src/data/generate_mock_data.py
import os
import datetime
import chromadb
from chromadb.utils import embedding_functions
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.models import (
    Base, AdvisorDetails, AdvisorPerformance, ClientDetails, AdvisorClient,
    PortfolioData, FinancialPlanningFacts, ComplianceHub, UpcomingClientMeetings, TranscriptSummary,
    Email, EmailReply, EmailInsight, TranscriptInsights, NextBestAction, MarketHighlights, PolicyBenchmark
)

LOCAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_local'))
SQLITE_DB_PATH = os.path.join(LOCAL_DATA_DIR, 'aeon_mvp.db')
CHROMA_DB_PATH = os.path.join(LOCAL_DATA_DIR, 'chroma_db')

def init_databases():
    engine = create_engine(f'sqlite:///{SQLITE_DB_PATH}')
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    
    try:
        chroma_client.delete_collection(name="advisor_notes")
    except:
        pass
    collection = chroma_client.create_collection(name="advisor_notes", embedding_function=emb_fn)

    return Session(), collection

def populate_data(session, collection):
    print("Populating fully-compliant enterprise synthetic data...")
    today = datetime.date.today()
    
    # --- 1. Advisor & Benchmarks ---
    advisor = AdvisorDetails(AdvisorName="Sarah Jenkins", Title="Senior Wealth Advisor")
    session.add(advisor)
    session.commit()
    session.add(AdvisorPerformance(AdvisorId=advisor.Id, TotalAUM=210000000.0, Revenue=1800000.0, GrowthYTD=4.2))
    
    session.add_all([
        PolicyBenchmark(BenchmarkName="Conservative Target", AssetClass="Mixed", ReturnYTD=2.1, Return1Y=4.5, Return3Y=15.0),
        PolicyBenchmark(BenchmarkName="Growth Target", AssetClass="Mixed", ReturnYTD=5.5, Return1Y=12.0, Return3Y=35.0)
    ])
    
    # --- 2. Clients ---
    c1 = ClientDetails(ClientName="Robert & Susan Johnson", RiskProfile="Moderate", MaritalStatus="Married", Segment="HNW", Age=68, UpcomingLifeEvents="Birth of first grandchild", FinancialGoals="Estate transfer", ClientSentiment="Positive", ClientNPS=8, TaxSensitivity="Medium")
    c2 = ClientDetails(ClientName="Emily Chen", RiskProfile="Aggressive", MaritalStatus="Single", Segment="Emerging Wealth", Age=35, UpcomingLifeEvents="IPO at tech company", FinancialGoals="Aggressive growth, liquidity", ClientSentiment="At Risk", AttritionRisk="High", ClientNPS=4, TaxSensitivity="High", OccupationInfo="Tech Executive")
    c3 = ClientDetails(ClientName="The Miller Family Trust", RiskProfile="Conservative", MaritalStatus="Married", Segment="UHNW", Age=75, UpcomingLifeEvents="Retirement", FinancialGoals="Capital preservation", ClientSentiment="Neutral", ClientNPS=6, TaxSensitivity="Low")
    c4 = ClientDetails(ClientName="David Alverez", RiskProfile="Growth", MaritalStatus="Divorced", Segment="HNW", Age=52, UpcomingLifeEvents="Selling business", FinancialGoals="Business transition, tax planning", ClientSentiment="Frustrated", AttritionRisk="Medium", ClientNPS=5, TaxSensitivity="High")
    
    session.add_all([c1, c2, c3, c4])
    session.commit()

    session.add_all([
        AdvisorClient(AdvisorId=advisor.Id, ClientId=c1.Id, WalletShare=0.8),
        AdvisorClient(AdvisorId=advisor.Id, ClientId=c2.Id, WalletShare=1.0),
        AdvisorClient(AdvisorId=advisor.Id, ClientId=c3.Id, WalletShare=0.6),
        AdvisorClient(AdvisorId=advisor.Id, ClientId=c4.Id, WalletShare=0.4)
    ])

    # --- 3. Planning & Compliance ---
    session.add_all([
        FinancialPlanningFacts(ClientId=c1.Id, LastPlanUpdate=today - datetime.timedelta(days=180), EstatePlanDate=today - datetime.timedelta(days=1800), HasRMD=False),
        ComplianceHub(ClientId=c2.Id, FlagType="Missing Signature", Description="Missing signature on updated IPS document.", IsResolved=False),
        ComplianceHub(ClientId=c2.Id, FlagType="Stale KYC", Description="KYC profile has not been updated in 3 years.", IsResolved=False),
        FinancialPlanningFacts(ClientId=c2.Id, LastPlanUpdate=today - datetime.timedelta(days=30), EstatePlanDate=today - datetime.timedelta(days=100), HasRMD=False),
        FinancialPlanningFacts(ClientId=c3.Id, LastPlanUpdate=today - datetime.timedelta(days=800), EstatePlanDate=today - datetime.timedelta(days=400), HasRMD=True),
        ComplianceHub(ClientId=c4.Id, FlagType="Unsigned Margin Agreement", Description="Margin trading enabled but agreement pending.", IsResolved=False),
    ])

    # --- 4. Portfolio Data ---
    session.add_all([
        PortfolioData(ClientId=c2.Id, Ticker="NVDA", AssetName="Nvidia Corp", AssetClass="Equity", MarketValue=2500000.0, CostBasis=50000.0),
        PortfolioData(ClientId=c2.Id, Ticker="AAPL", AssetName="Apple Inc.", AssetClass="Equity", MarketValue=300000.0, CostBasis=250000.0),
        PortfolioData(ClientId=c1.Id, Ticker="SPY", AssetName="S&P 500 ETF", AssetClass="Equity", MarketValue=1500000.0, CostBasis=1000000.0),
        PortfolioData(ClientId=c1.Id, Ticker="BND", AssetName="Total Bond ETF", AssetClass="Fixed Income", MarketValue=800000.0, CostBasis=820000.0),
    ])

    # --- 5. Meetings & NBA (Fixed Gap #1) ---
    session.add_all([
        UpcomingClientMeetings(ClientId=c1.Id, MeetingType="Estate Plan Review", MeetingDate=today + datetime.timedelta(days=10), AgendaDrafted=False),
        UpcomingClientMeetings(ClientId=c2.Id, MeetingType="Urgent Portfolio Review", MeetingDate=today + datetime.timedelta(days=2), AgendaDrafted=False),
        UpcomingClientMeetings(ClientId=c3.Id, MeetingType="Annual RMD Check-in", MeetingDate=today + datetime.timedelta(days=15), AgendaDrafted=False),
        UpcomingClientMeetings(ClientId=c4.Id, MeetingType="Business Exit Planning", MeetingDate=today + datetime.timedelta(days=7), AgendaDrafted=False),
        NextBestAction(ClientId=c1.Id, ActionCategory="Estate Planning", Recommendation="Initiate 529 plan setup for new grandchild.", ConfidenceScore=0.88),
        NextBestAction(ClientId=c2.Id, ActionCategory="Tax Harvesting", Recommendation="Propose exchange fund to diversify NVDA concentration without triggering immediate capital gains.", ConfidenceScore=0.92),
        NextBestAction(ClientId=c3.Id, ActionCategory="Compliance", Recommendation="Remind client about pending RMD distributions.", ConfidenceScore=0.95),
        NextBestAction(ClientId=c4.Id, ActionCategory="Business Advisory", Recommendation="Schedule intro with commercial banking M&A team.", ConfidenceScore=0.89)
    ])

    # --- 6. Emails, Replies, Insights (Fixed Gap #2, #8, #16) ---
    email1 = Email(ClientId=c2.Id, DateSent=today - datetime.timedelta(days=1), Subject="Urgent: Portfolio Review", Body="Sarah, NVDA is swinging wildly. Please call me today.")
    session.add(email1)
    session.commit()
    
    session.add(EmailReply(EmailId=email1.Id, DateSent=today, Body="Hi Emily, I am reviewing your concentration risk now. Let's talk at 2 PM.", FromAdvisor=True))
    session.add(EmailInsight(
        EmailId=email1.Id, ClientId=c2.Id, Sentiment="Negative", RequiresAction=True, 
        ExtractedThemes="Volatility Anxiety, Concentration Risk", 
        Participants="Emily Chen, Sarah Jenkins", ProductsMentioned="NVDA", 
        Summary="Client is highly anxious about NVDA concentration and requested an urgent call."
    ))

    session.add(MarketHighlights(Date=today, AssetClass="Equity", HighlightText="Semiconductor stocks show intense intraday volatility amid macro tech selloff.", MarketImpact="Bearish"))

    # --- 7. Transcripts & ChromaDB (Fixed Gap #5 & #9) ---
    transcripts = [
        {"client": c1, "id": "TR-001", "date": today - datetime.timedelta(days=30), "text": "Robert noted their new grandchild was born last month. We need to look at setting up a 529 plan.", "sentiment": "Positive"},
        {"client": c2, "id": "TR-002", "date": today - datetime.timedelta(days=14), "text": "Emily expressed anxiety about market volatility affecting her net worth but doesn't want to trigger massive capital gains.", "sentiment": "Negative"},
    ]

    for t in transcripts:
        session.add(TranscriptSummary(ClientId=t['client'].Id, TranscriptId=t['id'], InteractionDate=t['date'], Summary=t['text']))
        session.add(TranscriptInsights(TranscriptId=t['id'], OverallSentiment=t['sentiment'], ActionItems="Review documents", KeyLifeEventsMentioned=t['client'].UpcomingLifeEvents))
        collection.add(
            documents=[t['text']],
            metadatas=[{
                "client_id": t['client'].Id, 
                "client_name": t['client'].ClientName, 
                "sentiment": t['sentiment'],
                "interaction_date": t['date'].isoformat(),
                "themes": "Tax planning" if "capital gains" in t['text'] else "Estate planning"
            }],
            ids=[t['id']]
        )

    session.commit()
    print("Database successfully generated with full coverage for all Enterprise constraints.")

if __name__ == "__main__":
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    db_session, vector_col = init_databases()
    populate_data(db_session, vector_col)
    db_session.close()