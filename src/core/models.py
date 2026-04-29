# src/core/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, Date, Boolean
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class AdvisorDetails(Base):
    __tablename__ = 'AdvisorDetails'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    AdvisorName = Column(String(255), nullable=False)
    Title = Column(String(100))
    
    clients = relationship("AdvisorClient", back_populates="advisor")
    performance = relationship("AdvisorPerformance", back_populates="advisor", uselist=False)

class AdvisorPerformance(Base):
    __tablename__ = 'AdvisorPerformance'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    AdvisorId = Column(Integer, ForeignKey('AdvisorDetails.Id'))
    TotalAUM = Column(Float)
    Revenue = Column(Float)
    GrowthYTD = Column(Float)
    
    advisor = relationship("AdvisorDetails", back_populates="performance")

class ClientDetails(Base):
    __tablename__ = 'ClientDetails'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    ClientName = Column(String(255), nullable=False)
    RiskProfile = Column(String(50))
    MaritalStatus = Column(String(50))
    Segment = Column(String(50))
    Age = Column(Integer)
    FinancialGoals = Column(Text)
    UpcomingLifeEvents = Column(String(255))
    ClientSentiment = Column(String(50)) 
    AttritionRisk = Column(String(50))
    ClientNPS = Column(Integer)
    InvestorType = Column(String(100))
    TaxSensitivity = Column(String(50))
    Household = Column(String(100))
    NextAppointment = Column(Date)
    OccupationInfo = Column(String(255))
    ClientTenure = Column(Float) 
    Interests = Column(Text)
    
    advisors = relationship("AdvisorClient", back_populates="client")
    meetings = relationship("UpcomingClientMeetings", back_populates="client")
    transcripts = relationship("TranscriptSummary", back_populates="client")
    compliance = relationship("ComplianceHub", back_populates="client")
    portfolio = relationship("PortfolioData", back_populates="client")
    planning_facts = relationship("FinancialPlanningFacts", back_populates="client", uselist=False)

class AdvisorClient(Base):
    __tablename__ = 'AdvisorClient'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    AdvisorId = Column(Integer, ForeignKey('AdvisorDetails.Id'))
    ClientId = Column(Integer, ForeignKey('ClientDetails.Id'))
    WalletShare = Column(Float)
    
    advisor = relationship("AdvisorDetails", back_populates="clients")
    client = relationship("ClientDetails", back_populates="advisors")

class PortfolioData(Base):
    __tablename__ = 'PortfolioData'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    ClientId = Column(Integer, ForeignKey('ClientDetails.Id'))
    Ticker = Column(String(20))
    AssetName = Column(String(255))
    AssetClass = Column(String(100))
    MarketValue = Column(Float)
    CostBasis = Column(Float)
    
    client = relationship("ClientDetails", back_populates="portfolio")

class FinancialPlanningFacts(Base):
    __tablename__ = 'FinancialPlanningFacts'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    ClientId = Column(Integer, ForeignKey('ClientDetails.Id'))
    LastPlanUpdate = Column(Date)
    EstatePlanDate = Column(Date)
    HasRMD = Column(Boolean, default=False)
    
    client = relationship("ClientDetails", back_populates="planning_facts")

class ComplianceHub(Base):
    __tablename__ = 'ComplianceHub'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    ClientId = Column(Integer, ForeignKey('ClientDetails.Id'))
    FlagType = Column(String(100)) 
    Description = Column(Text)
    IsResolved = Column(Boolean, default=False)
    
    client = relationship("ClientDetails", back_populates="compliance")

class UpcomingClientMeetings(Base):
    __tablename__ = 'UpcomingClientMeetings'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    ClientId = Column(Integer, ForeignKey('ClientDetails.Id'))
    MeetingType = Column(String(100))
    MeetingDate = Column(Date)
    AgendaDrafted = Column(Boolean, default=False)
    
    client = relationship("ClientDetails", back_populates="meetings")

class TranscriptSummary(Base):
    __tablename__ = 'TranscriptSummary'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    ClientId = Column(Integer, ForeignKey('ClientDetails.Id'))
    TranscriptId = Column(String(100), unique=True)
    InteractionDate = Column(Date) # ADDED: Gap #9 "Time since last contact" signal
    Summary = Column(Text)
    
    client = relationship("ClientDetails", back_populates="transcripts")

class Email(Base):
    __tablename__ = 'Email'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    ClientId = Column(Integer, ForeignKey('ClientDetails.Id'))
    DateSent = Column(Date)
    Subject = Column(String(255))
    Body = Column(Text)

class EmailReply(Base): # ADDED: Gap #16 Un-merged from Email
    __tablename__ = 'EmailReply'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    EmailId = Column(Integer, ForeignKey('Email.Id'))
    DateSent = Column(Date)
    Body = Column(Text)
    FromAdvisor = Column(Boolean, default=True)

class EmailInsight(Base):
    __tablename__ = 'EmailInsight'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    EmailId = Column(Integer, ForeignKey('Email.Id'))
    ClientId = Column(Integer, ForeignKey('ClientDetails.Id')) # ADDED: Gap #8 Denormalized for fast aggregation
    Sentiment = Column(String(50)) # FIXED: Gap #2 String categorical per Excel 
    RequiresAction = Column(Boolean, default=False)
    ExtractedThemes = Column(Text)
    Participants = Column(String(255)) # ADDED: Gap #2 Inventory requirement
    ProductsMentioned = Column(String(255)) # ADDED: Gap #2 Inventory requirement
    Summary = Column(Text) # ADDED: Gap #2 Inventory requirement

class TranscriptInsights(Base):
    __tablename__ = 'TranscriptInsights'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    TranscriptId = Column(String(100), ForeignKey('TranscriptSummary.TranscriptId'))
    OverallSentiment = Column(String(50))
    ActionItems = Column(Text)
    KeyLifeEventsMentioned = Column(Text)

class NextBestAction(Base):
    __tablename__ = 'NextBestAction'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    ClientId = Column(Integer, ForeignKey('ClientDetails.Id'))
    ActionCategory = Column(String(100)) 
    Recommendation = Column(Text)
    ConfidenceScore = Column(Float)

class MarketHighlights(Base):
    __tablename__ = 'MarketHighlights'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    Date = Column(Date)
    AssetClass = Column(String(100))
    HighlightText = Column(Text)
    MarketImpact = Column(String(50))

class PolicyBenchmark(Base): # ADDED: Gap #4 For Performance Benchmark Agent
    __tablename__ = 'PolicyBenchmark'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    BenchmarkName = Column(String(100)) 
    AssetClass = Column(String(100))
    ReturnYTD = Column(Float)
    Return1Y = Column(Float)
    Return3Y = Column(Float)