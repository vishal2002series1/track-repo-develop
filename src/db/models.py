# src/db/models.py
from sqlalchemy import Column, String, Text, Table, ForeignKey, JSON
from sqlalchemy.orm import relationship
from src.db.database import Base

# Association Table for Many-to-Many mapping
workflow_agent_association = Table(
    'workflow_agent_map',
    Base.metadata,
    Column('workflow_id', String, ForeignKey('workflows.id'), primary_key=True),
    Column('agent_id', String, ForeignKey('domain_agents.id'), primary_key=True)
)

class DomainAgent(Base):
    __tablename__ = 'domain_agents'

    id = Column(String, primary_key=True, index=True) # e.g., 'crm_activities_domain_agent'
    name = Column(String, nullable=False)
    routing_description = Column(Text, nullable=False)
    persona = Column(Text, nullable=False)
    authorized_tools = Column(JSON, nullable=False) # e.g., ["execute_sql", "search_transcripts"]

    # Links back to workflows
    workflows = relationship("Workflow", secondary=workflow_agent_association, back_populates="agents")

class Workflow(Base):
    __tablename__ = 'workflows'

    id = Column(String, primary_key=True, index=True) # e.g., 'WF_003'
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)

    # Links to agents
    agents = relationship("DomainAgent", secondary=workflow_agent_association, back_populates="workflows")