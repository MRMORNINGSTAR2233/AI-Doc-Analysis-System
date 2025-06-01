# Sample Processing Outputs

This document shows examples of the outputs produced by the AI Document Analysis System for different document types.

## Invoice Processing

### Input Document
A PDF invoice from Acme Corp for office supplies.

### Classification Output
```json
{
  "format": "pdf",
  "type": "invoice",
  "confidence": 0.799,
  "intent": {
    "type": "payment_request",
    "confidence": 0.87,
    "keywords": ["invoice", "payment", "due date", "amount due"],
    "reasoning": "Document contains structured payment information including invoice number, due date, and itemized list with total amount due. Format and layout match typical invoice patterns."
  },
  "routing": {
    "suggested_department": "Accounts Payable",
    "priority": "medium"
  }
}
```

### Extraction Output
```json
{
  "invoice_number": "INV-2023-0042",
  "date": "2025-05-15",
  "due_date": "2025-06-15",
  "vendor": {
    "name": "Acme Corp",
    "address": "123 Business St, Commerce City, CA 90210",
    "tax_id": "12-3456789"
  },
  "customer": {
    "name": "Tech Solutions Inc",
    "address": "456 Innovation Ave, Tech Park, NY 10001",
    "customer_id": "TECH-001"
  },
  "items": [
    {
      "description": "Office Chair - Ergonomic Model",
      "quantity": 5,
      "unit_price": 199.99,
      "total": 999.95
    },
    {
      "description": "Standing Desk - Adjustable",
      "quantity": 3,
      "unit_price": 349.99,
      "total": 1049.97
    },
    {
      "description": "Monitor Arms - Dual Setup",
      "quantity": 5,
      "unit_price": 129.99,
      "total": 649.95
    }
  ],
  "subtotal": 2699.87,
  "tax_rate": 0.0875,
  "tax_amount": 236.24,
  "total_amount": 2936.11,
  "payment_terms": "Net 30",
  "payment_instructions": {
    "bank_name": "Commerce Bank",
    "account_number": "XXXX-XXXX-7890",
    "routing_number": "XXXXX0123"
  }
}
```

### Analysis Output
```json
{
  "content_analysis": {
    "length": 2345,
    "readability": {
      "complexity": "low",
      "score": 8.4
    },
    "entities": {
      "dates": ["2025-05-15", "2025-06-15"],
      "amounts": ["$199.99", "$349.99", "$129.99", "$2699.87", "$236.24", "$2936.11"],
      "emails": ["accounts@acmecorp.com", "billing@techsolutions.com"]
    }
  },
  "recommendations": [
    {
      "action": "Schedule payment for June 10, 2025",
      "reason": "Payment due on June 15, 2025 with Net 30 terms",
      "priority": "medium"
    },
    {
      "action": "Verify tax calculation",
      "reason": "Tax rate (8.75%) appears standard but should be verified against local rates",
      "priority": "low"
    }
  ],
  "validation": {
    "is_valid": true,
    "warnings": []
  }
}
```

### Actions Determined
```json
[
  {
    "type": "schedule_payment",
    "description": "Schedule payment for invoice INV-2023-0042",
    "priority": "medium",
    "params": {
      "amount": 2936.11,
      "due_date": "2025-06-15",
      "vendor": "Acme Corp",
      "payment_date": "2025-06-10",
      "account": "Operations Budget"
    }
  },
  {
    "type": "update_accounting",
    "description": "Record expense in accounting system",
    "priority": "medium",
    "params": {
      "category": "Office Equipment",
      "amount": 2936.11,
      "tax_amount": 236.24,
      "date": "2025-05-15"
    }
  }
]
```

## Email Processing

### Input Document
An email requesting information about product pricing.

### Classification Output
```json
{
  "format": "email",
  "type": "inquiry",
  "confidence": 0.92,
  "intent": {
    "type": "information_request",
    "confidence": 0.85,
    "keywords": ["pricing", "information", "products", "catalog"],
    "reasoning": "Email contains direct questions about product pricing and availability. Sender is requesting specific information rather than placing an order or reporting an issue."
  },
  "routing": {
    "suggested_department": "Sales",
    "priority": "normal"
  }
}
```

### Extraction Output
```json
{
  "metadata": {
    "from": "john.customer@example.com",
    "to": "sales@techsolutions.com",
    "subject": "Product Pricing Information Request",
    "date": "2025-05-30T14:23:45Z"
  },
  "sender": {
    "name": "John Customer",
    "email": "john.customer@example.com",
    "company": "Customer Corp"
  },
  "content": {
    "greeting": "Hello Sales Team,",
    "body": "I'm interested in your enterprise software solutions, particularly the data analytics package. Could you please send me information about your pricing tiers and available features? We're looking to implement a solution for approximately 50 users by the end of Q3.\n\nAlso, do you offer any volume discounts or academic pricing?\n\nThank you for your assistance.",
    "signature": "Best regards,\nJohn Customer\nIT Director\nCustomer Corp\n(555) 123-4567"
  },
  "products_mentioned": ["data analytics package"],
  "questions": [
    "pricing tiers and available features",
    "volume discounts",
    "academic pricing"
  ]
}
```

### Analysis Output
```json
{
  "content_analysis": {
    "length": 876,
    "sentiment": "neutral",
    "urgency": "medium",
    "readability": {
      "complexity": "medium",
      "score": 9.2
    },
    "entities": {
      "people": ["John Customer"],
      "organizations": ["Customer Corp"],
      "products": ["data analytics package"],
      "contact_info": ["john.customer@example.com", "(555) 123-4567"]
    }
  },
  "recommendations": [
    {
      "action": "Send product brochure and pricing information",
      "reason": "Customer has explicitly requested pricing information",
      "priority": "high"
    },
    {
      "action": "Follow up with sales call",
      "reason": "Potential enterprise customer with specific timeline (end of Q3)",
      "priority": "medium"
    },
    {
      "action": "Add to CRM system",
      "reason": "New lead with contact information",
      "priority": "medium"
    }
  ],
  "validation": {
    "is_valid": true,
    "warnings": []
  }
}
```

## Processing Statistics

| Document Type | Average Processing Time | Classification Accuracy | Extraction Accuracy |
|---------------|-------------------------|------------------------|---------------------|
| Invoices      | 9.8 seconds             | 92%                    | 89%                 |
| Emails        | 7.2 seconds             | 95%                    | 87%                 |
| Contracts     | 15.6 seconds            | 88%                    | 82%                 |
| Reports       | 12.3 seconds            | 90%                    | 85%                 |

## System Performance

- **Throughput**: ~350 documents/hour
- **Average Response Time**: 8.7 seconds
- **Error Rate**: 2.3%
- **Success Rate**: 97.7% 