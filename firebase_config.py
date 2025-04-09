import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
import numpy as np

# Initialize Firebase Admin
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

# Constants based on dataset statistics
DEFAULT_WAIT_TIME = 17  # Dataset mean wait time
DEFAULT_STD_DEV = 7  # Dataset standard deviation

def generate_synthetic_wait_times(size=5, mean=DEFAULT_WAIT_TIME, std=DEFAULT_STD_DEV):
    """Generate synthetic wait times using normal distribution based on dataset statistics"""
    times = np.random.normal(mean, std, size)
    return [max(5, float(t)) for t in times]  # Min of 5 minutes to avoid unrealistic times

def check_branch_exists(branch_id):
    """Check if branch exists in Firestore using the 'id' field"""
    try:
        # Query branches where 'id' field matches branch_id
        query = db.collection('branches').where('id', '==', branch_id).limit(1)
        docs = query.get()
        return len(docs) > 0
    except Exception as e:
        print(f"Error checking branch existence: {str(e)}")
        return False

def get_branch_wait_times(branch_id, limit=5):
    """
    Fetch most recent wait times for a specific branch
    Returns exactly 'limit' number of wait times (uses synthetic data if needed)
    """
    # First check if branch exists using id field
    if not check_branch_exists(branch_id):
        raise ValueError(f"Branch with ID {branch_id} does not exist")
    
    bookings_ref = db.collection('bookings')
    
    # Query using where clauses
    query = (bookings_ref
             .where('branchId', '==', branch_id)
             .where('status', '==', 'completed')
             .order_by('completedAt', direction=firestore.Query.DESCENDING)
             .limit(limit))
    
    try:
        docs = query.get()
        wait_times = []
        
        # Extract wait times, calculate if not present
        for doc in docs:
            try:
                doc_data = doc.to_dict()
                # Try to get waitTime directly
                wait_time = doc_data.get('waitTime')
                
                # If waitTime doesn't exist, calculate from timestamps
                if wait_time is None and 'joinedAt' in doc_data and 'completedAt' in doc_data:
                    joined_at = doc_data['joinedAt']
                    completed_at = doc_data['completedAt']
                    if joined_at and completed_at:
                        # Calculate wait time in minutes
                        wait_time = (completed_at.timestamp() - joined_at.timestamp()) / 60
                
                # If we still don't have a valid wait time, use dataset mean
                if wait_time is None or not isinstance(wait_time, (int, float)):
                    wait_time = DEFAULT_WAIT_TIME
                    
                wait_times.append(float(wait_time))
            except Exception as e:
                print(f"Error processing document {doc.id}: {str(e)}")
                wait_times.append(DEFAULT_WAIT_TIME)  # Use dataset mean for problematic documents
        
        # If we don't have enough data, generate synthetic data
        if len(wait_times) < limit:
            # Calculate mean from existing data or use dataset mean
            existing_mean = np.mean(wait_times) if wait_times else DEFAULT_WAIT_TIME
            synthetic_size = limit - len(wait_times)
            synthetic_times = generate_synthetic_wait_times(
                size=synthetic_size, 
                mean=existing_mean,
                std=DEFAULT_STD_DEV
            )
            wait_times.extend(synthetic_times)
            print(f"Added {synthetic_size} synthetic wait times for branch {branch_id}")
            
        return wait_times[:limit]  # Ensure we return exactly 'limit' number of times
        
    except Exception as e:
        print(f"Error fetching branch wait times: {str(e)}")
        # Instead of raising, return synthetic data for complete failure
        print(f"Using fully synthetic data for branch {branch_id}")
        return generate_synthetic_wait_times(size=limit)