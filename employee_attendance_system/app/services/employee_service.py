#!/usr/bin/env python3
"""
employee_service.py - EMPLOYEE MANAGEMENT SERVICE
Enhanced employee CRUD operations with optimized database handling
Integrated with new app/ structure
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# Database imports
from ..models.models import Employee, VectorFace, AttendanceLog, DatabaseManager
from ..database.database import db_session_scope, get_database_manager
from ..config import Config

logger = logging.getLogger(__name__)


class EmployeeService:
    """Enhanced employee management service"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager or get_database_manager()
        self.config = Config()
        
        # Performance tracking
        self.stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'last_operation': None
        }
        
        logger.info("✅ EmployeeService initialized")
    
    def create_employee(
        self, 
        name: str, 
        employee_code: str, 
        department: Optional[str] = None,
        position: Optional[str] = None, 
        email: Optional[str] = None, 
        phone: Optional[str] = None
    ) -> Employee:
        """Create new employee with validation"""
        try:
            # Validate required fields
            if not name or not name.strip():
                raise ValueError("Employee name is required")
            
            if not employee_code or not employee_code.strip():
                raise ValueError("Employee code is required")
            
            with db_session_scope() as session:
                # Check if employee code already exists
                existing = session.query(Employee).filter(
                    Employee.employee_code == employee_code.strip()
                ).first()
                
                if existing:
                    raise ValueError(f"Employee code '{employee_code}' already exists")
                
                # Create new employee
                employee = Employee(
                    name=name.strip(),
                    employee_code=employee_code.strip(),
                    department=department.strip() if department else None,
                    position=position.strip() if position else None,
                    email=email.strip() if email else None,
                    phone=phone.strip() if phone else None,
                    is_active=True
                )
                
                session.add(employee)
                session.commit()
                session.refresh(employee)
                
                # Update stats
                self.stats['total_operations'] += 1
                self.stats['successful_operations'] += 1
                self.stats['last_operation'] = datetime.now()
                
                logger.info(f"✅ Created employee: {employee.name} (ID: {employee.id})")
                return employee
                
        except Exception as e:
            self.stats['total_operations'] += 1
            self.stats['failed_operations'] += 1
            logger.error(f"❌ Error creating employee: {e}")
            raise
    
    def get_employee_by_id(self, employee_id: int) -> Optional[Employee]:
        """Get employee by ID with relationships loaded"""
        try:
            with db_session_scope() as session:
                employee = session.query(Employee).filter(
                    Employee.id == employee_id
                ).first()
                
                if employee:
                    # Create detached copy with basic info
                    return self._create_employee_copy(employee)
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting employee {employee_id}: {e}")
            return None
    
    def get_employee_by_code(self, employee_code: str) -> Optional[Employee]:
        """Get employee by employee code"""
        try:
            with db_session_scope() as session:
                employee = session.query(Employee).filter(
                    Employee.employee_code == employee_code.strip()
                ).first()
                
                if employee:
                    return self._create_employee_copy(employee)
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting employee by code {employee_code}: {e}")
            return None
    
    def get_all_employees(self, include_inactive: bool = False) -> List[Employee]:
        """Get all employees with optional inactive filter"""
        try:
            with db_session_scope() as session:
                query = session.query(Employee)
                
                if not include_inactive:
                    query = query.filter(Employee.is_active == True)
                
                employees = query.order_by(Employee.name).all()
                
                # Return detached copies
                return [self._create_employee_copy(emp) for emp in employees]
                
        except Exception as e:
            logger.error(f"❌ Error getting all employees: {e}")
            return []
    
    def get_employees_by_department(self, department: str) -> List[Employee]:
        """Get employees by department"""
        try:
            with db_session_scope() as session:
                employees = session.query(Employee).filter(
                    Employee.department == department,
                    Employee.is_active == True
                ).order_by(Employee.name).all()
                
                return [self._create_employee_copy(emp) for emp in employees]
                
        except Exception as e:
            logger.error(f"❌ Error getting employees by department {department}: {e}")
            return []
    
    def search_employees(self, search_term: str) -> List[Employee]:
        """Search employees by name, code, or email"""
        if not search_term or not search_term.strip():
            return []
        
        try:
            search_pattern = f"%{search_term.strip()}%"
            
            with db_session_scope() as session:
                employees = session.query(Employee).filter(
                    (Employee.name.ilike(search_pattern) |
                     Employee.employee_code.ilike(search_pattern) |
                     Employee.email.ilike(search_pattern)),
                    Employee.is_active == True
                ).order_by(Employee.name).all()
                
                return [self._create_employee_copy(emp) for emp in employees]
                
        except Exception as e:
            logger.error(f"❌ Error searching employees: {e}")
            return []
    
    def update_employee(
        self, 
        employee_id: int, 
        name: Optional[str] = None,
        employee_code: Optional[str] = None,
        department: Optional[str] = None,
        position: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> Optional[Employee]:
        """Update employee information"""
        try:
            with db_session_scope() as session:
                employee = session.query(Employee).filter(
                    Employee.id == employee_id
                ).first()
                
                if not employee:
                    raise ValueError(f"Employee with ID {employee_id} not found")
                
                # Check if employee code is being changed and if it conflicts
                if employee_code and employee_code.strip() != employee.employee_code:
                    existing = session.query(Employee).filter(
                        Employee.employee_code == employee_code.strip(),
                        Employee.id != employee_id
                    ).first()
                    
                    if existing:
                        raise ValueError(f"Employee code '{employee_code}' already exists")
                
                # Update fields
                if name is not None:
                    employee.name = name.strip()
                if employee_code is not None:
                    employee.employee_code = employee_code.strip()
                if department is not None:
                    employee.department = department.strip() if department else None
                if position is not None:
                    employee.position = position.strip() if position else None
                if email is not None:
                    employee.email = email.strip() if email else None
                if phone is not None:
                    employee.phone = phone.strip() if phone else None
                if is_active is not None:
                    employee.is_active = is_active
                
                employee.updated_at = datetime.utcnow()
                
                session.commit()
                session.refresh(employee)
                
                # Update stats
                self.stats['total_operations'] += 1
                self.stats['successful_operations'] += 1
                self.stats['last_operation'] = datetime.now()
                
                logger.info(f"✅ Updated employee: {employee.name} (ID: {employee.id})")
                return self._create_employee_copy(employee)
                
        except Exception as e:
            self.stats['total_operations'] += 1
            self.stats['failed_operations'] += 1
            logger.error(f"❌ Error updating employee {employee_id}: {e}")
            raise
    
    def delete_employee(self, employee_id: int, hard_delete: bool = False) -> bool:
        """Delete employee (soft delete by default)"""
        try:
            with db_session_scope() as session:
                employee = session.query(Employee).filter(
                    Employee.id == employee_id
                ).first()
                
                if not employee:
                    logger.warning(f"Employee with ID {employee_id} not found")
                    return False
                
                employee_name = employee.name
                
                if hard_delete:
                    # Hard delete - remove from database completely
                    session.delete(employee)
                    logger.info(f"✅ Hard deleted employee: {employee_name} (ID: {employee_id})")
                else:
                    # Soft delete - mark as inactive
                    employee.is_active = False
                    employee.updated_at = datetime.utcnow()
                    logger.info(f"✅ Soft deleted employee: {employee_name} (ID: {employee_id})")
                
                session.commit()
                
                # Update stats
                self.stats['total_operations'] += 1
                self.stats['successful_operations'] += 1
                self.stats['last_operation'] = datetime.now()
                
                return True
                
        except Exception as e:
            self.stats['total_operations'] += 1
            self.stats['failed_operations'] += 1
            logger.error(f"❌ Error deleting employee {employee_id}: {e}")
            return False
    
    def restore_employee(self, employee_id: int) -> bool:
        """Restore soft-deleted employee"""
        try:
            with db_session_scope() as session:
                employee = session.query(Employee).filter(
                    Employee.id == employee_id
                ).first()
                
                if not employee:
                    return False
                
                employee.is_active = True
                employee.updated_at = datetime.utcnow()
                
                session.commit()
                
                logger.info(f"✅ Restored employee: {employee.name} (ID: {employee_id})")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error restoring employee {employee_id}: {e}")
            return False
    
    def get_employee_statistics(self, employee_id: int) -> Dict[str, Any]:
        """Get detailed statistics for employee"""
        try:
            with db_session_scope() as session:
                employee = session.query(Employee).filter(
                    Employee.id == employee_id
                ).first()
                
                if not employee:
                    return {'error': 'Employee not found'}
                
                # Count face vectors
                face_vector_count = session.query(VectorFace).filter(
                    VectorFace.employee_id == employee_id
                ).count()
                
                # Count attendance logs
                attendance_count = session.query(AttendanceLog).filter(
                    AttendanceLog.employee_id == employee_id
                ).count()
                
                # Get recent attendance
                recent_attendance = session.query(AttendanceLog).filter(
                    AttendanceLog.employee_id == employee_id
                ).order_by(AttendanceLog.check_in_time.desc()).limit(5).all()
                
                # Get face vector details
                face_vectors = session.query(VectorFace).filter(
                    VectorFace.employee_id == employee_id
                ).order_by(VectorFace.created_at.desc()).all()
                
                return {
                    'employee_id': employee_id,
                    'employee_name': employee.name,
                    'employee_code': employee.employee_code,
                    'department': employee.department,
                    'position': employee.position,
                    'is_active': employee.is_active,
                    'created_at': employee.created_at.isoformat() if employee.created_at else None,
                    'face_vector_count': face_vector_count,
                    'attendance_count': attendance_count,
                    'recent_attendance': [
                        {
                            'id': log.id,
                            'check_in_time': log.check_in_time.isoformat() if log.check_in_time else None,
                            'confidence_score': log.confidence_score,
                            'detection_method': log.detection_method
                        }
                        for log in recent_attendance
                    ],
                    'face_vectors': [
                        {
                            'id': vec.id,
                            'created_at': vec.created_at.isoformat() if vec.created_at else None,
                            'model_name': vec.model_name,
                            'confidence_score': vec.confidence_score
                        }
                        for vec in face_vectors
                    ]
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting employee statistics {employee_id}: {e}")
            return {'error': str(e)}
    
    def get_department_statistics(self) -> Dict[str, Any]:
        """Get statistics by department"""
        try:
            with db_session_scope() as session:
                # Get department counts
                dept_query = session.query(
                    Employee.department,
                    session.query(Employee.id).filter(
                        Employee.department == Employee.department,
                        Employee.is_active == True
                    ).count().label('count')
                ).filter(
                    Employee.is_active == True
                ).group_by(Employee.department).all()
                
                departments = {}
                total_active = 0
                
                for dept, count in dept_query:
                    dept_name = dept or 'No Department'
                    departments[dept_name] = count
                    total_active += count
                
                # Get total employees (including inactive)
                total_employees = session.query(Employee).count()
                inactive_employees = total_employees - total_active
                
                return {
                    'total_employees': total_employees,
                    'active_employees': total_active,
                    'inactive_employees': inactive_employees,
                    'departments': departments,
                    'department_count': len(departments)
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting department statistics: {e}")
            return {'error': str(e)}
    
    def _create_employee_copy(self, employee: Employee) -> Employee:
        """Create detached copy of employee for safe return"""
        # Create a simple object with employee data
        employee_copy = type('Employee', (), {})()
        employee_copy.id = employee.id
        employee_copy.name = employee.name
        employee_copy.employee_code = employee.employee_code
        employee_copy.department = employee.department
        employee_copy.position = employee.position
        employee_copy.email = employee.email
        employee_copy.phone = employee.phone
        employee_copy.is_active = employee.is_active
        employee_copy.created_at = employee.created_at
        employee_copy.updated_at = employee.updated_at
        
        return employee_copy
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get employee service statistics"""
        try:
            with db_session_scope() as session:
                total_employees = session.query(Employee).count()
                active_employees = session.query(Employee).filter(
                    Employee.is_active == True
                ).count()
                
                # Get department distribution
                departments = session.query(Employee.department).distinct().all()
                dept_list = [dept[0] for dept in departments if dept[0]]
                
                return {
                    'total_employees': total_employees,
                    'active_employees': active_employees,
                    'inactive_employees': total_employees - active_employees,
                    'departments': dept_list,
                    'department_count': len(dept_list),
                    'operation_stats': self.stats,
                    'service_status': 'healthy'
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting service stats: {e}")
            return {
                'service_status': 'unhealthy',
                'error': str(e),
                'operation_stats': self.stats
            }
    
    def bulk_import_employees(self, employees_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk import employees from data list"""
        successful = 0
        failed = 0
        errors = []
        
        try:
            with db_session_scope() as session:
                for i, emp_data in enumerate(employees_data):
                    try:
                        # Validate required fields
                        if not emp_data.get('name') or not emp_data.get('employee_code'):
                            errors.append(f"Row {i+1}: Name and employee code are required")
                            failed += 1
                            continue
                        
                        # Check for duplicates
                        existing = session.query(Employee).filter(
                            Employee.employee_code == emp_data['employee_code'].strip()
                        ).first()
                        
                        if existing:
                            errors.append(f"Row {i+1}: Employee code '{emp_data['employee_code']}' already exists")
                            failed += 1
                            continue
                        
                        # Create employee
                        employee = Employee(
                            name=emp_data['name'].strip(),
                            employee_code=emp_data['employee_code'].strip(),
                            department=emp_data.get('department', '').strip() or None,
                            position=emp_data.get('position', '').strip() or None,
                            email=emp_data.get('email', '').strip() or None,
                            phone=emp_data.get('phone', '').strip() or None,
                            is_active=emp_data.get('is_active', True)
                        )
                        
                        session.add(employee)
                        successful += 1
                        
                    except Exception as e:
                        errors.append(f"Row {i+1}: {str(e)}")
                        failed += 1
                
                # Commit all successful imports
                if successful > 0:
                    session.commit()
                
                logger.info(f"✅ Bulk import completed: {successful} successful, {failed} failed")
                
                return {
                    'successful': successful,
                    'failed': failed,
                    'total': len(employees_data),
                    'errors': errors
                }
                
        except Exception as e:
            logger.error(f"❌ Bulk import error: {e}")
            return {
                'successful': 0,
                'failed': len(employees_data),
                'total': len(employees_data),
                'errors': [f"Bulk import failed: {str(e)}"]
            }