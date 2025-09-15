# services/employee_service.py
from database.models import Employee, VectorFace
from database.database import DatabaseManager
from sqlalchemy.orm import joinedload

class EmployeeService:
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def create_employee(self, name, employee_code, department=None, position=None, email=None, phone=None):
        session = self.db_manager.get_session()
        try:
            employee = Employee(
                name=name,
                employee_code=employee_code,
                department=department,
                position=position,
                email=email,
                phone=phone
            )
            session.add(employee)
            session.commit()
            session.refresh(employee)
            return employee
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def get_employee(self, employee_id):
        session = self.db_manager.get_session()
        try:
            return session.query(Employee).options(joinedload(Employee.face_vectors)).filter_by(id=employee_id).first()
        finally:
            self.db_manager.close_session(session)
    
    def get_all_employees(self):
        session = self.db_manager.get_session()
        try:
            return session.query(Employee).options(joinedload(Employee.face_vectors)).all()
        finally:
            self.db_manager.close_session(session)
    
    def update_employee(self, employee_id, **kwargs):
        session = self.db_manager.get_session()
        try:
            employee = session.query(Employee).filter_by(id=employee_id).first()
            if employee:
                for key, value in kwargs.items():
                    if hasattr(employee, key):
                        setattr(employee, key, value)
                session.commit()
                session.refresh(employee)
            return employee
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def delete_employee(self, employee_id):
        session = self.db_manager.get_session()
        try:
            employee = session.query(Employee).filter_by(id=employee_id).first()
            if employee:
                session.delete(employee)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def get_employee_image_count(self, employee_id):
        session = self.db_manager.get_session()
        try:
            return session.query(VectorFace).filter_by(employee_id=employee_id).count()
        finally:
            self.db_manager.close_session(session)
