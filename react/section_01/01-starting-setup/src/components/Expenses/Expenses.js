import React, { useState } from 'react';
import ExpenseItem from "./ExpenseItem";
import ExpensesFilter from "./ExpenseFilter";
import Card from "../UI/Card";
import './Expenses.css';

function Expenses(props) {
    const [filterYear, setFilterYear] = useState('2020');

    const expenseFilterHandler = ( filterYear ) => {
        setFilterYear( filterYear );
    }

    return (
        <div>
            <Card className="expenses">
                <ExpensesFilter 
                    filterYear={filterYear} 
                    onExpenseFilter={expenseFilterHandler}
                />
                {props.expenses.filter( 
                    ( obj ) => (
                        obj.date.getFullYear().toString() === filterYear
                    )
                ).map( 
                    ( obj ) => (
                        <ExpenseItem key={obj.id} title={obj.title} amount={obj.amount} date={obj.date} />
                    ) 
                )}
            </Card>
        </div>
    )
}

export default Expenses;
