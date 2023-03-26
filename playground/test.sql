select t1.col_int32,t2.col_int32 from test as t1 join test as t2 on t1.col_int32 = t2.col_int32
where t1.col_ts_nano_none is null
